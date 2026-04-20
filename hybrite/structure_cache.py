from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import require_mapping, resolve_repo_path
from .io import load_json, save_json

CACHE_FORMAT_VERSION = 1
PROFILES_FILE_NAME = "profiles.pt"
DEFAULT_WINDOW_SIZE = 200
DEFAULT_MAX_BP_SPAN = 150
RNA_PLFOLD_PAIR_PROBABILITY_CUTOFF = 0.001


@dataclass(frozen=True)
class StructureProfile:
    unpaired_probs: np.ndarray
    pair_indices: np.ndarray
    pair_probabilities: np.ndarray
    sequence_length: int
    sequence_sha256: str


def coalesce_sequence(value: Any | None) -> str:
    return value if isinstance(value, str) else ""


def assemble_full_sequence(
    tx_sequence: str,
    utr5: str,
    cds: str,
    utr3: str,
) -> tuple[str, str]:
    if tx_sequence:
        return tx_sequence, "tx_column"
    return f"{utr5}{cds}{utr3}", "region_concat"


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def resolve_structure_cache_dir(config: dict[str, Any]) -> Path | None:
    dataset_config = require_mapping(config, "dataset")
    return resolve_repo_path(dataset_config.get("structure_cache_dir"))


class GlobalFoldingEngine:
    _checked = False

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        max_bp_span: int = DEFAULT_MAX_BP_SPAN,
    ) -> None:
        self.window_size = int(window_size)
        self.max_bp_span = int(max_bp_span)
        if not GlobalFoldingEngine._checked:
            self._check_rnaplfold()
            GlobalFoldingEngine._checked = True

    def _check_rnaplfold(self) -> None:
        try:
            subprocess.run(["RNAplfold", "--help"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            raise RuntimeError("RNAplfold not found in PATH") from error

    def run(self, sequence: str) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
        normalized_sequence = sequence.upper().replace("T", "U")
        length = len(normalized_sequence)
        if length == 0:
            return np.array([], dtype=np.float32), []

        window_size = min(self.window_size, length)
        max_bp_span = min(self.max_bp_span, length)

        with tempfile.TemporaryDirectory() as temp_dir:
            fasta = f">seq\n{normalized_sequence}\n"
            subprocess.run(
                [
                    "RNAplfold",
                    "-W",
                    str(window_size),
                    "-L",
                    str(max_bp_span),
                    "-c",
                    str(RNA_PLFOLD_PAIR_PROBABILITY_CUTOFF),
                    "-u",
                    "1",
                ],
                input=fasta,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            unpaired = self._parse_lunp(Path(temp_dir) / "seq_lunp", length)
            pairs = self._parse_dp_ps(Path(temp_dir) / "seq_dp.ps")
            return unpaired, pairs

    def _parse_lunp(self, path: Path, expected_length: int) -> np.ndarray:
        if not path.exists():
            return np.full(expected_length, 0.5, dtype=np.float32)

        values = np.full(expected_length, 0.5, dtype=np.float32)
        with open(path) as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) < 2:
                    continue
                index = int(parts[0]) - 1
                if 0 <= index < expected_length:
                    values[index] = float(parts[1])
        return values

    def _parse_dp_ps(self, path: Path) -> list[tuple[int, int, float]]:
        if not path.exists():
            return []

        pairs: list[tuple[int, int, float]] = []
        pattern = re.compile(
            r"(\d+)\s+(\d+)\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+ubox"
        )
        with open(path) as handle:
            for line in handle:
                match = pattern.search(line)
                if match is None:
                    continue
                left = int(match.group(1)) - 1
                right = int(match.group(2)) - 1
                probability = float(match.group(3)) ** 2
                pairs.append((left, right, probability))
        return pairs


def _serialize_pairs(
    pairs: list[tuple[int, int, float]],
) -> tuple[np.ndarray, np.ndarray]:
    if not pairs:
        return (
            np.empty((0, 2), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )
    pair_indices = np.asarray(
        [(left, right) for left, right, _ in pairs], dtype=np.int32
    )
    pair_probabilities = np.asarray(
        [probability for _, _, probability in pairs],
        dtype=np.float32,
    )
    return pair_indices, pair_probabilities


def _fold_sequence_task(
    sample_id: str,
    full_sequence: str,
) -> tuple[str, dict[str, Any] | None, str | None]:
    try:
        engine = GlobalFoldingEngine()
        unpaired_probs, pairs = engine.run(full_sequence)
        pair_indices, pair_probabilities = _serialize_pairs(pairs)
        payload = {
            "unpaired_probs": np.asarray(unpaired_probs, dtype=np.float32),
            "pair_indices": pair_indices,
            "pair_probabilities": pair_probabilities,
            "sequence_length": int(len(full_sequence)),
            "sequence_sha256": sequence_sha256(full_sequence),
        }
        return sample_id, payload, None
    except Exception as error:
        return sample_id, None, str(error)


def _ids_sha256(ids: list[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in ids:
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _dataset_sequence_sha256(records: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["sample_id"].encode("utf-8"))
        digest.update(b"\t")
        digest.update(record["sequence_sha256"].encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _full_sequence_from_row(
    row: Mapping[str, Any],
    tx_column: str,
    utr5_column: str,
    cds_column: str,
    utr3_column: str,
) -> tuple[str, str]:
    tx_sequence = coalesce_sequence(row.get(tx_column))
    utr5 = coalesce_sequence(row.get(utr5_column))
    cds = coalesce_sequence(row.get(cds_column))
    utr3 = coalesce_sequence(row.get(utr3_column))
    return assemble_full_sequence(tx_sequence, utr5, cds, utr3)


def precompute_structure_cache(
    config: dict[str, Any],
    output_dir: Path | None = None,
    limit: int | None = None,
) -> Path:
    dataset_config = require_mapping(config, "dataset")
    graph_config = require_mapping(config, "graphs")
    if not bool(graph_config["use_structure"]):
        raise ValueError("graphs.use_structure must be true for structure precompute")

    table_path = resolve_repo_path(dataset_config.get("table_path"))
    cache_dir = output_dir or resolve_structure_cache_dir(config)
    if table_path is None or cache_dir is None:
        raise ValueError("Config is missing dataset table or structure cache path")

    dataframe = pd.read_csv(table_path)
    if limit is not None:
        dataframe = dataframe.head(limit).copy()

    id_column = str(dataset_config["id_column"])
    tx_column = str(graph_config["tx_column"])
    utr5_column = str(graph_config["utr5_column"])
    cds_column = str(graph_config["cds_column"])
    utr3_column = str(graph_config["utr3_column"])
    num_workers = int(graph_config["num_workers"])
    if num_workers <= 0:
        num_workers = os.cpu_count() or 1

    records: list[dict[str, str]] = []
    source_counts = {"tx_column": 0, "region_concat": 0}
    task_args: list[tuple[str, str]] = []
    for _, row in dataframe.iterrows():
        row_dict = {str(key): value for key, value in row.to_dict().items()}
        sample_id = str(row_dict[id_column])
        full_sequence, source = _full_sequence_from_row(
            row_dict,
            tx_column,
            utr5_column,
            cds_column,
            utr3_column,
        )
        records.append(
            {
                "sample_id": sample_id,
                "sequence_sha256": sequence_sha256(full_sequence),
            }
        )
        source_counts[source] += 1
        task_args.append((sample_id, full_sequence))

    profiles: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_fold_sequence_task, sample_id, full_sequence)
            for sample_id, full_sequence in task_args
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Precomputing structure",
        ):
            sample_id, payload, error = future.result()
            if error is not None or payload is None:
                failures[sample_id] = error or "unknown error"
                continue
            profiles[sample_id] = payload

    cache_dir.mkdir(parents=True, exist_ok=True)
    profiles_path = cache_dir / PROFILES_FILE_NAME
    torch.save(profiles, profiles_path)

    ordered_ids = [record["sample_id"] for record in records]
    manifest = {
        "format_version": CACHE_FORMAT_VERSION,
        "config_name": config["config_name"],
        "config_path": config["config_path"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "table_path": str(table_path),
        "cache_dir": str(cache_dir),
        "id_column": id_column,
        "sequence_columns": {
            "tx_column": tx_column,
            "utr5_column": utr5_column,
            "cds_column": cds_column,
            "utr3_column": utr3_column,
        },
        "rows_requested": int(len(records)),
        "rows_cached": int(len(profiles)),
        "ordered_ids_sha256": _ids_sha256(ordered_ids),
        "dataset_sequence_sha256": _dataset_sequence_sha256(records),
        "sequence_source_counts": source_counts,
        "rnaplfold": {
            "window_size": DEFAULT_WINDOW_SIZE,
            "max_bp_span": DEFAULT_MAX_BP_SPAN,
            "u_length": 1,
            "pair_probability_cutoff": RNA_PLFOLD_PAIR_PROBABILITY_CUTOFF,
        },
        "files": {"profiles": PROFILES_FILE_NAME},
        "failures": failures,
    }
    save_json(manifest, cache_dir / "manifest.json")
    if failures:
        raise RuntimeError(f"Structure precompute failed for {len(failures)} rows")
    return cache_dir


def load_structure_cache(
    cache_dir: str | Path,
) -> tuple[dict[str, StructureProfile], dict[str, Any]]:
    resolved_cache_dir = Path(cache_dir).expanduser().resolve()
    manifest_path = resolved_cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Structure cache manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    if int(manifest.get("format_version", -1)) != CACHE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported structure cache format: {manifest.get('format_version')}"
        )

    profiles_file = manifest.get("files", {}).get("profiles")
    if not isinstance(profiles_file, str) or not profiles_file:
        raise ValueError("Structure cache manifest is missing files.profiles")
    profiles_path = resolved_cache_dir / profiles_file
    if not profiles_path.exists():
        raise FileNotFoundError(f"Structure cache payload not found: {profiles_path}")

    raw_profiles = torch.load(profiles_path, map_location="cpu", weights_only=False)
    if not isinstance(raw_profiles, dict):
        raise ValueError("Structure cache payload must be a dict keyed by sample id")

    profiles: dict[str, StructureProfile] = {}
    for sample_id, payload in raw_profiles.items():
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid structure payload for sample {sample_id}")
        unpaired_probs = np.asarray(payload["unpaired_probs"], dtype=np.float32)
        pair_indices = np.asarray(payload["pair_indices"], dtype=np.int32)
        pair_probabilities = np.asarray(payload["pair_probabilities"], dtype=np.float32)
        sequence_length = int(payload["sequence_length"])
        sequence_hash = str(payload["sequence_sha256"])
        if pair_indices.shape != (len(pair_probabilities), 2):
            raise ValueError(f"Invalid pair array shape for sample {sample_id}")
        if len(unpaired_probs) != sequence_length:
            raise ValueError(f"Invalid unpaired length for sample {sample_id}")
        profiles[str(sample_id)] = StructureProfile(
            unpaired_probs=unpaired_probs,
            pair_indices=pair_indices,
            pair_probabilities=pair_probabilities,
            sequence_length=sequence_length,
            sequence_sha256=sequence_hash,
        )
    return profiles, manifest


def validate_structure_cache_manifest(
    manifest: dict[str, Any],
    table_path: Path,
    id_column: str,
    tx_column: str,
    utr5_column: str,
    cds_column: str,
    utr3_column: str,
    structure_probability_threshold: float,
) -> None:
    if manifest.get("table_path") != str(table_path):
        raise ValueError(
            "Structure cache table_path does not match the current dataset table"
        )
    if manifest.get("id_column") != id_column:
        raise ValueError("Structure cache id_column does not match the config")
    expected_columns = {
        "tx_column": tx_column,
        "utr5_column": utr5_column,
        "cds_column": cds_column,
        "utr3_column": utr3_column,
    }
    if manifest.get("sequence_columns") != expected_columns:
        raise ValueError("Structure cache sequence columns do not match the config")
    rnaplfold_settings = manifest.get("rnaplfold", {})
    pair_probability_cutoff = rnaplfold_settings.get("pair_probability_cutoff")
    if (
        pair_probability_cutoff is None
        and rnaplfold_settings.get("probability_filter") == "none"
    ):
        pair_probability_cutoff = RNA_PLFOLD_PAIR_PROBABILITY_CUTOFF
    if pair_probability_cutoff is None:
        raise ValueError(
            "Structure cache manifest is missing rnaplfold.pair_probability_cutoff"
        )
    cutoff = float(pair_probability_cutoff)
    if structure_probability_threshold < cutoff:
        raise ValueError(
            "graphs.structure_probability_threshold must be >= "
            f"structure cache cutoff {cutoff:g}"
        )
