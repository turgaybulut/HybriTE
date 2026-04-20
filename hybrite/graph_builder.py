from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from .config import require_mapping, resolve_repo_path
from .io import save_json
from .structure_cache import (
    StructureProfile,
    assemble_full_sequence,
    coalesce_sequence,
    load_structure_cache,
    resolve_structure_cache_dir,
    sequence_sha256,
    validate_structure_cache_manifest,
)

CODON_TAI_VALUES: dict[str, float] = {
    "TTT": 1.0,
    "TTC": 0.58,
    "TTA": 0.11,
    "TTG": 0.29,
    "TCT": 0.15,
    "TCC": 0.28,
    "TCA": 0.12,
    "TCG": 0.05,
    "TAT": 1.0,
    "TAC": 0.43,
    "TAA": 1.0,
    "TAG": 0.2,
    "TGT": 1.0,
    "TGC": 0.54,
    "TGA": 0.31,
    "TGG": 1.0,
    "CTT": 0.11,
    "CTC": 0.18,
    "CTA": 0.07,
    "CTG": 0.41,
    "CCT": 0.28,
    "CCC": 0.33,
    "CCA": 0.27,
    "CCG": 0.11,
    "CAT": 1.0,
    "CAC": 0.42,
    "CAA": 1.0,
    "CAG": 0.34,
    "CGT": 0.08,
    "CGC": 0.19,
    "CGA": 0.11,
    "CGG": 0.21,
    "ATT": 1.0,
    "ATC": 0.46,
    "ATA": 0.17,
    "ATG": 1.0,
    "ACT": 0.24,
    "ACC": 0.36,
    "ACA": 0.28,
    "ACG": 0.11,
    "AAT": 1.0,
    "AAC": 0.45,
    "AAA": 1.0,
    "AAG": 0.42,
    "AGT": 0.12,
    "AGC": 0.27,
    "AGA": 0.2,
    "AGG": 0.2,
    "GTT": 0.11,
    "GTC": 0.24,
    "GTA": 0.07,
    "GTG": 1.0,
    "GCT": 0.27,
    "GCC": 0.4,
    "GCA": 0.23,
    "GCG": 0.11,
    "GAT": 1.0,
    "GAC": 0.38,
    "GAA": 1.0,
    "GAG": 0.42,
    "GGT": 0.16,
    "GGC": 0.34,
    "GGA": 0.25,
    "GGG": 0.25,
}

CODON_STABILITY_COEFF: dict[str, float] = {
    "TTT": -0.4,
    "TTC": 0.3,
    "TTA": -1.2,
    "TTG": -0.1,
    "TCT": -0.2,
    "TCC": 0.4,
    "TCA": -0.3,
    "TCG": -0.6,
    "TAT": -0.3,
    "TAC": 0.5,
    "TAA": 0.0,
    "TAG": 0.0,
    "TGT": 0.1,
    "TGC": 0.6,
    "TGA": 0.0,
    "TGG": 0.8,
    "CTT": -0.6,
    "CTC": 0.2,
    "CTA": -0.9,
    "CTG": 0.7,
    "CCT": 0.1,
    "CCC": 0.5,
    "CCA": -0.1,
    "CCG": -0.2,
    "CAT": -0.2,
    "CAC": 0.4,
    "CAA": -0.5,
    "CAG": 0.6,
    "CGT": -0.8,
    "CGC": 0.3,
    "CGA": -0.9,
    "CGG": 0.1,
    "ATT": -0.1,
    "ATC": 0.4,
    "ATA": -0.7,
    "ATG": 0.5,
    "ACT": -0.1,
    "ACC": 0.5,
    "ACA": -0.2,
    "ACG": -0.3,
    "AAT": -0.4,
    "AAC": 0.3,
    "AAA": -0.6,
    "AAG": 0.2,
    "AGT": -0.4,
    "AGC": 0.2,
    "AGA": -1.0,
    "AGG": -0.8,
    "GTT": -0.3,
    "GTC": 0.3,
    "GTA": -0.5,
    "GTG": 0.8,
    "GCT": 0.0,
    "GCC": 0.6,
    "GCA": -0.1,
    "GCG": -0.2,
    "GAT": -0.2,
    "GAC": 0.4,
    "GAA": -0.4,
    "GAG": 0.3,
    "GGT": -0.1,
    "GGC": 0.4,
    "GGA": -0.3,
    "GGG": 0.1,
}

ARE_PATTERNS = ["AUUUA", "UAUUUAU", "AUUUAUUUA", "WWAUUUAWW", "UUAUUUAUU"]
RBP_DESTABILIZING = ["UGUANAUA", "UGUAHAUA", "UUUUUUU", "WUUUUUW", "UUAUUUA"]
MIRNA_SEEDS = [
    "UGUGCUU",
    "GAGGUAG",
    "AAAGUGC",
    "AGCACUU",
    "ACAUUCA",
    "UAAAGCU",
    "ACAGUAC",
    "GCCUACU",
    "CAGUGCA",
    "ACCCUGU",
    "GGCAGUG",
    "UACCUCA",
    "GCAAAAG",
    "AACUGCC",
    "UGCACUU",
]
POLYA_SIGNALS = [
    "AAUAAA",
    "AUUAAA",
    "UAUAAA",
    "AGUAAA",
    "AAGAAA",
    "AAUAUA",
    "AAUACA",
    "CAUAAA",
    "GAUAAA",
    "AAUAAG",
]
KOZAK_PATTERNS = ["GCCGCCAUG", "GCCACCAUG", "ACCAUGG", "GNNAUGG"]
UORF_START = ["AUG"]
TOP_MOTIF = ["CUUUCC", "CCCUUC", "CUCCCU"]


@dataclass(frozen=True)
class RegionBinConfig:
    utr5_bins: int
    cds_bins: int
    utr3_bins: int


def _normalize_seq(sequence: str) -> str:
    return sequence.upper().replace("T", "U")


def _codon_chunks(sequence: str) -> list[str]:
    trimmed = sequence[: len(sequence) - (len(sequence) % 3)]
    return [trimmed[index : index + 3].upper() for index in range(0, len(trimmed), 3)]


def _count_motifs(sequence: str, patterns: list[str]) -> int:
    normalized = _normalize_seq(sequence)
    total = 0
    for pattern in patterns:
        regex_pattern = (
            pattern.replace("W", "[AU]")
            .replace("H", "[ACU]")
            .replace("M", "[AC]")
            .replace("N", "[ACGU]")
        )
        total += len(re.findall(regex_pattern, normalized))
    return total


def _extract_codon_features(sequence: str) -> dict[str, float]:
    codons = [
        codon
        for codon in _codon_chunks(sequence)
        if all(base in "ATGC" for base in codon)
    ]
    if not codons:
        return {
            "tAI_mean": 0.5,
            "csc_mean": 0.0,
            "gc3_percent": 0.5,
            "rare_codon_density": 0.0,
        }

    tai_values = [CODON_TAI_VALUES.get(codon, 0.1) for codon in codons]
    csc_values = [CODON_STABILITY_COEFF.get(codon, 0.0) for codon in codons]
    gc3_percent = sum(1 for codon in codons if codon[2] in "GC") / len(codons)
    rare_codon_density = sum(1 for value in tai_values if value < 0.2) / len(codons)
    return {
        "tAI_mean": float(np.mean(tai_values)),
        "csc_mean": float(np.mean(csc_values)),
        "gc3_percent": float(gc3_percent),
        "rare_codon_density": float(rare_codon_density),
    }


def _tai_ramp_ratio(sequence: str, ramp_length: int = 40) -> float:
    codons = _codon_chunks(sequence)
    if not codons:
        return 1.0
    ramp = codons[:ramp_length]
    rest = codons[ramp_length:]
    ramp_values = [CODON_TAI_VALUES.get(codon, 0.1) for codon in ramp] or [0.1]
    rest_values = [CODON_TAI_VALUES.get(codon, 0.1) for codon in rest] or ramp_values
    rest_mean = float(np.mean(rest_values))
    if rest_mean == 0.0:
        return 1.0
    return float(np.mean(ramp_values) / rest_mean)


def _positive_charge_density(sequence: str) -> float:
    codons = _codon_chunks(sequence)
    if not codons:
        return 0.0
    basic_codons = {"AAA", "AAG", "AGA", "AGG", "CGA", "CGC", "CGG", "CGT"}
    return sum(1 for codon in codons if codon in basic_codons) / len(codons)


def _count_g4(sequence: str) -> float:
    normalized = _normalize_seq(sequence)
    if len(normalized) == 0:
        return 0.0
    pattern = re.compile(r"G{3,}\w{1,7}G{3,}\w{1,7}G{3,}\w{1,7}G{3,}")
    return len(pattern.findall(normalized)) / len(normalized)


def _uaug_min_distance_norm(sequence: str) -> float:
    starts = [match.start() for match in re.finditer("ATG", sequence.upper())]
    if not starts:
        return 1.0
    distance = min(len(sequence) - (start + 1) for start in starts)
    return distance / max(1, len(sequence))


def _m6a_density(sequence: str) -> float:
    normalized = _normalize_seq(sequence)
    if len(normalized) == 0:
        return 0.0
    return len(re.findall(r"[AGU][AG]AC[ACU]", normalized)) / len(normalized)


def _tail_au_content(sequence: str, window: int = 50) -> float:
    if len(sequence) == 0:
        return 0.5
    tail = sequence[-window:].upper()
    au_count = tail.count("A") + tail.count("U") + tail.count("T")
    return au_count / len(tail)


class GraphBuilder:
    def __init__(
        self,
        bin_config: RegionBinConfig,
        use_structure: bool,
        structure_probability_threshold: float = 0.001,
    ) -> None:
        if min(bin_config.utr5_bins, bin_config.cds_bins, bin_config.utr3_bins) < 0:
            raise ValueError("Region bin counts must be non-negative")
        if structure_probability_threshold < 0:
            raise ValueError("structure_probability_threshold must be non-negative")
        self.bin_config = bin_config
        self.use_structure = use_structure
        self.structure_probability_threshold = float(structure_probability_threshold)

    def _create_region_bins_indices(
        self,
        length: int,
        num_bins: int,
        offset: int,
    ) -> list[tuple[int, int]]:
        if num_bins <= 0:
            return []
        if length == 0:
            return [(offset, offset) for _ in range(num_bins)]

        bin_size = max(1, length // num_bins)
        ranges: list[tuple[int, int]] = []
        for bin_index in range(num_bins):
            start = bin_index * bin_size
            end = start + bin_size if bin_index < num_bins - 1 else length
            ranges.append((offset + start, offset + end))
        return ranges

    def _extract_bin_features(
        self,
        bin_sequence: str,
        region_type: str,
        bin_index: int,
        total_bins: int,
        unpaired_probs: np.ndarray | None,
        context: dict[str, float],
    ) -> list[float]:
        relative_position = bin_index / max(1, total_bins - 1)
        base_empty = [0.0, 0.0, 0.0, 0.0, 0.0, relative_position]
        if self.use_structure:
            base_empty.append(0.5)

        if not bin_sequence:
            return base_empty + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        length = len(bin_sequence)
        base_features = [
            bin_sequence.count("A") / length,
            (bin_sequence.count("U") + bin_sequence.count("T")) / length,
            bin_sequence.count("G") / length,
            bin_sequence.count("C") / length,
            float(np.log1p(length)),
            relative_position,
        ]
        if self.use_structure:
            mean_unpaired = (
                float(np.mean(unpaired_probs)) if unpaired_probs is not None else 0.5
            )
            base_features.append(mean_unpaired)

        if region_type == "utr5":
            region_features = [
                bin_sequence.count("CG") / max(1, length - 1),
                float(np.log1p(_count_motifs(bin_sequence, UORF_START))),
                float(np.log1p(_count_motifs(bin_sequence, KOZAK_PATTERNS))),
                float(np.log1p(_count_motifs(bin_sequence, TOP_MOTIF))),
                _count_g4(bin_sequence),
                context.get("uaug_dist", 1.0),
            ]
        elif region_type == "cds":
            codon_features = _extract_codon_features(bin_sequence)
            region_features = [
                codon_features["tAI_mean"],
                codon_features["csc_mean"],
                codon_features["gc3_percent"],
                codon_features["rare_codon_density"],
                context.get("ramp_ratio", 1.0),
                _positive_charge_density(bin_sequence),
            ]
        else:
            region_features = [
                float(np.log1p(_count_motifs(bin_sequence, ARE_PATTERNS))),
                float(np.log1p(_count_motifs(bin_sequence, RBP_DESTABILIZING))),
                float(np.log1p(_count_motifs(bin_sequence, MIRNA_SEEDS))),
                float(np.log1p(_count_motifs(bin_sequence, POLYA_SIGNALS))),
                _m6a_density(bin_sequence),
                _tail_au_content(bin_sequence),
            ]
        return base_features + region_features

    def _create_edges(
        self,
        num_utr5_bins: int,
        num_cds_bins: int,
        num_utr3_bins: int,
        bin_ranges: list[tuple[int, int]],
        pair_indices: np.ndarray | None,
        pair_probabilities: np.ndarray | None,
    ) -> tuple[list[tuple[int, int]], list[list[float]]]:
        edges: list[tuple[int, int]] = []
        edge_attrs: list[list[float]] = []
        edge_dim = 4 if self.use_structure else 3

        def add_edge(
            left: int, right: int, edge_type: str, weight: float = 0.0
        ) -> None:
            if edge_type == "sequential":
                template = [1.0, 0.0, 0.0]
            elif edge_type == "structural":
                template = [0.0, 1.0, 0.0]
            else:
                template = [0.0, 0.0, 1.0]
            attr = template + [weight] if edge_dim == 4 else template
            edges.extend([(left, right), (right, left)])
            edge_attrs.extend([attr, attr])

        total_bins = num_utr5_bins + num_cds_bins + num_utr3_bins
        if total_bins == 0:
            return edges, edge_attrs

        for index in range(num_utr5_bins - 1):
            add_edge(index, index + 1, "sequential")
        if num_utr5_bins > 0 and num_cds_bins > 0:
            add_edge(num_utr5_bins - 1, num_utr5_bins, "sequential")
        for index in range(num_cds_bins - 1):
            offset = num_utr5_bins
            add_edge(offset + index, offset + index + 1, "sequential")
        if num_cds_bins > 0 and num_utr3_bins > 0:
            add_edge(
                num_utr5_bins + num_cds_bins - 1,
                num_utr5_bins + num_cds_bins,
                "sequential",
            )
        for index in range(num_utr3_bins - 1):
            offset = num_utr5_bins + num_cds_bins
            add_edge(offset + index, offset + index + 1, "sequential")

        if (
            self.use_structure
            and bin_ranges
            and pair_indices is not None
            and pair_probabilities is not None
        ):
            structure_weights: dict[tuple[int, int], float] = {}
            max_length = bin_ranges[-1][1]
            base_to_bin = np.full(max_length, -1, dtype=np.int64)
            for bin_index, (start, end) in enumerate(bin_ranges):
                base_to_bin[start:end] = bin_index
            for pair_index, probability in zip(
                pair_indices, pair_probabilities, strict=False
            ):
                if probability < self.structure_probability_threshold:
                    continue
                left = int(pair_index[0])
                right = int(pair_index[1])
                if left >= max_length or right >= max_length:
                    continue
                source = int(base_to_bin[left])
                target = int(base_to_bin[right])
                if source == -1 or target == -1 or source == target:
                    continue
                if source > target:
                    source, target = target, source
                structure_weights[(source, target)] = (
                    structure_weights.get((source, target), 0.0) + probability
                )
            for (left, right), weight in structure_weights.items():
                add_edge(left, right, "structural", weight)

        utr5_start = 0
        utr5_end = num_utr5_bins - 1
        cds_start = num_utr5_bins
        cds_end = num_utr5_bins + num_cds_bins - 1
        utr3_start = num_utr5_bins + num_cds_bins
        utr3_end = total_bins - 1

        if num_utr5_bins > 0 and num_utr3_bins > 0:
            add_edge(utr5_start, utr3_end, "long_range")
            add_edge(utr5_end, utr3_start, "long_range")
        if num_utr5_bins > 0 and num_cds_bins > 0:
            add_edge(utr5_start, cds_start, "long_range")
        if num_cds_bins > 0 and num_utr3_bins > 0:
            add_edge(cds_end, utr3_end, "long_range")

        return edges, edge_attrs

    def build_graph_from_sequences(
        self,
        utr5: str,
        cds: str,
        utr3: str,
        full_sequence: str,
        structure_profile: StructureProfile | None = None,
    ) -> Data:
        unpaired_probs = None
        pair_indices = None
        pair_probabilities = None
        if self.use_structure:
            if structure_profile is None:
                raise ValueError(
                    "Structure profile is required when graphs.use_structure is true"
                )
            if structure_profile.sequence_length != len(full_sequence):
                raise ValueError("Structure profile length does not match sequence")
            if structure_profile.sequence_sha256 != sequence_sha256(full_sequence):
                raise ValueError("Structure profile sequence hash does not match")
            unpaired_probs = structure_profile.unpaired_probs
            pair_indices = structure_profile.pair_indices
            pair_probabilities = structure_profile.pair_probabilities

        utr5_ranges = self._create_region_bins_indices(
            len(utr5), self.bin_config.utr5_bins, 0
        )
        cds_ranges = self._create_region_bins_indices(
            len(cds), self.bin_config.cds_bins, len(utr5)
        )
        utr3_ranges = self._create_region_bins_indices(
            len(utr3),
            self.bin_config.utr3_bins,
            len(utr5) + len(cds),
        )
        all_ranges = utr5_ranges + cds_ranges + utr3_ranges
        context = {
            "uaug_dist": _uaug_min_distance_norm(utr5),
            "ramp_ratio": _tai_ramp_ratio(cds),
        }

        node_features: list[list[float]] = []
        for ranges, region_type in (
            (utr5_ranges, "utr5"),
            (cds_ranges, "cds"),
            (utr3_ranges, "utr3"),
        ):
            for bin_index, (start, end) in enumerate(ranges):
                region_unpaired = None
                if unpaired_probs is not None and len(unpaired_probs) > start:
                    region_unpaired = unpaired_probs[start:end]
                node_features.append(
                    self._extract_bin_features(
                        full_sequence[start:end],
                        region_type,
                        bin_index,
                        len(ranges),
                        region_unpaired,
                        context,
                    )
                )

        edges, edge_attrs = self._create_edges(
            len(utr5_ranges),
            len(cds_ranges),
            len(utr3_ranges),
            all_ranges,
            pair_indices,
            pair_probabilities,
        )

        edge_dim = 4 if self.use_structure else 3
        return Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=(
                torch.tensor(edges, dtype=torch.long).t().contiguous()
                if edges
                else torch.empty((2, 0), dtype=torch.long)
            ),
            edge_attr=(
                torch.tensor(edge_attrs, dtype=torch.float32)
                if edge_attrs
                else torch.empty((0, edge_dim), dtype=torch.float32)
            ),
            num_nodes=len(node_features),
        )


def _process_row(
    args: tuple[
        dict[str, Any],
        str,
        str,
        str,
        str,
        str,
        RegionBinConfig,
        bool,
        float,
        StructureProfile | None,
    ],
) -> tuple[str, Data | None, str | None]:
    (
        row,
        id_column,
        tx_column,
        utr5_column,
        cds_column,
        utr3_column,
        bin_config,
        use_structure,
        structure_probability_threshold,
        structure_profile,
    ) = args
    sample_id = str(row[id_column])
    try:
        builder = GraphBuilder(
            bin_config,
            use_structure,
            structure_probability_threshold=structure_probability_threshold,
        )
        tx_sequence = coalesce_sequence(row.get(tx_column))
        utr5 = coalesce_sequence(row.get(utr5_column))
        cds = coalesce_sequence(row.get(cds_column))
        utr3 = coalesce_sequence(row.get(utr3_column))
        full_sequence, _ = assemble_full_sequence(tx_sequence, utr5, cds, utr3)
        graph = builder.build_graph_from_sequences(
            utr5,
            cds,
            utr3,
            full_sequence,
            structure_profile=structure_profile,
        )
        return sample_id, graph, None
    except Exception as error:
        return sample_id, None, str(error)


def build_graphs(
    config: dict[str, Any], output_path: Path | None = None, limit: int | None = None
) -> Path:
    dataset_config = require_mapping(config, "dataset")
    graph_config = require_mapping(config, "graphs")
    table_path = resolve_repo_path(dataset_config["table_path"])
    default_output = resolve_repo_path(dataset_config["graph_path"])
    if table_path is None or default_output is None:
        raise ValueError("Config is missing dataset table or graph path")

    final_output = output_path or default_output
    dataframe = pd.read_csv(table_path)
    if limit is not None:
        dataframe = dataframe.head(limit).copy()

    bin_config = RegionBinConfig(
        utr5_bins=int(graph_config["utr5_bins"]),
        cds_bins=int(graph_config["cds_bins"]),
        utr3_bins=int(graph_config["utr3_bins"]),
    )
    id_column = str(dataset_config["id_column"])
    tx_column = str(graph_config["tx_column"])
    utr5_column = str(graph_config["utr5_column"])
    cds_column = str(graph_config["cds_column"])
    utr3_column = str(graph_config["utr3_column"])
    use_structure = bool(graph_config["use_structure"])
    num_workers = int(graph_config["num_workers"])
    structure_probability_threshold = float(
        graph_config.get("structure_probability_threshold", 0.001)
    )
    if num_workers <= 0:
        num_workers = os.cpu_count() or 1

    graphs: dict[str, Data] = {}
    failures: dict[str, str] = {}
    structure_cache_dir: Path | None = None
    structure_cache_manifest: dict[str, Any] | None = None
    structure_profiles: dict[str, StructureProfile] | None = None
    if use_structure:
        structure_cache_dir = resolve_structure_cache_dir(config)
        if structure_cache_dir is None:
            raise ValueError(
                "dataset.structure_cache_dir is required when "
                "graphs.use_structure is true"
            )
        structure_profiles, structure_cache_manifest = load_structure_cache(
            structure_cache_dir
        )
        validate_structure_cache_manifest(
            structure_cache_manifest,
            table_path,
            id_column,
            tx_column,
            utr5_column,
            cds_column,
            utr3_column,
            structure_probability_threshold,
        )

    task_args = []
    for _, row in dataframe.iterrows():
        row_dict = {str(key): value for key, value in row.to_dict().items()}
        structure_profile = None
        if use_structure:
            sample_id = str(row_dict[id_column])
            tx_sequence = coalesce_sequence(row_dict.get(tx_column))
            utr5 = coalesce_sequence(row_dict.get(utr5_column))
            cds = coalesce_sequence(row_dict.get(cds_column))
            utr3 = coalesce_sequence(row_dict.get(utr3_column))
            full_sequence, _ = assemble_full_sequence(tx_sequence, utr5, cds, utr3)
            structure_profile = (
                structure_profiles.get(sample_id)
                if structure_profiles is not None
                else None
            )
            if structure_profile is None:
                failures[sample_id] = "Missing structure profile in cache"
                continue
            if structure_profile.sequence_length != len(full_sequence):
                failures[sample_id] = "Cached structure length does not match sequence"
                continue
            if structure_profile.sequence_sha256 != sequence_sha256(full_sequence):
                failures[sample_id] = "Cached structure sequence hash does not match"
                continue
        task_args.append(
            (
                row_dict,
                id_column,
                tx_column,
                utr5_column,
                cds_column,
                utr3_column,
                bin_config,
                use_structure,
                structure_probability_threshold,
                structure_profile,
            )
        )

    if task_args:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_process_row, task_arg) for task_arg in task_args
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Building graphs"
            ):
                sample_id, graph, error = future.result()
                if error is not None or graph is None:
                    failures[sample_id] = error or "unknown error"
                    continue
                graphs[sample_id] = graph

    ordered_ids = dataframe[id_column].astype(str).tolist()
    graph_list = [graphs[sample_id] for sample_id in ordered_ids if sample_id in graphs]
    final_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph_list, final_output)
    save_json(
        {
            "config_name": config["config_name"],
            "input_table": str(table_path),
            "output_graphs": str(final_output),
            "rows_requested": int(len(dataframe)),
            "rows_saved": int(len(graph_list)),
            "use_structure": use_structure,
            "structure_cache_dir": (
                str(structure_cache_dir) if structure_cache_dir is not None else None
            ),
            "structure_cache_manifest": (
                str(structure_cache_dir / "manifest.json")
                if structure_cache_dir is not None
                else None
            ),
            "structure_cache_rows": (
                int(structure_cache_manifest.get("rows_cached", 0))
                if structure_cache_manifest is not None
                else None
            ),
            "structure_probability_threshold": structure_probability_threshold,
            "bins": {
                "utr5": int(bin_config.utr5_bins),
                "cds": int(bin_config.cds_bins),
                "utr3": int(bin_config.utr3_bins),
            },
            "failures": failures,
        },
        final_output.with_suffix(".json"),
    )
    if failures:
        raise RuntimeError(f"Graph generation failed for {len(failures)} rows")
    return final_output
