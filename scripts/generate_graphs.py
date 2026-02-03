from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

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
    utr5_bins: int = 8
    cds_bins: int = 32
    utr3_bins: int = 16


class GlobalFoldingEngine:
    _checked = False

    def __init__(self, window_size: int = 200, max_bp_span: int = 150):
        self.window_size = window_size
        self.max_bp_span = max_bp_span
        if not GlobalFoldingEngine._checked:
            self._check_rnaplfold()
            GlobalFoldingEngine._checked = True

    def _check_rnaplfold(self) -> None:
        try:
            subprocess.run(["RNAplfold", "--help"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "RNAplfold not found. Install ViennaRNA package:\n"
                "conda install -c bioconda viennarna"
            )

    def run(self, sequence: str) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
        sequence = sequence.upper().replace("T", "U")
        length = len(sequence)

        if length == 0:
            return np.array([]), []

        w = min(self.window_size, length)
        L = min(self.max_bp_span, length)

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                fasta_content = f">seq\n{sequence}\n"
                cmd = ["RNAplfold", "-W", str(w), "-L", str(L), "-u", "1"]
                subprocess.run(
                    cmd,
                    input=fasta_content,
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    check=True,
                )

                unpaired_probs = self._parse_lunp(tmpdir, "seq_lunp", length)
                pairs = self._parse_dp_ps(tmpdir, "seq_dp.ps")

                return unpaired_probs, pairs

            except subprocess.CalledProcessError as e:
                print(f"Error running RNAplfold: {e}")
                return np.full(length, 0.5), []
            except Exception as e:
                print(f"Error processing folding results: {e}")
                return np.full(length, 0.5), []

    def _parse_lunp(
        self, tmpdir: str, filename: str, expected_length: int
    ) -> np.ndarray:
        file_path = Path(tmpdir) / filename
        if not file_path.exists():
            return np.full(expected_length, 0.5)

        probs = np.zeros(expected_length)
        try:
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        idx = int(parts[0]) - 1
                        val = float(parts[1])
                        if 0 <= idx < expected_length:
                            probs[idx] = val
        except Exception:
            return np.full(expected_length, 0.5)

        return probs

    def _parse_dp_ps(self, tmpdir: str, filename: str) -> list[tuple[int, int, float]]:
        file_path = Path(tmpdir) / filename
        pairs = []
        if not file_path.exists():
            return pairs

        pattern = re.compile(r"(\d+)\s+(\d+)\s+([\d\.]+)\s+ubox")

        try:
            with open(file_path) as f:
                for line in f:
                    if "ubox" in line:
                        match = pattern.search(line)
                        if match:
                            i = int(match.group(1)) - 1
                            j = int(match.group(2)) - 1
                            sqrt_p = float(match.group(3))
                            p = sqrt_p * sqrt_p
                            if p > 0.001:
                                pairs.append((i, j, p))
        except Exception:
            pass

        return pairs


def _coalesce_sequence(value: Any | None) -> str:
    if isinstance(value, str) and len(value) > 0:
        return value
    return ""


def _normalize_seq(seq: str) -> str:
    return seq.upper().replace("T", "U")


def _codon_chunks(cds_sequence: str) -> list[str]:
    trimmed = cds_sequence[: len(cds_sequence) - (len(cds_sequence) % 3)]
    return [
        trimmed[i : i + 3].upper()
        for i in range(0, len(trimmed), 3)
        if len(trimmed[i : i + 3]) == 3
    ]


def _count_motifs(sequence: str, patterns: list[str]) -> int:
    sequence = sequence.upper().replace("T", "U")
    count = 0
    for pattern in patterns:
        regex_pattern = (
            pattern.replace("W", "[AU]")
            .replace("H", "[ACU]")
            .replace("M", "[AC]")
            .replace("N", "[ACGU]")
        )
        count += len(re.findall(regex_pattern, sequence))
    return count


def _extract_codon_features(cds_sequence: str) -> dict[str, float]:
    codons = _codon_chunks(cds_sequence)
    valid_codons = [c for c in codons if all(n in "ATGC" for n in c)]

    if not valid_codons:
        return {
            "tAI_mean": 0.5,
            "csc_mean": 0.0,
            "gc3_percent": 0.5,
            "rare_codon_density": 0.0,
        }

    tai_values = [CODON_TAI_VALUES.get(c, 0.1) for c in valid_codons]
    csc_values = [CODON_STABILITY_COEFF.get(c, 0.0) for c in valid_codons]

    gc3_positions = [c[2] for c in valid_codons]
    gc3_count = sum(1 for base in gc3_positions if base in "GC")

    rare_codons = sum(1 for tai in tai_values if tai < 0.2)

    return {
        "tAI_mean": np.mean(tai_values),
        "csc_mean": np.mean(csc_values),
        "gc3_percent": gc3_count / len(gc3_positions) if gc3_positions else 0.5,
        "rare_codon_density": rare_codons / len(valid_codons),
    }


def _tai_ramp_ratio(cds_sequence: str, ramp_len: int = 40) -> float:
    codons = _codon_chunks(cds_sequence)
    if not codons:
        return 1.0
    ramp = codons[:ramp_len]
    rest = codons[ramp_len:]
    ramp_vals = [CODON_TAI_VALUES.get(c, 0.1) for c in ramp] or [0.1]
    rest_vals = [CODON_TAI_VALUES.get(c, 0.1) for c in rest] or ramp_vals
    ramp_mean = float(np.mean(ramp_vals))
    rest_mean = float(np.mean(rest_vals))
    if rest_mean == 0.0:
        return 1.0
    return ramp_mean / rest_mean


def _positive_charge_density(bin_sequence: str) -> float:
    codons = _codon_chunks(bin_sequence)
    if not codons:
        return 0.0
    basic = {"AAA", "AAG", "AGA", "AGG", "CGA", "CGC", "CGG", "CGT"}
    hits = sum(1 for c in codons if c in basic)
    return hits / len(codons)


def _count_g4(sequence: str) -> float:
    seq = _normalize_seq(sequence)
    pattern = re.compile(r"G{3,}\w{1,7}G{3,}\w{1,7}G{3,}\w{1,7}G{3,}")
    matches = len(pattern.findall(seq))
    if len(seq) == 0:
        return 0.0
    return matches / len(seq)


def _uaug_min_distance_norm(utr5_sequence: str) -> float:
    seq = utr5_sequence.upper()
    positions = [m.start() for m in re.finditer("ATG", seq)]
    if not positions:
        return 1.0
    dist = min(len(seq) - (p + 1) for p in positions)
    return dist / max(1, len(seq))


def _m6a_density(sequence: str) -> float:
    seq = _normalize_seq(sequence)
    pattern = re.compile(r"[AGU][AG]AC[ACU]")
    matches = len(pattern.findall(seq))
    if len(seq) == 0:
        return 0.0
    return matches / len(seq)


def _tail_au_content(sequence: str, window: int = 50) -> float:
    if len(sequence) == 0:
        return 0.5
    tail = sequence[-window:].upper()
    au = tail.count("A") + tail.count("U") + tail.count("T")
    return au / len(tail)


class EnhancedGraphBuilder:
    def __init__(
        self, bin_config: RegionBinConfig, folding_engine: GlobalFoldingEngine
    ):
        self.bin_config = bin_config
        self.folding_engine = folding_engine

    def _create_region_bins_indices(
        self, length: int, num_bins: int, offset: int
    ) -> list[tuple[int, int]]:
        if num_bins <= 0:
            return []

        if length == 0:
            return [(offset, offset) for _ in range(num_bins)]

        bin_size = max(1, length // num_bins)
        bins = []
        for i in range(num_bins):
            start = i * bin_size
            end = start + bin_size if i < num_bins - 1 else length
            bins.append((offset + start, offset + end))
        return bins

    def _extract_bin_features(
        self,
        bin_sequence: str,
        region_type: str,
        bin_index: int,
        total_bins: int,
        unpaired_probs: np.ndarray | None = None,
        context: dict[str, float] | None = None,
    ) -> list[float]:
        relative_position = bin_index / max(1, total_bins - 1)
        context = context or {}

        if not bin_sequence:
            base_features = [0.0, 0.0, 0.0, 0.0, 0.0, relative_position, 0.5]
            region_features = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            return base_features + region_features

        length = len(bin_sequence)
        a_prop = bin_sequence.count("A") / length
        u_prop = (bin_sequence.count("U") + bin_sequence.count("T")) / length
        g_prop = bin_sequence.count("G") / length
        c_prop = bin_sequence.count("C") / length
        mean_unpaired_prob = np.mean(unpaired_probs)

        base_features = [
            a_prop,
            u_prop,
            g_prop,
            c_prop,
            np.log1p(length),
            relative_position,
            mean_unpaired_prob,
        ]

        if region_type == "utr5":
            cpg_count = bin_sequence.count("CG")
            cpg_density = cpg_count / max(1, length - 1)
            uorf_count = _count_motifs(bin_sequence, UORF_START)
            kozak_count = _count_motifs(bin_sequence, KOZAK_PATTERNS)
            top_count = _count_motifs(bin_sequence, TOP_MOTIF)
            g4_density = _count_g4(bin_sequence)
            uaug_dist = context.get("uaug_dist", 1.0)
            features = [
                cpg_density,
                np.log1p(uorf_count),
                np.log1p(kozak_count),
                np.log1p(top_count),
                g4_density,
                uaug_dist,
            ]

        elif region_type == "cds":
            codon_features = _extract_codon_features(bin_sequence)
            ramp_ratio = context.get("ramp_ratio", 1.0)
            basic_density = _positive_charge_density(bin_sequence)
            features = [
                codon_features["tAI_mean"],
                codon_features["csc_mean"],
                codon_features["gc3_percent"],
                codon_features["rare_codon_density"],
                ramp_ratio,
                basic_density,
            ]

        elif region_type == "utr3":
            are_count = _count_motifs(bin_sequence, ARE_PATTERNS)
            rbp_destab_count = _count_motifs(bin_sequence, RBP_DESTABILIZING)
            mirna_count = _count_motifs(bin_sequence, MIRNA_SEEDS)
            polya_count = _count_motifs(bin_sequence, POLYA_SIGNALS)
            m6a_density = _m6a_density(bin_sequence)
            tail_au = _tail_au_content(bin_sequence)
            features = [
                np.log1p(are_count),
                np.log1p(rbp_destab_count),
                np.log1p(mirna_count),
                np.log1p(polya_count),
                m6a_density,
                tail_au,
            ]
        else:
            features = [0.0] * 6

        return base_features + features

    def _create_edges(
        self,
        num_utr5_bins: int,
        num_cds_bins: int,
        num_utr3_bins: int,
        bin_ranges: list[tuple[int, int]],
        pair_list: list[tuple[int, int, float]],
    ) -> tuple[list[tuple[int, int]], list[list[float]]]:
        edges = []
        edge_attrs = []

        total_bins = num_utr5_bins + num_cds_bins + num_utr3_bins
        if total_bins == 0:
            return [], []

        offset = 0
        for i in range(num_utr5_bins - 1):
            u, v = offset + i, offset + i + 1
            edges.extend([(u, v), (v, u)])
            edge_attrs.extend([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

        if num_utr5_bins > 0 and num_cds_bins > 0:
            u, v = num_utr5_bins - 1, num_utr5_bins
            edges.extend([(u, v), (v, u)])
            edge_attrs.extend([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

        offset = num_utr5_bins
        for i in range(num_cds_bins - 1):
            u, v = offset + i, offset + i + 1
            edges.extend([(u, v), (v, u)])
            edge_attrs.extend([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

        if num_cds_bins > 0 and num_utr3_bins > 0:
            u, v = num_utr5_bins + num_cds_bins - 1, num_utr5_bins + num_cds_bins
            edges.extend([(u, v), (v, u)])
            edge_attrs.extend([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

        offset = num_utr5_bins + num_cds_bins
        for i in range(num_utr3_bins - 1):
            u, v = offset + i, offset + i + 1
            edges.extend([(u, v), (v, u)])
            edge_attrs.extend([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

        structure_weights = {}
        max_len = bin_ranges[-1][1] if bin_ranges else 0
        base_to_bin = np.full(max_len, -1, dtype=int)
        for b_idx, (start, end) in enumerate(bin_ranges):
            base_to_bin[start:end] = b_idx

        for i, j, p in pair_list:
            if i >= max_len or j >= max_len:
                continue
            u, v = base_to_bin[i], base_to_bin[j]

            if u != -1 and v != -1 and u != v:
                if u > v:
                    u, v = v, u
                if (u, v) not in structure_weights:
                    structure_weights[(u, v)] = 0.0
                structure_weights[(u, v)] += p

        for (u, v), total_p in structure_weights.items():
            weight = total_p
            edges.extend([(u, v), (v, u)])
            edge_attrs.extend([[0.0, 1.0, 0.0, weight], [0.0, 1.0, 0.0, weight]])

        utr5_start = 0
        utr5_end = num_utr5_bins - 1
        cds_start = num_utr5_bins
        cds_end = num_utr5_bins + num_cds_bins - 1
        utr3_start = num_utr5_bins + num_cds_bins
        utr3_end = total_bins - 1

        if num_utr5_bins > 0 and num_utr3_bins > 0:
            edges.extend([(utr5_start, utr3_end), (utr3_end, utr5_start)])
            edge_attrs.extend([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

        if num_utr5_bins > 0 and num_cds_bins > 0:
            edges.extend([(utr5_start, cds_start), (cds_start, utr5_start)])
            edge_attrs.extend([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

        if num_cds_bins > 0 and num_utr3_bins > 0:
            edges.extend([(cds_end, utr3_end), (utr3_end, cds_end)])
            edge_attrs.extend([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

        if num_utr5_bins > 0 and num_utr3_bins > 0:
            edges.extend([(utr5_end, utr3_start), (utr3_start, utr5_end)])
            edge_attrs.extend([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

        return edges, edge_attrs

    def build_graph_from_sequences(
        self, utr5: str, cds: str, utr3: str, full_sequence: str
    ) -> Data:
        unpaired_probs, pair_list = self.folding_engine.run(full_sequence)

        utr5_len, cds_len, utr3_len = len(utr5), len(cds), len(utr3)
        utr5_ranges = self._create_region_bins_indices(
            utr5_len, self.bin_config.utr5_bins, 0
        )
        cds_ranges = self._create_region_bins_indices(
            cds_len, self.bin_config.cds_bins, utr5_len
        )
        utr3_ranges = self._create_region_bins_indices(
            utr3_len, self.bin_config.utr3_bins, utr5_len + cds_len
        )
        all_ranges = utr5_ranges + cds_ranges + utr3_ranges

        uaug_dist = _uaug_min_distance_norm(utr5)
        ramp_ratio = _tai_ramp_ratio(cds)

        node_features = []
        for ranges, region_type in [
            (utr5_ranges, "utr5"),
            (cds_ranges, "cds"),
            (utr3_ranges, "utr3"),
        ]:
            for i, (start, end) in enumerate(ranges):
                bin_seq = full_sequence[start:end]
                bin_unpaired_probs = (
                    unpaired_probs[start:end] if len(unpaired_probs) > start else None
                )
                feats = self._extract_bin_features(
                    bin_seq,
                    region_type,
                    i,
                    len(ranges),
                    bin_unpaired_probs,
                    {"uaug_dist": uaug_dist, "ramp_ratio": ramp_ratio},
                )
                node_features.append(feats)

        edges, edge_attrs = self._create_edges(
            len(utr5_ranges), len(cds_ranges), len(utr3_ranges), all_ranges, pair_list
        )

        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges
            else torch.empty((2, 0), dtype=torch.long)
        )
        edge_attr = (
            torch.tensor(edge_attrs, dtype=torch.float32)
            if edge_attrs
            else torch.empty((0, 4), dtype=torch.float32)
        )

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(node_features),
        )


def _assemble_full_sequence(tx: str, utr5: str, cds: str, utr3: str) -> str:
    if tx:
        return tx
    return f"{utr5}{cds}{utr3}" if utr5 or cds or utr3 else ""


def process_enhanced_row(
    args: tuple[pd.Series, str, str, str, str, str, RegionBinConfig],
) -> tuple[str, Data] | None:
    row, id_col, tx_col, utr5_col, cds_col, utr3_col, bin_config = args
    ensid = str(row[id_col])
    tx = _coalesce_sequence(row.get(tx_col))
    utr5 = _coalesce_sequence(row.get(utr5_col))
    cds = _coalesce_sequence(row.get(cds_col))
    utr3 = _coalesce_sequence(row.get(utr3_col))
    full_sequence = _assemble_full_sequence(tx, utr5, cds, utr3)

    try:
        folding_engine = GlobalFoldingEngine()
        builder = EnhancedGraphBuilder(bin_config, folding_engine)
        graph_data = builder.build_graph_from_sequences(utr5, cds, utr3, full_sequence)
        return ensid, graph_data
    except Exception as e:
        print(f"Warning: Failed to process {ensid}: {e}")
        return None


def create_and_save_enhanced_graphs(
    df: pd.DataFrame,
    output_path: Path,
    id_col: str,
    tx_col: str,
    utr5_col: str,
    cds_col: str,
    utr3_col: str,
    bin_config: RegionBinConfig,
    num_workers: int,
) -> None:
    id_order = df[id_col].tolist()
    print(f"Found {len(df)} sequences to process.")
    graphs: dict[str, Data] = {}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        args_list = [
            (row, id_col, tx_col, utr5_col, cds_col, utr3_col, bin_config)
            for _, row in df.iterrows()
        ]
        futures = {executor.submit(process_enhanced_row, args) for args in args_list}

        for future in tqdm(
            as_completed(futures),
            total=len(df),
            desc="Creating Enhanced Graphs",
            unit="sequence",
        ):
            try:
                result = future.result()
                if result:
                    graphs[result[0]] = result[1]
            except Exception as e:
                print(f"A task failed with an error: {e}")

    print(f"\nSuccessfully generated {len(graphs)} out of {len(df)} enhanced graphs.")
    graph_list = [graphs.get(str(ensid)) for ensid in id_order if str(ensid) in graphs]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph_list, output_path)
    print(f"Saved {len(graph_list)} enhanced graphs to {output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate enhanced graphs with global RNA folding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_file", type=Path, help="Input CSV file")
    parser.add_argument("output_file", type=Path, help="Output .pt file")
    parser.add_argument("--id-col", type=str, default="gene_id")
    parser.add_argument("--tx-col", type=str, default="tx_sequence")
    parser.add_argument("--utr5-col", type=str, default="utr5_sequence")
    parser.add_argument("--cds-col", type=str, default="cds_sequence")
    parser.add_argument("--utr3-col", type=str, default="utr3_sequence")
    parser.add_argument("--utr5-bins", type=int, default=8)
    parser.add_argument("--cds-bins", type=int, default=32)
    parser.add_argument("--utr3-bins", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count())
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    print(f"Starting Global Folding Enhanced Graph Generation for {args.input_file}...")
    df = pd.read_csv(args.input_file)
    bin_config = RegionBinConfig(
        utr5_bins=args.utr5_bins, cds_bins=args.cds_bins, utr3_bins=args.utr3_bins
    )
    create_and_save_enhanced_graphs(
        df,
        args.output_file,
        args.id_col,
        args.tx_col,
        args.utr5_col,
        args.cds_col,
        args.utr3_col,
        bin_config,
        args.num_workers if args.num_workers > 0 else os.cpu_count(),
    )


if __name__ == "__main__":
    main()
