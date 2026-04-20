# HybriTE

HybriTE is a graph neural network for mRNA translation efficiency (TE) prediction. It represents each transcript with region-aware graph nodes that combine sequence-derived features, RNA secondary-structure priors from RNAplfold, and biochemical annotations. This repository contains the HybriTE code used for data preparation, graph construction, HybriTE training, LightGBM baseline training, cross-species evaluation, interpretability, and figure generation.

## Requirements

- Python 3.10
- ViennaRNA (`RNAplfold` on `PATH`)

## Setup

```bash
conda env create -f environment.yaml
conda activate hybrite
pip install -e .
```

## Dataset

Download the prepared dataset files from this Google Drive folder:

<https://drive.google.com/drive/folders/1h4gT797xGT1nZgT0iuO-dTuFukhOvsNT?usp=sharing>

Place the downloaded files at the paths expected by the configs:

- `data/raw/human/translation_efficiency_with_biochemistry.csv`
- `data/raw/mouse/translation_efficiency_raw.csv`
- `data/derived/mouse/translation_efficiency_with_biochemistry.csv` (if using the transferred mouse table directly)

If you want to rebuild the mouse transferred table yourself, also provide:

- `data/raw/orthology/mart_export.txt`

## Main folders

- `hybrite/` — core model and utilities
- `scripts/` — runnable paper scripts
- `configs/` — paper configs only

## Scripts

### Data and graph preparation

- `scripts/transfer_orthology.py` — transfer human biochemical features to mouse through one-to-one orthology
- `scripts/precompute_structure.py` — run RNAplfold and save structure caches
- `scripts/build_graphs.py` — build HybriTE transcript graphs from input tables
- `scripts/prepare.py` — create CV folds and fold-specific biochemical feature manifests

### Training

- `scripts/train.py` — train the HybriTE graph model
- `scripts/train_baseline.py` — train the LightGBM biochemical-only baseline

### Cross-species evaluation

- `scripts/cross_species.py` — evaluate a HybriTE checkpoint across species
- `scripts/cross_species_baseline.py` — evaluate a LightGBM baseline across species

### Analysis and figures

- `scripts/compare_runs.py` — compare two matched run directories fold by fold
- `scripts/generate_interpretability_artifacts.py` — generate human interpretability outputs
- `scripts/generate_all_figures.py` — generate all main paper figures

### Figure scripts

- `scripts/plots/plot_figure_performance_analysis.py` — main benchmark, ablation, and transfer figure
- `scripts/plots/plot_figure_interpretability_analysis.py` — interpretability figure
- `scripts/plots/plot_figure_per_target_performance.py` — per-source performance figure
- `scripts/plots/plot_figure_target_correlation.py` — prediction-similarity figure

## Main configs

- `configs/main/human.yaml` — full human HybriTE model
- `configs/main/human_nobio.yaml` — human `-Bio` ablation
- `configs/main/human_nostruct.yaml` — human `-Struct` ablation
- `configs/main/mouse.yaml` — full mouse HybriTE model
- `configs/main/mouse_nobio.yaml` — mouse `-Bio` ablation
- `configs/main/mouse_nostruct.yaml` — mouse `-Struct` ablation
- `configs/baselines/human_lightgbm.yaml` — human LightGBM baseline
- `configs/baselines/mouse_lightgbm.yaml` — mouse LightGBM baseline

## Sensitivity configs used in the paper

- `configs/controls/human_bins_coarse.yaml`
- `configs/controls/human_bins_fine.yaml`
- `configs/controls/human_threshold_1e_2.yaml`
- `configs/controls/human_threshold_1e_1.yaml`
- `configs/controls/human_hp_layers_2_hidden_64.yaml`
- `configs/controls/human_hp_layers_2_hidden_128.yaml`
- `configs/controls/human_hp_layers_2_hidden_256.yaml`
- `configs/controls/human_hp_layers_3_hidden_64.yaml`
- `configs/controls/human_hp_layers_3_hidden_128.yaml`
- `configs/controls/human_hp_layers_3_hidden_256.yaml`
