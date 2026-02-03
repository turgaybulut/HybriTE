# HybriTE

HybriTE is a graph neural network for mRNA translation efficiency (TE) prediction. It models each transcript as a heterogeneous graph that combines sequence composition, RNA secondary structure from RNAplfold, and biochemical priors (RBP binding, modifications). This repository contains the code used in the HybriTE paper, including data preparation, graph construction, training, prediction, and explainability.

## Requirements

- Python 3.10
- ViennaRNA (RNAplfold)

Quick setup (conda):

```bash
git clone https://github.com/turgaybulut/HybriTE.git && cd HybriTE
conda env create -f environment.yaml
conda activate hybrite
pip install -e .
```

## Data download

Download the prepared CSV files from this [Google Drive folder](https://drive.google.com/drive/folders/1h4gT797xGT1nZgT0iuO-dTuFukhOvsNT?usp=sharing) and place them here:

- `data/te/human_te_data_with_biochemicals.csv`
- `data/te/mouse_te_data_with_biochemicals.csv`

These paths are used by the commands below.

## Workflow

Follow these steps in order. Paths below match the repository layout.

### 1) Prepare data arrays

Convert the CSV into NumPy arrays for targets and biochemical features.

```bash
python scripts/prepare_data.py \
  --input_csv data/te/human_te_data_with_biochemicals.csv \
  --out_dir data/te/human \
  --select_k 100
```

For mouse, reuse the human-selected biochemical feature set:

```bash
python scripts/prepare_data.py \
  --input_csv data/te/mouse_te_data_with_biochemicals.csv \
  --out_dir data/te/mouse \
  --feature_meta data/te/human/meta.json
```

Outputs per species:

- `data/te/<species>/target.npy`
- `data/te/<species>/feature.npy`
- `data/te/<species>/meta.json`

### 2) Build graphs

Requires RNAplfold in your PATH.

```bash
python scripts/generate_graphs.py \
  data/te/human_te_data_with_biochemicals.csv \
  data/te/human/human_graph.pt

python scripts/generate_graphs.py \
  data/te/mouse_te_data_with_biochemicals.csv \
  data/te/mouse/mouse_graph.pt
```

For the no-structure ablation:

```bash
python scripts/generate_graphs_no_structure.py \
  data/te/human_te_data_with_biochemicals.csv \
  data/te/human/human_graph_nostruct.pt
```

### 3) Train

Model settings are loaded from `config.yaml` and `plugins/hybrite/config.yaml`.
Update `plugins/hybrite/config.yaml` to switch between human and mouse:

- `data.root: data/te/human` or `data/te/mouse`
- `data.graphs_pt: human_graph.pt` or `mouse_graph.pt`

Single split:

```bash
python train.py --config config.yaml
```

K-fold cross-validation:

```bash
python train_fold.py --config config.yaml
```

Checkpoints and logs are written under `results/hybrite/`.

### 4) Predict

Run inference from a saved checkpoint.

```bash
python predict.py \
  --checkpoint results/hybrite/fold_00/checkpoints/<checkpoint>.ckpt \
  --config config.yaml \
  --output_dir results/hybrite/fold_00
```

Run inference for all folds:

```bash
./predict_all_folds.sh results/hybrite/
```

Outputs:

- `predictions.csv`
- `ground_truth.csv`
- `metrics.csv`
- `target_metrics.csv` (per-target)

### 5) Explainability

Generate SHAP and graph importance plots.

```bash
python scripts/explain_model.py \
  --checkpoint results/hybrite/fold_00/checkpoints/<checkpoint>.ckpt \
  --species human \
  --data_dir data/te/human \
  --output_dir figures/shap
```

### 6) Cross-species prediction

Use a model trained on one species to predict another.

```bash
python predict_cross_species.py \
  --checkpoint results/hybrite/fold_00/checkpoints/<checkpoint>.ckpt \
  --train_config config.yaml \
  --test_inputs data/te/mouse/mouse_graph.pt \
  --test_targets data/te/mouse/target.npy \
  --test_biochemical_features data/te/mouse/feature.npy \
  --suffix human_to_mouse
```
