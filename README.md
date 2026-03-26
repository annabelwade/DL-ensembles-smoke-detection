# Improving Satellite-based Wildfire Smoke Plume Detection with Deep Ensembles

This repository contains the code for the paper **"Improving Satellite-based Wildfire Smoke Plume Detection with Deep Ensemble"**, presented at the ICLR 2026 Machine Learning for Remote Sensing Workshop. 

This project trains and evaluates deep ensembles to enhance the accuracy of GOES satellite-based wildfire smoke detection.

## Paper Citation


## Repo Structure

The codebase supports model training, ensemble generation, and evaluation:
```
.
├── configs/                  # JSON experiment configs (one per experiment)
│   └── exp<N>.json
├── SmokeDataset.py           # PyTorch Dataset for training/validation
├── TestSmokeDataset.py       # PyTorch Dataset for testing (returns filename)
├── model.py                  # Training script
├── test_models.py            # Evaluation and ensembling script
├── testing_ckpt_utils.py     # Utilities: checkpoint loading, IoU, plotting, ensemble inference
├── metrics.py                # IoU computation functions
├── run_model.script          # SLURM job script for training
├── run_seeds.sh              # Shell script to launch training across multiple seeds
```
## Installation & Environment Setup

This codebase was designed to run on a SLURM-managed HPC cluster with CUDA-enabled GPUs. 

```bash
git clone [https://github.com/annabelwade/DL-ensembles-smoke-detection.git](https://github.com/annabelwade/DL-ensembles-smoke-detection.git)
cd DL-ensembles-smoke-detection
pip install torch torchvision segmentation-models-pytorch scikit-image matplotlib tqdm tabulate pyproj cartopy
```

## Configuration
 
Each experiment is defined by a JSON config file in `configs/`. The experiment number `N` corresponds to `configs/expN.json`.
 
**Example config** (`configs/exp1.json`):
 
```json
{
    "architecture": "DeepLabV3Plus",
    "encoder": "timm-efficientnet-b2",
    "batch_size": 16,
    "lr": 0.0001,
    "loss": "BCEWithLogitsLoss",
    "use_chkpt": "True",
    "dn_weights": [3, 2, 1]
}
```
 
| Field | Description |
|---|---|
| `architecture` | Model architecture (any supported by `segmentation_models_pytorch`) |
| `encoder` | Encoder backbone name |
| `batch_size` | Training batch size |
| `lr` | Learning rate for Adam optimizer |
| `loss` | Loss function: `BCEWithLogitsLoss`, `DiceLoss`, or `CombinedLoss` |
| `use_chkpt` | `"True"` to resume from the latest checkpoint in `./models/` |
| `dn_weights` | Per-density loss weights `[heavy, medium, light]`; defaults to `[3, 2, 1]` |
| `use_best_model` | (Optional) `"True"` to load the best-validation-IoU checkpoint instead of the latest |
 
---

### Model training Via SLURM
 
```bash
sbatch --export=EXP_NUM=1,SEED=1 \
       --output=logs/exp1_seed1.log \
       --job-name=exp1_seed1 \
       run_model.script
```
 
### Training across multiple seeds
 
Use `run_seeds.sh` to launch a training sweep across seeds (edit the script to set your experiment numbers and seed range).

### Checkpoints
 
Checkpoints are saved to `./models/` using the naming convention:
 
```
<architecture>_exp<N>_best_<timestamp>.pth      # best validation IoU
<architecture>_exp<N>_seed<S>_best_<timestamp>.pth  # with seed
<architecture>_exp<N>_<timestamp>.pth           # every 10 epochs
```
 
Each checkpoint contains `epoch`, `model_state_dict`, `optimizer_state_dict`, `val_iou`, and `history` (train/val losses and IoU over epochs). After training, a loss and IoU curve plot is saved to `./train_results/exp<N>_training_history.png`.


## Evaluating and Ensembling Models
 
Use `test_models.py` to evaluate one or more models and optionally ensemble them.
 
### Usage
 
```bash
python test_models.py <INPUT_STRING> [PARAM1 PARAM2 ...]
```
 
**`INPUT_STRING` format:** `<exp1>_<exp2>_..._<ensemble_flag>`
 
- Experiment numbers separated by `_`, with the last element being an ensemble flag
- Ensemble flags:
  - `F` — evaluate experiments individually, no ensemble
  - `T` — evaluate individually and ensemble together
  - `S` — ensemble across seeds (used with experiments 1 and 3, which have 12 seeds each)
  - `ST` — seed-based ensemble
 
**`PARAM` arguments:** optional hyperparameter names from the config to display in the results table (default: `architecture`)
 
### Examples
 
```bash
# Evaluate experiment 3 alone
python test_models.py 3_F
 
# Evaluate experiments 1 and 3 individually, then ensemble them
python test_models.py 1_3_T architecture loss
 
# Ensemble experiments 1 and 3 across their 12 seeds
python test_models.py 1_3_S
 
# Evaluate the base model (exp 0) alongside exp 1.0.2, then ensemble
python test_models.py 0_1.0.2_T
```
 
### Output
 
Results are printed as a formatted table (and a LaTeX version) showing per-density IoU (heavy, medium, light) and overall IoU for each model and the ensemble. When incremental ensemble sizes are evaluated, a plot of overall IoU vs. ensemble size is saved to:
 
```
results_tables/<input_string>_ensemble_plot.png
```
 
The full results table is also saved as a pickle file:
 
```
results_tables/<input_string>.pkl
```
 
---
 
## Output Files
 
| Path | Description |
|---|---|
| `models/*.pth` | Model checkpoints |
| `train_results/exp<N>_training_history.png` | Training/validation loss and IoU curves |
| `results_tables/<input>.pkl` | Pickled results table from test run |
| `results_tables/<input>_ensemble_plot.png` | IoU vs. ensemble size plot |
| `test_results/` | Per-sample prediction TIFFs and metadata JSON (saved by `save_test_results`) |
 
---
