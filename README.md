# LSTM Split-Test Replication

Code-only package to rebuild LSTM case inputs and train/evaluate the gap-filling models from the Paper 1A experiments (C0–C3). Prebuilt inputs, outputs and other datasets for replication are available at https://doi.org/10.5281/zenodo.23717919

Trained weights and ready-made `Inputs/` / `Outputs/` are **not** included. Place the required datasets under `Datasets/`, rebuild case folders, then train.

## Code Attribution and Licensing

This repository implements a multi-site LSTM framework for reconstructing missing stream nitrate concentrations. The LSTM architecture and selected training components are adapted from **HydroDL** ([mhpi/hydroDL](https://github.com/mhpi/hydroDL)), developed by the Multi-scale Hydrology, Processes and Intelligence (MHPI) research group at The Pennsylvania State University.

The HydroDL-derived components (`Models/LSTM_SplitTest/model.py`, `dropout.py`, and `train_functions.py`) are subject to the original HydroDL Non-Commercial Software License Agreement. The original copyright notice and license terms are retained in `THIRD_PARTY_LICENSES/hydroDL-LICENSE.txt`.

The nitrate reconstruction experiments, donor-selection strategies, data-processing workflows, and evaluation procedures were developed for this research project.


## Layout

```
LSTM_Replication/
  LICENSE
  THIRD_PARTY_LICENSES/
    hydroDL-LICENSE.txt
  Datasets/                  # you provide parquets (see below)
  Models/LSTM_SplitTest/     # trainer + config template
  Scripts/                   # rebuild Inputs/
    Auxilliary_Scripts/experiment_case_sets.py
  requirements.txt
  README.md
```

Paths match the original project layout (`PAPER_ROOT` = this folder).

## Requirements

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

GPU training needs a CUDA build of PyTorch matching your driver.

## Datasets to provide

Put these under `Datasets/` (same filenames):

| File | Used by |
|------|---------|
| `forcing_117_V1.parquet` | C1 splits / signature donor pool |
| `forcing_SS_52_V1.parquet` | Script 12 → C2/C3 sensor scenarios |
| `attribute_117_V1.parquet` | LSTM configs + signature features |
| `Discharge_0060/nitrate_site_nldi_candidates.csv` | Regional (REG) donor topology |
| Raw nitrate parquet (one of the paths below) | Signature / donor features |

Optional but produced by script 12 (or provide if you skip 12):

- `forcing_BASE_50_V1.parquet`
- `forcing_REDUCED_50_V1.parquet`
- `forcing_ALL_50_V1.parquet`

Raw nitrate search order in `14_donor_assign.py`:

1. `Similarity/USGS_raw_nitrate_99133.parquet`
2. `Datasets/Nitrate_99133/USGS_MRBCB_raw_nitrate_154_99133.parquet`

Site coordinates CSV (first hit wins): `Datasets/sites_metadata_99133.csv` (or paths listed in the script).

## Rebuild Inputs

From this folder root:

```bash
# 1) Soft-sensor BASE / REDUCED / ALL (50-site) forcings
python Scripts/12_forcing_scenarios.py

# 2) Train/test splits + DNP + lstm_config per case
#    Writes Models/LSTM_SplitTest/Inputs/{gap}_{C1|C2|C3}_Seed{seed}_{scenario}/
python Scripts/13_train_test_split.py --force-populations

# 3) Attach REG / SIG / combined donors (needs seeds 21, 42, 84 per family)
python Scripts/14_donor_assign.py
```

Meta plots and split summaries go to `Models/Similarity/` (optional for training).

Gaps: `R20B14`, `R20B60`, `R20B120`, `R20B365`. Seeds: `21`, `42`, `84`.

| Set | Scenarios |
|-----|-----------|
| C0 | BASELINE only (the plain no-donor / no-extra-sensor run; same as each family’s `BASELINE` scenario) |
| C1 | BASELINE, REGDONOR, SIGDONOR, REGSIGDONOR |
| C2 | BASELINE, ALLSENSOR, REDUCEDSENSOR |
| C3 | BASELINE, REGDONOR, SIGDONOR, ALLSENSOR, REDUCEDSENSOR, REGSIGDONOR, REGSIGALLSENSOR |

Edit controls inside `Scripts/13_train_test_split.py` if you need a subset of cases.

## Train

```bash
cd Models/LSTM_SplitTest

python main_LSTM.py --config Inputs/R20B60_C1_Seed21_BASELINE/lstm_config_BASELINE.yaml
```

Batch / Slurm helper:

```bash
# dry-run
DRY_RUN=1 bash lstm_cases.sh

# submit (edit account/partition in the script header as needed)
sbatch lstm_cases.sh
```

Outputs land under `Models/LSTM_SplitTest/Outputs/{case_tag}/`.

## What is intentionally omitted
Prebuilt inputs and outputs are hosted at https://doi.org/10.5281/zenodo.23717919 
- Prebuilt `Inputs/` and `Outputs/` (large; regenerable)
- Plotting / table post-process scripts
- C4 single-site pipeline (`15_c4_single_site.py` → separate `LSTM_SingleSite` tree)
- Upstream raw data builders (`Scripts/01`–`11`)

