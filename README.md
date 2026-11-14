# SoundGuard-OL

Reproducible machine learning pipeline for **acoustic event classification** on the ESC-50 dataset.
The long-term goal is open-world acoustic event discovery (recognising known sounds and eventually
flagging unfamiliar ones). This repository is the MLOps foundation for that goal: validated and
versioned data, a DVC pipeline, a baseline and a main model, MLflow tracking on DagsHub, and a
registered model.

- GitHub: https://github.com/nabihanoor19/SoundGuard-OL
- DagsHub (data, DVC remote, MLflow): https://dagshub.com/nabihanoorkhalique/SoundGuard-OL
- MLflow experiment: https://dagshub.com/nabihanoorkhalique/SoundGuard-OL.mlflow/#/experiments/0

## What it does

| Stage (`dvc.yaml`) | Purpose |
|---|---|
| `validate_dataset` | Checks the ESC-50 metadata schema, nulls, label range (0-49), fold range (1-5), that all audio files exist, and sampling rate and duration. |
| `preprocess_audio` | Resamples audio to 22,050 Hz and extracts 64-band log-Mel spectrograms. Output shape: (2000, 64, 216). |
| `train_baseline` | Logistic Regression on the mean and standard deviation of each Mel band over time. Logged to MLflow. |
| `train_main` | `AcousticCNN` (PyTorch) trained on the log-Mel spectrograms. Logged to MLflow. |

Both models are trained on ESC-50 folds 1-4 (1,600 samples) and tested on fold 5 (400 samples).

| Metric (fold 5) | Logistic Regression (baseline) | Acoustic CNN (main) |
|---|---|---|
| Accuracy | 0.4400 | 0.3225 |
| Macro F1 | 0.4215 | 0.2659 |

The baseline model is registered in the MLflow Model Registry as `SoundGuard-Baseline`
(version 2, stage **Staging**).

## Setup

Requires Python 3.12 and Git. Commands are for Windows PowerShell; on Linux or macOS use
`source .venv/bin/activate` to activate the environment.

```powershell
git clone https://github.com/nabihanoor19/SoundGuard-OL.git
cd SoundGuard-OL

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### DagsHub credentials

The data lives on a DagsHub DVC remote (`origin`). Create an access token in
DagsHub (Settings, Tokens) and store it locally. These commands write to `.dvc/config.local`,
which is not committed:

```powershell
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <your-dagshub-username>
dvc remote modify origin --local password <your-dagshub-token>
```

The training scripts log to the DagsHub MLflow server. If prompted, authenticate with DagsHub, or set
`MLFLOW_TRACKING_USERNAME` (your DagsHub username) and `MLFLOW_TRACKING_PASSWORD` (your token).
Never commit tokens.

## How to run

```powershell
dvc pull          # download the ESC-50 dataset from the DagsHub remote
dvc repro         # run the full pipeline: validate, preprocess, train baseline, train CNN
dvc dag           # show the stage dependencies
dvc status -c     # check that the local cache and the remote are in sync
dvc push          # upload new data and outputs to the remote
```

To run only the validation step: `python data/validate.py`.
Parameters (sampling rate, Mel bands, training settings) are in `params.yaml`.

## Folder structure

```
SoundGuard-OL/
├── .dvc/                 DVC configuration (credentials go in config.local, not committed)
├── configs/              Project configuration files
├── data/
│   ├── raw/              ESC-50 dataset, tracked by DVC (ESC-50.dvc), not stored in Git
│   └── validate.py       Dataset validation script
├── models/               Model outputs (weights are ignored by Git)
├── reports/              Reports and evaluation outputs
├── src/
│   ├── preprocessing/
│   │   └── audio_features.py   Log-Mel feature extraction
│   └── models/
│       └── cnn.py              AcousticCNN (PyTorch)
├── dvc.yaml              Pipeline definition (four named stages)
├── dvc.lock              Exact dependencies and outputs of the last pipeline run
├── params.yaml           Pipeline and model parameters
├── requirements.txt      Python dependencies
├── .gitignore            Keeps raw data and model files out of Git
└── .dvcignore
```

The training scripts for the baseline and the CNN are the ones referenced by the
`train_baseline` and `train_main` stages in `dvc.yaml`.

## Reproducibility notes

- Raw data and model weights are not committed to Git. DVC tracks them and stores them on DagsHub.
- `dvc.lock` records the exact input and output hashes of every stage, and seeds are fixed (seed 42).
- Every training run logs its hyperparameters, metrics and model artifacts to MLflow on DagsHub.

## Team

- Nabiha Noor Khalique (23L-0967)
- Saarah Tauseef (23L-0552)

Course: MLOps, FAST NUCES.