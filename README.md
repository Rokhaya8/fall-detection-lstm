# Fall Detection with LSTM - SisFall & Daphnet

Detecting falls from wearable accelerometer signals with an LSTM network. The model is
trained and evaluated on **SisFall**, then tested on **Daphnet** (Parkinson's disease
patients) to check how well it transfers to a different population and task.

> **Team project.** This repository contains my part of a team project on Parkinson's
> disease and fall detection (ENSA, 2025–2026). The exploratory analyses and the
> preprocessing pipelines were built together with the team; the **LSTM model** — its
> hyperparameter search, training and evaluation — was my part. The GRU and CNN models
> compared in the project were developed by other team members and are not included here.

## Highlights

- **No data leakage**: the train / validation / test split is done **by subject**, and the
  standardization is fitted on the training set only. The results measure performance
  on people the model has never seen.
- **Systematic model selection**: hyperparameters chosen with a random search tracked in
  Weights & Biases, instead of manual trial and error.
- **Decisions driven by the use case**: class weighting and a lowered decision threshold
  favour recall, because a missed fall is more costly than a false alarm.
- **Honest evaluation**: a transfer test on a second dataset, and an analysis of *why*
  the model fails where it fails (label granularity, different task and sensor
  placement), with the first improvements to make.
- **Reproducible**: notebooks run in order on Colab, with a single configurable path,
  and the trained model is provided to reproduce the results without retraining.

## The Problem

Falls are a major risk for elderly people and for patients with Parkinson's disease.
A wearable sensor that detects a fall automatically could trigger an alert when the
person cannot call for help. The goal is to classify short windows of accelerometer
signal as **fall** or **normal activity**.

## Data

| Dataset | Content | Use |
|---------|---------|-----|
| [SisFall](https://www.kaggle.com/datasets/nvnikhil0001/sis-fall-original-dataset) | 38 young and elderly participants, simulated falls and daily activities, sensor at the waist, 200 Hz | Training, validation, test |
| [Daphnet Freezing of Gait](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait) | 10 Parkinson's patients, sensors at the ankle, thigh and trunk, 64 Hz, labelled for *freezing of gait* | Transfer test |

The datasets are not included in this repository: download them from the links above.

## Pipeline

```
Raw signals → cleaning → windowing → split by subject → standardization → LSTM → evaluation
```

| Notebook | Step |
|----------|------|
| `00_eda_sisfall` / `00_eda_daphnet` | Exploratory analysis of both datasets |
| `01_preprocessing_sisfall` | Cleaning, unit conversion (bits → g), windowing, split by subject, standardization |
| `02_preprocessing_daphnet` | Same pipeline for Daphnet, with resampling from 64 Hz to 200 Hz |
| `03_lstm_hyperparameter_search` | Random search with Weights & Biases (15 runs, Hyperband early stopping) |
| `04_lstm_training` | Training of the final model |
| `05_evaluation_sisfall` | Evaluation on the SisFall test set |
| `06_evaluation_daphnet` | Transfer test on Daphnet |

### Key Preprocessing Choices

- **1-second windows** (200 samples at 200 Hz) with 50 % overlap; a window is labelled
  *fall* if more than half of its samples are.
- **Split by subject**: each participant is entirely in train, validation *or* test.
  A random split would put windows of the same person in both train and test, and the
  model would partly recognize people instead of falls.
- **Standardization fitted on the training set only**, then applied to validation and
  test, to avoid any information leak from the test data.
- **Daphnet resampled to 200 Hz**, so that a 200-sample window covers one second in both
  datasets.

### Model

| Parameter | Value |
|-----------|-------|
| Architecture | 1 LSTM layer (128 units) → Dropout (0.3) → Dense (sigmoid) |
| Parameters | 69,249 |
| Optimizer | Adam, learning rate 0.0001, batch size 256 |
| Class weights | 1 (activity) / 3 (fall) |
| Decision threshold | 0.40 |

The decision threshold is lowered from 0.50 to 0.40 because missing a fall is more
costly than a false alarm.

## Results

| Test set | Recall | F1-score | False positive rate | ROC-AUC |
|----------|--------|----------|---------------------|---------|
| SisFall (unseen subjects) | **82.0 %** | 67.9 % | 29.1 % | — |
| Daphnet (transfer test) | 97.3 % | 21.1 % | 77.8 % | 0.59 |

On SisFall, the model detects 82 % of falls on participants it has never seen, at the
cost of a high false alarm rate. On Daphnet, it labels almost every window as an
anomaly: the model does not transfer.

## Analysis and Limitations

**Labels are too broad.** A SisFall fall recording lasts about 15 seconds: the person
walks, falls, then lies on the ground. Every window of a fall recording is labelled
*fall*, although the fall itself lasts about one second. The model therefore learns that
walking or lying can be a fall, which explains most of the false alarms. Labelling only
the windows around the acceleration peak would be the first improvement to make.

**Daphnet is a different task.** Its labels mark *freezing of gait* — a sudden inability
to keep walking, typical of Parkinson's disease — not falls. The sensors are also placed
differently (ankle and trunk instead of waist). The poor results show that a model
trained to detect falls cannot be reused as is to detect freezing of gait; they do not
measure fall detection on Parkinson's patients. Adapting the model would require
retraining or fine-tuning on Daphnet.

**Other limitations.**

- SisFall falls are simulated by volunteers, not real falls.
- Windows are created on the concatenated signal, so a few windows overlap two
  recordings.
- The reported latency (~100 ms) measures single calls to `model.predict` in Colab and
  is dominated by call overhead, not by the model itself.

## How to Run

The notebooks run on **Google Colab** and read their data from **Google Drive**.

### 1. Create the project folder in your Google Drive

```
MyDrive/
└── Projet_Parkinson/
    ├── Datasets_bruts/
    └── Models/
```

The other folders (`Data_Preprocessed/`, `Figures/`) are created automatically by the
notebooks.

### 2. Add the data and the model

- Download **SisFall** and **Daphnet** (links in the [Data](#data) section) and put the
  zip files in `Datasets_bruts/`. Rename them `SisFall_dataset.zip` and
  `Daphnet_dataset.zip` (downloads are often named `archive.zip`). The internal
  structure of the zips does not matter: the notebooks find the data files
  automatically.
- Download `models/LSTM_final.keras` from this repository and put it in `Models/`.

### 3. Run the notebooks

Open each notebook with its **Open in Colab** button, then use
**Runtime → Run all**. Allow Colab to access your Google Drive when asked.

| Goal | Notebooks to run, in order |
|------|---------------------------|
| Reproduce the results with the provided model | `01` → `02` → `05` → `06` |
| Retrain everything | `01` → `02` → `03` → `04` → `05` → `06` |

Notebook `03` requires a free [Weights & Biases](https://wandb.ai) account. For
notebook `04`, select a GPU in Colab (**Runtime → Change runtime type**).

### Using another folder

All paths are built from `BASE_DIR`, defined in the first code cell of each notebook:

```python
BASE_DIR = '/content/drive/MyDrive/Projet_Parkinson'
```

`/content/drive/MyDrive` is the root of your Google Drive in Colab. If you follow step 1,
there is nothing to change. To store the project elsewhere, change this line in each
notebook.

## Tools

Python, TensorFlow / Keras, scikit-learn, NumPy, pandas, SciPy, Matplotlib, Seaborn,
Weights & Biases, Google Colab
