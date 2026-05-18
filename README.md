# MaestroML: Symbolic Music Feature Engineering and Temporal Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)

## Introduction

MaestroML is a lightweight symbolic music (MIDI) analysis pipeline built for CSE 153 Assignment 2. Rather than processing raw audio waveforms, it works directly on the symbolic structure of MIDI files — extracting a handcrafted 36-dimensional feature vector that encodes pitch, rhythm, dynamics, and voice structure. The goal is high-accuracy music classification and temporal reasoning at minimal computational cost, with features that remain musically interpretable.

## Key Features

**Task 1: Composer Classification**

Given a MIDI excerpt, the pipeline predicts which composer wrote it. Features are extracted along dimensions such as harmonic motion (pitch-class histogram), melodic interval distribution (stepwise vs. leap ratios), velocity dynamics, and polyphony level. A Random Forest classifier with 500 trees is trained on these 36-dimensional vectors and evaluated with 5-fold cross-validation.

**Task 2: Temporal Order Prediction**

Given two MIDI excerpts from the same piece, the model determines which comes first. The input is a 115-dimensional vector formed by concatenating the features of excerpt A, excerpt B, their element-wise difference, and boundary pitch features. Training uses a reverse-pair augmentation strategy: each labeled pair (A, B, 1) generates a symmetric pair (B, A, 0), doubling the dataset and teaching the model to capture temporal directionality.

**Generative Plugin**

A Markov-chain-based generator produces novel MIDI sequences conditioned on the style parameters derived from Task 1 composer profiles. Two output files are produced: `symbolic_conditioned.mid` (style-constrained) and `symbolic_unconditioned.mid` (unconstrained random walk).

## Tech Stack

- Language: Python 3.8+
- MIDI processing: `miditoolkit`, `mido`
- Feature engineering: `NumPy`, `SciPy`
- Machine learning: `scikit-learn` (RandomForestClassifier)
- Visualization: `matplotlib`, `seaborn`
- Notebook: `Jupyter`

## Project Structure

```
MaestroML/
├── README.md
├── requirements.txt
├── .gitignore
├── workbook.ipynb              # Main analysis notebook for peer review
├── data/                       # MIDI dataset (not tracked by git)
│   └── <composer_name>/        # One subdirectory per composer
│       └── *.mid
├── outputs/                    # Generated MIDI files
│   ├── symbolic_conditioned.mid
│   └── symbolic_unconditioned.mid
├── notebooks/                  # Scratch / exploratory notebooks
└── src/
    ├── extract_features.py     # 36-dim feature extraction from MIDI
    ├── task1_composer.py       # ComposerClassifier class
    ├── task2_temporal.py       # TemporalPredictor class (115-dim)
    └── generative_plugin.py    # Markov-chain MIDI generator
```

## Quick Start

**1. Clone the repository**

```bash
git clone https://github.com/Dennis-T-cloud/Music-Feature.git
cd MaestroML
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Prepare data**

Place MIDI files under `data/` with one subdirectory per composer:

```
data/
├── beethoven/
│   ├── sonata01.mid
│   └── sonata02.mid
└── chopin/
    ├── nocturne01.mid
    └── nocturne02.mid
```

**4. Run the workbook**

```bash
jupyter notebook workbook.ipynb
```

Run all cells from top to bottom. The notebook will extract features, train both classifiers, and write the two generated MIDI files to `outputs/`.

## Feature Vector Layout

The 36-dimensional feature vector produced by `extract_features.py` is structured as follows:

| Dimensions | Description |
|------------|-------------|
| [0:12] | Normalized pitch-class histogram |
| [12:16] | Pitch statistics (mean, std, min, max) |
| [16:20] | Duration statistics (mean, std, min, max) |
| [20:22] | Velocity statistics (mean, std) |
| [22:27] | Interval features (absolute mean, std, stepwise ratio, skip ratio, leap ratio) |
| [27] | Upward motion ratio |
| [28:30] | Inter-onset interval (mean, std) |
| [30] | Polyphony level |
| [31] | Note density (notes per second) |
| [32:36] | Reserved |

## Output Files

After running `workbook.ipynb`, the following files are written to `outputs/`:

- `symbolic_conditioned.mid` — generated with style parameters derived from the target composer's median feature profile
- `symbolic_unconditioned.mid` — generated with no style prior (random walk baseline)

Both files can be played back with any MIDI-capable application such as GarageBand, MuseScore, or fluidsynth.
