# MobileKT

MobileKT is a Knowledge Tracing research codebase for experimenting with mobile-friendly KT models, question-aware representations, and MIKT-style backbones.

## Repository Layout

```text
MobileKT/
├── config.py
├── train.py
├── datasets/
├── models/
├── tools/
├── utils/
└── material/
```

## Main Files

- `config.py`  
  Dataclass-style configuration for datasets, model dimensions, training options, and MobileKT v4 settings.

- `train.py`  
  Main training entrypoint. Supports `mobilekt`, `mobilekt2`, `mobilekt3`, `mobilekt3b`, and `mobilekt4`.

## Folders

### `datasets/`

Dataset loading and batching utilities.

- `kt_dataset.py`  
  Loads MobileKT sequence CSV files, handles multi-concept sequences, masks padding, and optionally attaches cached question features.

- `__init__.py`  
  Exports dataset loader and collate helpers.

### `models/`

Model implementations.

- `mobilekt.py`, `mobilekt_v2.py`, `mobilekt_v3.py`, `mobilekt_v3b.py`  
  Earlier MobileKT model variants.

- `mobilekt_v4.py`  
  Current MobileKT v4 wrapper combining a Question Encoder with a MIKT-style backbone.

Subfolders:

- `models/qe/`  
  Question Encoder modules and feature-building scripts.
  - `question_encoder.py`: maps question IDs or cached raw-question features to MIKT-compatible `(question embedding, difficulty)` outputs.
  - `build_statics2011_text_features.py`: builds deterministic text feature cache.
  - `build_statics2011_harrier_features.py`: builds Harrier-based question feature cache.

- `models/backbone/`  
  KT backbone modules.
  - `mikt.py`: compact MIKT-style per-concept state tracker used by MobileKT v4.

- `models/irt/`  
  Prediction and IRT-style loss utilities.

- `models/knowledge/`  
  Knowledge state representation and gathering utilities for earlier MobileKT variants.

- `models/question_analysis/`  
  Question difficulty/effect/relationship modules used by earlier MobileKT variants.

- `models/updater/`  
  Concept and domain state update modules.

- `models/tap/`  
  TAP-related probing modules for mastery readout experiments.

### `tools/`

Experiment and preprocessing scripts.

- `prepare_statics2011_mobilekt.py`  
  Converts Statics2011 raw files into MobileKT sequence CSV format.

- `validate_statics2011_setup.py`  
  Checks Statics2011 preprocessing outputs.

- `run_statics2011_v4_compare.py`  
  Runs repeated MobileKT v4 comparisons between ID embedding and Harrier QE settings.

### `utils/`

Shared utility functions.

- `metrics.py`  
  AUC and accuracy computation with padding masks.

### `material/`

Research notes, architecture drafts, diagrams, and related reading material.

- `material/v1/`  
  Earlier MobileKT design notes and diagrams.

- `material/v2/`  
  Current architecture design notes for Question Encoder, MIKT backbone, and TAP direction.

- `material/Reading Materials/`  
  Reference papers used during architecture research.

## Example Training Commands

Train MobileKT v4 with ID-based question embeddings:

```bash
python train.py \
  --model mobilekt4 \
  --qe_input_mode id \
  --dataset statics2011 \
  --data_dir ../../data/datasets/KT \
  --d 64 \
  --mikt_state_dim 64 \
  --batch_size 32 \
  --n_epochs 100 \
  --patience 15 \
  --lr 1e-3 \
  --dropout 0.2 \
  --device cuda
```

Train MobileKT v4 with cached Harrier question features:

```bash
python train.py \
  --model mobilekt4 \
  --qe_input_mode features \
  --question_features_path ../../data/datasets/KT/statics2011/question_harrier_features.pt \
  --dataset statics2011 \
  --data_dir ../../data/datasets/KT \
  --d 64 \
  --mikt_state_dim 64 \
  --batch_size 32 \
  --n_epochs 100 \
  --patience 15 \
  --lr 1e-3 \
  --dropout 0.2 \
  --device cuda
```

