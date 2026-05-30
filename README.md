# MobileKT

MobileKT is a Knowledge Tracing research codebase for experimenting with mobile-friendly KT models, question-aware representations, and MIKT-style backbones.

## Repository Layout

```text
MobileKT/
├── config.py
├── train.py
├── datasets/
├── models/
├── server/
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

- `train_qe_distill.py`  
  Trains a Harrier-feature Question Encoder against a MIKT-ID teacher. Supports frozen-backbone QE distillation and trainable-backbone teacher-guided joint fine-tuning.

- `run_statics2011_qe_distill.py`  
  Launches repeated Statics2011 QE distillation or teacher-guided joint runs from the saved ID-teacher checkpoints.

- `run_qe_server.sh`  
  Starts the server-side Question Encoder API for mobile app integration.

### `server/`

Server-side Question Encoder runtime for mobile apps.

- `service.py`  
  Loads the exported QE head, Harrier feature encoder, compatibility metadata, and concept map.

- `app.py`  
  Exposes `GET /healthz`, `POST /v1/question/encode`, and `POST /v1/question/encode-batch` through a stdlib HTTP server.

- `golden_test.py`  
  Smoke test for API wiring and exported QE compatibility.

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

Train the MIKT-first frozen-backbone Question Encoder distillation setting:

```bash
python tools/run_statics2011_qe_distill.py \
  --preset best \
  --teacher_session statics2011_v4_compare_20260526_dropout \
  --session statics2011_qe_distill_best \
  --gpus 0
```

Use `--preset core` for all three seeds at dropout 0.2, or `--preset dropout` to mirror the earlier seed/dropout grid.

Train the teacher-guided joint setting:

```bash
python tools/run_statics2011_qe_distill.py \
  --preset best \
  --teacher_session statics2011_v4_compare_20260526_dropout \
  --session statics2011_qe_e2e_teacher_guided_best \
  --gpus 0 \
  --backbone_mode trainable \
  --kt_loss_weight 1.0 \
  --q_loss_weight 1.0 \
  --diff_loss_weight 1.0 \
  --logit_loss_weight 1.0
```

## Mobile QE Server

Run inside the Maestro Docker container:

```bash
cd /workspace/maestro/MobileKT
tools/run_qe_server.sh
```

The Docker compose file publishes port `8091`, so after recreating the
container the mobile app can call:

```text
GET  http://<host-ip>:8091/healthz
POST http://<host-ip>:8091/v1/question/encode
POST http://<host-ip>:8091/v1/question/encode-batch
```

For a lightweight wiring test without loading Harrier:

```bash
MOBILEKT_QE_FEATURE_MODE=hash MOBILEKT_QE_DEVICE=cpu tools/run_qe_server.sh
```

`hash` mode is only for integration testing. Product/research runs should use
the default Harrier mode.
