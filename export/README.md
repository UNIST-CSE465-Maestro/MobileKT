# MobileKT Mobile Export Guide

Export date: 2026-05-29  
Model: MobileKT v4 `QE-E2E+Teacher`  
Mode: stateful mobile KT-Engine

This folder contains the mobile-side MIKT KT-Engine artifacts and the contracts needed to connect it with the server-side Question Encoder.

## Runtime Split

```text
Server
  raw question + options + optional visual description
    -> Harrier feature
    -> MobileKT Question Encoder
    -> question_embedding[64], difficulty, concept_ids

Mobile
  question_embedding + difficulty + concept_ids + local student state
    -> mobile_mikt_predict.onnx
    -> pred_correct

  after learner response
  question_embedding + difficulty + concept_ids + response + local student state
    -> mobile_mikt_update.onnx
    -> next local student state
```

The mobile app should not run Harrier. The app owns and persists the MIKT state.

## Core Files

| file | role |
|---|---|
| `mobile_mikt_predict.onnx` | Predicts `pred_correct` from current question representation and local student state. |
| `mobile_mikt_update.onnx` | Updates local MIKT state after the learner response is known. |
| `mobile_mikt_initial_state.npz` | Initial state for a new learner profile. |
| `mikt_predict_contract.json` | ONNX input/output names, dtypes, shapes, padding, and call order. |
| `mikt_state_contract.json` | Meaning and storage contract for `skill_state`, `all_state`, and `last_skill_time`. |
| `concept_id_map.json` | Canonical concept key to integer id map. |
| `concept_catalog.json` | Human-readable concept catalog with source metadata. |
| `kc_mapping_contract.json` | Concept padding and mapping rules. |
| `qe_server_api_contract.json` | Server API contract for question encoding. |
| `qe_mikt_compatibility.json` | Compatibility metadata tying QE output to this mobile MIKT engine. |
| `export_validation.json` | Golden validation values for app-side wiring checks. |
| `validation_sample_input.json` | Sample question representation for validation. |
| `validation_expected_update_state.npz` | Expected state after the sample update. |
| `onnx_check_report.json` | ONNX checker result and model IO names. |
| `evaluation_report.json` | Model performance and calibration context. |
| `mobile_export_manifest.json` | File sizes and SHA-256 checksums. |
| `qe_server_question_encoder.pt` | Server-side QE head weights matching this mobile backbone. |
| `qe_server_config.json` | Minimal config for loading the QE head on the server. |

## Mobile State

The app stores one state bundle per learner profile:

```text
skill_state:      float32 [1, 641, 64]
all_state:        float32 [1, 64]
last_skill_time:  float32 [1, 641]
step:             float32 scalar or [1]
```

`skill_state[c]` is the concept-level latent knowledge vector. It is not directly a user-facing mastery percentage. A TAP/readout export should be used later for calibrated mastery display.

Column `0` in state tensors is reserved for padding. Do not display it as a real concept.

## Mobile Call Order

1. For a new learner, load `mobile_mikt_initial_state.npz`.
2. Request or fetch cached QE output from the server:
   `question_embedding`, `difficulty`, `concept_ids`.
3. Pad `concept_ids` to length `10` with `-1`.
4. Run `mobile_mikt_predict.onnx`.
5. Show or consume `pred_correct`.
6. After the learner answers, run `mobile_mikt_update.onnx`.
7. Persist `next_skill_state`, `next_all_state`, `next_last_skill_time`.
8. Increment the local `step`.

## ONNX Inputs

See `mikt_predict_contract.json` for the exact contract. Summary:

```text
question_embedding: float32 [batch, 64]
difficulty:         float32 [batch]
concept_ids:        int64   [batch, 10]
response:           int64   [batch]        only for update
skill_state:        float32 [batch, 641, 64]
all_state:          float32 [batch, 64]
last_skill_time:    float32 [batch, 641]
step:               float32 [batch]
```

The exported ONNX models use opset 18 and passed `onnx.checker`.

## Server Contract

The app should call:

```text
POST /v1/question/encode
```

The response must include:

```text
question_embedding: length 64 float32
difficulty: float32
concept_ids: integer ids compatible with concept_id_map.json
qe_model_version
mikt_compatibility_version
```

The `difficulty` value is MIKT internal latent difficulty. It is not Bloom difficulty.

## Validation Fixture

Use `validation_sample_input.json` with `mobile_mikt_initial_state.npz`.

Expected:

```text
pred_correct = 0.8227500915527344
```

After update, compare the outputs against `validation_expected_update_state.npz` or use the SHA-256 digests in `export_validation.json`.

## Source Model

Source checkpoint:

```text
maestro/MobileKT/experiments/statics2011_qe_e2e_teacher_guided_best_20260528/
  qe_seed2024_lr1e-03_dp0p1/
  statics2011_qe_trainable_id_seed2024_lr1e-03_dp0p1_seed2024_lr1e-03_dp0.1_q1_d1_logit1_kt1/
  qe_distill_best.pt
```

Test metrics are copied into `evaluation_report.json` and `source_metrics.json`.

Important caveat: this model was evaluated on the current student-level Statics2011 split. It is not yet an unseen-question split result.
