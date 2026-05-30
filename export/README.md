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

Important distinction:

```text
Harrier model:
  external frozen text/image-question feature encoder

qe_server_question_encoder.pt:
  trained MobileKT QE head that maps Harrier feature[1024]
  -> MIKT-compatible question_embedding[64] and difficulty

mobile_mikt_*.onnx:
  trained stateful mobile KT engine
```

For existing Statics2011 benchmark questions, `question_harrier_features.pt`
was used during training as a precomputed Harrier feature cache. For new or
generated questions, the server must run Harrier at request time or use a
server-side representation cache.

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

The app should cache QE output by:

```text
question_hash + qe_model_version + mikt_compatibility_version
```

This representation cache is item-level data, not student state. It can be
shared across learner profiles on the same device. The MIKT state tensors are
student-specific and must be stored separately.

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

## Server Runtime

The server implementation is under:

```text
maestro/MobileKT/server/
```

Run it inside the Maestro Docker container:

```bash
cd /workspace/maestro/MobileKT
tools/run_qe_server.sh
```

By default this starts:

```text
host: 0.0.0.0
port: 8091
feature mode: harrier
```

If Harrier is not already cached in the container, allow the first request to
download it:

```bash
MOBILEKT_QE_ALLOW_MODEL_DOWNLOAD=1 tools/run_qe_server.sh
```

From the host machine, a convenient detached launch command is:

```bash
docker exec -d maestro_docker bash -lc '
cd /workspace/maestro/MobileKT
MOBILEKT_QE_HOST=0.0.0.0 \
MOBILEKT_QE_PORT=8091 \
MOBILEKT_QE_ALLOW_MODEL_DOWNLOAD=1 \
tools/run_qe_server.sh > /tmp/mobilekt_qe_server.log 2>&1
'
```

Check server status:

```bash
curl http://127.0.0.1:8091/healthz
```

Expected shape:

```json
{
  "status": "ok",
  "service": "mobilekt-qe",
  "qe_model_version": "qe-harrier-mobilekt4-d82c7b882b72",
  "mikt_compatibility_version": "mobilekt4-stat-20260528-qe-e2e-teacher-seed2024-dp0.1",
  "feature_mode": "harrier",
  "feature_dim": 1024,
  "embedding_dim": 64,
  "n_concepts": 640
}
```

For LAN/mobile testing, use the server host IP:

```text
http://<server-lan-ip>:8091
```

Example:

```text
http://192.168.0.2:8091
```

Docker compose publishes:

```text
0.0.0.0:8091 -> container:8091
```

If `curl http://127.0.0.1:8091/healthz` fails on the host but works inside the
container, recreate the container with the updated compose file:

```bash
cd /home/hserver/workspace
docker compose -f maestro/docker/docker-compose.yaml up -d --force-recreate --no-deps blackwell-dev
```

For a lightweight wiring test without loading Harrier, use:

```bash
MOBILEKT_QE_FEATURE_MODE=hash MOBILEKT_QE_DEVICE=cpu tools/run_qe_server.sh
```

`hash` mode is deterministic and useful for API/mobile wiring tests only. It is
not valid for research metrics or product predictions.

## Server Contract

The app should call:

```text
POST /v1/question/encode
```

or batch-prefetch a lesson with:

```text
POST /v1/question/encode-batch
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

The current exported QE head does not infer concept IDs by itself. The request
must include `concept_keys` or `concept_ids`, and the server validates them
against `concept_id_map.json`. Unknown concepts are rejected instead of being
silently guessed, because incorrect concept IDs would contaminate the learner's
local state.

### Single Question Request

```bash
curl -X POST http://127.0.0.1:8091/v1/question/encode \
  -H "Content-Type: application/json" \
  -d '{
    "client_question_id": "lesson1-q1",
    "question": "Which coordinate can be determined by inspection, using symmetry? x_G",
    "options": [
      {"label": "yes", "text": "Yes"},
      {"label": "no", "text": "No"}
    ],
    "visual_description": "",
    "concept_keys": ["find_symmetry_plane"],
    "question_type": "multiple_choice",
    "mikt_compatibility_version": "mobilekt4-stat-20260528-qe-e2e-teacher-seed2024-dp0.1"
  }'
```

Response shape:

```json
{
  "question_hash": "3f0ad13d81af2478528df932bb514b9ccc7d672705537752bd7f798f941e178e",
  "representation_id": "22a82c98eff5427343f7c11d407dec7f176a62f484e7b4670e237532aac50421",
  "qe_model_version": "qe-harrier-mobilekt4-d82c7b882b72",
  "mikt_compatibility_version": "mobilekt4-stat-20260528-qe-e2e-teacher-seed2024-dp0.1",
  "embedding_dim": 64,
  "embedding_dtype": "float32",
  "question_embedding": ["64 float32 values"],
  "difficulty": -1.018261194229126,
  "concept_ids": [583],
  "concept_keys": ["find_symmetry_plane"],
  "max_concepts_per_question": 10,
  "feature_encoder": "microsoft/harrier-oss-v1-0.6b",
  "feature_mode": "harrier"
}
```

The mobile app should convert:

```text
concept_ids: [583]
```

to:

```text
concept_ids: int64 [1, 10] = [[583, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
```

### Batch Prefetch Request

Use this before a lesson to reduce latency and support offline answering for
already-prefetched questions.

```bash
curl -X POST http://127.0.0.1:8091/v1/question/encode-batch \
  -H "Content-Type: application/json" \
  -d '{
    "mikt_compatibility_version": "mobilekt4-stat-20260528-qe-e2e-teacher-seed2024-dp0.1",
    "questions": [
      {
        "client_question_id": "lesson1-q1",
        "question": "Which coordinate can be determined by inspection, using symmetry? x_G",
        "options": [
          {"label": "yes", "text": "Yes"},
          {"label": "no", "text": "No"}
        ],
        "concept_keys": ["find_symmetry_plane"]
      },
      {
        "client_question_id": "lesson1-q2",
        "question": "Compute the centroid of the composite area.",
        "options": [],
        "concept_keys": ["centroid_of_composite_area"]
      }
    ]
  }'
```

Response shape:

```json
{
  "items": [
    {
      "client_question_id": "lesson1-q1",
      "status": "ok",
      "question_hash": "string",
      "representation_id": "string",
      "question_embedding": ["64 float32 values"],
      "difficulty": -1.0,
      "concept_ids": [583],
      "concept_keys": ["find_symmetry_plane"]
    }
  ],
  "qe_model_version": "qe-harrier-mobilekt4-d82c7b882b72",
  "mikt_compatibility_version": "mobilekt4-stat-20260528-qe-e2e-teacher-seed2024-dp0.1"
}
```

Each batch item has its own `status`. A malformed item can fail while other
items succeed.

### Common Server Errors

Missing concept metadata:

```json
{
  "error": {
    "code": "missing_concepts",
    "message": "concept_keys or concept_ids are required; current QE does not infer concepts."
  }
}
```

Unknown concept:

```json
{
  "error": {
    "code": "unknown_concept_key",
    "message": "Unknown concept key: ..."
  }
}
```

Version mismatch:

```json
{
  "error": {
    "code": "incompatible_mikt_version",
    "message": "Requested ..., server serves ..."
  }
}
```

Missing Harrier dependencies or cache:

```json
{
  "error": {
    "code": "internal_error",
    "message": "No module named 'transformers'"
  }
}
```

Fix by rebuilding the Docker image with the updated `maestro/docker/Dockerfile`
or installing `transformers` in the running container. If the model itself is
missing, run the server with `MOBILEKT_QE_ALLOW_MODEL_DOWNLOAD=1`.

## Mobile ONNX Wiring

After receiving a successful QE response, the app calls the mobile ONNX models.

### Predict Before Answer

Inputs to `mobile_mikt_predict.onnx`:

```text
question_embedding: float32 [1, 64]
difficulty:         float32 [1]
concept_ids:        int64   [1, 10]
skill_state:        float32 [1, 641, 64]
all_state:          float32 [1, 64]
last_skill_time:    float32 [1, 641]
step:               float32 [1]
```

Output:

```text
pred_correct: float32 [1]
```

### Update After Answer

Inputs to `mobile_mikt_update.onnx`:

```text
question_embedding: float32 [1, 64]
difficulty:         float32 [1]
concept_ids:        int64   [1, 10]
response:           int64   [1]      correct=1, incorrect=0
skill_state:        float32 [1, 641, 64]
all_state:          float32 [1, 64]
last_skill_time:    float32 [1, 641]
step:               float32 [1]
```

Outputs:

```text
next_skill_state:      float32 [1, 641, 64]
next_all_state:        float32 [1, 64]
next_last_skill_time:  float32 [1, 641]
```

Then persist:

```text
skill_state = next_skill_state
all_state = next_all_state
last_skill_time = next_last_skill_time
step = step + 1
```

State writes should be atomic with the local answer event. If the app crashes
after the learner answers but before state persistence, the event/state pair
can diverge.

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
