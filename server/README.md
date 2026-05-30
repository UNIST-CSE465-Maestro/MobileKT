# MobileKT Question Encoder Server

This folder exposes the server-side Question Encoder used by the mobile
MobileKT runtime. The mobile app owns the MIKT state and calls this server only
to obtain a question representation.

## Run

Inside the Maestro Docker container:

```bash
cd /workspace/maestro/MobileKT
python3 -m server.app --host 0.0.0.0 --port 8091 --device cuda
```

If the Harrier model is not already cached in the container, allow the first
startup/request to download it:

```bash
MOBILEKT_QE_ALLOW_MODEL_DOWNLOAD=1 tools/run_qe_server.sh
```

For a lightweight wiring test that does not load Harrier:

```bash
python3 -m server.app --host 0.0.0.0 --port 8091 --device cpu --feature_mode hash
```

The hash mode is deterministic but not valid for research metrics or product
predictions. Production should use the default Harrier mode.

## Endpoints

```text
GET  /healthz
POST /v1/question/encode
POST /v1/question/encode-batch
```

`concept_keys` or `concept_ids` are required because the current exported QE
head returns only `question_embedding` and `difficulty`; it does not infer
concept IDs.

## Smoke Test

```bash
cd /workspace/maestro/MobileKT
python3 -m server.golden_test --feature_mode hash --device cpu
```

Use `--feature_mode harrier` to validate the full production path after the
Harrier model is available in the container cache.
