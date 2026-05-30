#!/usr/bin/env python3
"""Golden smoke test for the MobileKT Question Encoder service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .service import QuestionRepresentationService, ServiceConfig, ServiceError


SAMPLE_PAYLOAD = {
    "client_question_id": "golden-find-symmetry-plane",
    "question": (
        "You need to determine the location of the center of gravity of the body shown. "
        "Which coordinate can be determined by inspection, using symmetry? x_G"
    ),
    "options": [
        {"label": "yes", "text": "Yes"},
        {"label": "no", "text": "No"},
    ],
    "visual_description": "",
    "concept_keys": ["find_symmetry_plane"],
    "question_type": "multiple_choice",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_dir", default=str(Path(__file__).resolve().parents[1] / "export"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--feature_mode",
        choices=["harrier", "hash"],
        default="hash",
        help="Use hash for a lightweight wiring test; use harrier for production parity.",
    )
    parser.add_argument("--allow_model_download", action="store_true")
    args = parser.parse_args()

    service = QuestionRepresentationService(
        ServiceConfig(
            export_dir=Path(args.export_dir),
            device=args.device,
            feature_mode=args.feature_mode,
            local_files_only=not args.allow_model_download,
        )
    )
    try:
        response = service.encode_one(SAMPLE_PAYLOAD)
    except ServiceError as exc:
        raise SystemExit(json.dumps(exc.to_response(), indent=2, ensure_ascii=False))

    checks = {
        "embedding_len_ok": len(response["question_embedding"]) == 64,
        "difficulty_is_float": isinstance(response["difficulty"], float),
        "concept_ids_ok": response["concept_ids"] == [583],
        "version_ok": bool(response["mikt_compatibility_version"]),
        "hash_ok": len(response["question_hash"]) == 64,
    }
    ok = all(checks.values())
    print(
        json.dumps(
            {
                "ok": ok,
                "checks": checks,
                "summary": {
                    "question_hash": response["question_hash"],
                    "representation_id": response["representation_id"],
                    "qe_model_version": response["qe_model_version"],
                    "mikt_compatibility_version": response["mikt_compatibility_version"],
                    "difficulty": response["difficulty"],
                    "concept_ids": response["concept_ids"],
                    "feature_mode": response["feature_mode"],
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
