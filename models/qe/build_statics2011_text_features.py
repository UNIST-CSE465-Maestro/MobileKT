#!/usr/bin/env python3
"""Build cached raw-question features for MobileKT v4.

This script is an authoring-time bridge from ``qe_manifest.jsonl`` to the
feature matrix consumed by ``MIKTQuestionEncoder(input_mode="features")``.

The research target is to replace this lightweight hashing encoder with
Harrier/LLM embeddings while keeping the output contract identical:

    question_text_features.pt
      {
        "features": FloatTensor[n_questions + 1, feature_dim],
        "feature_dim": int,
        "source": str,
        "encoder": str
      }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def format_raw_question(row: dict) -> str:
    options = row.get("options") or []
    option_text = " ".join(
        f"{opt.get('label', '')} {opt.get('text', '')}".strip()
        for opt in options
        if opt
    )
    pieces = [
        str(row.get("question") or ""),
        option_text,
        str(row.get("visual_description") or ""),
    ]
    return "\n".join(piece for piece in pieces if piece.strip())


def stable_bucket(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="little", signed=False)
    bucket = value % dim
    sign = 1.0 if ((value >> 63) & 1) == 0 else -1.0
    return bucket, sign


def encode_text(text: str, dim: int) -> torch.Tensor:
    vec = torch.zeros(dim, dtype=torch.float32)
    for token in TOKEN_RE.findall(text.lower()):
        bucket, sign = stable_bucket(token, dim)
        vec[bucket] += sign
    return F.normalize(vec, dim=0) if vec.norm() > 0 else vec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/workspace/data/datasets/KT/statics2011")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--question2id", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--feature_dim", type=int, default=512)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    manifest = Path(args.manifest) if args.manifest else data_root / "qe_manifest.jsonl"
    question2id_path = Path(args.question2id) if args.question2id else data_root / "question2id.json"
    out_path = Path(args.out) if args.out else data_root / "question_text_features.pt"

    question2id = json.loads(question2id_path.read_text())
    n_questions = max(int(v) for v in question2id.values()) if question2id else 0

    sums = torch.zeros(n_questions + 1, args.feature_dim, dtype=torch.float32)
    counts = torch.zeros(n_questions + 1, dtype=torch.float32)
    rows = 0
    matched = 0

    with manifest.open() as fin:
        for line in fin:
            row = json.loads(line)
            rows += 1
            qid = row.get("matched_question_id")
            if qid is None:
                continue
            qid = int(qid)
            if qid <= 0 or qid > n_questions:
                continue
            sums[qid] += encode_text(format_raw_question(row), args.feature_dim)
            counts[qid] += 1.0
            matched += 1

    nonzero = counts > 0
    features = sums.clone()
    features[nonzero] = features[nonzero] / counts[nonzero].unsqueeze(-1)
    features[nonzero] = F.normalize(features[nonzero], dim=-1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features,
            "feature_dim": args.feature_dim,
            "source": str(manifest),
            "encoder": "hashing_raw_question_v0",
            "rows": rows,
            "matched_rows": matched,
            "covered_questions": int(nonzero.sum().item()),
        },
        out_path,
    )
    print(
        json.dumps(
            {
                "out": str(out_path),
                "feature_dim": args.feature_dim,
                "rows": rows,
                "matched_rows": matched,
                "covered_questions": int(nonzero.sum().item()),
                "n_questions": n_questions,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
