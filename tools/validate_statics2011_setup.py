#!/usr/bin/env python3
"""Validate the Statics 2011 MobileKT/QE setup without importing torch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_ids(value: object, sep: str = ",") -> list[str]:
    return [x for x in str(value).split(sep) if x.strip()]


def validate_split(path: Path, meta: dict) -> dict:
    df = pd.read_csv(path)
    n_interactions = 0
    max_q = 0
    max_c = 0
    bad_rows = 0

    for _, row in df.iterrows():
        qs = [int(x) for x in parse_ids(row["questions"])]
        rs = [int(x) for x in parse_ids(row["responses"])]
        cs_tokens = parse_ids(row["concepts"])
        if not (len(qs) == len(rs) == len(cs_tokens)):
            bad_rows += 1
            continue
        n_interactions += len(qs)
        if qs:
            max_q = max(max_q, max(qs))
        for token in cs_tokens:
            concepts = [int(x) for x in parse_ids(token, ":")]
            if concepts:
                max_c = max(max_c, max(concepts))

    assert bad_rows == 0, f"{path}: {bad_rows} malformed rows"
    assert max_q <= meta["n_questions"], f"{path}: max q id {max_q} > meta n_questions"
    assert max_c <= meta["n_concepts"], f"{path}: max c id {max_c} > meta n_concepts"
    return {
        "rows": len(df),
        "interactions": n_interactions,
        "max_question_id": max_q,
        "max_concept_id": max_c,
    }


def validate_manifest(path: Path) -> dict:
    rows = 0
    matched = 0
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            rows += 1
            if row.get("matched_question_id") is not None:
                matched += 1
    return {"rows": rows, "matched_question_id": matched}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/datasets/KT/statics2011")
    args = parser.parse_args()

    root = Path(args.data_root)
    meta = json.loads((root / "meta.json").read_text())
    result = {"meta": meta, "splits": {}}
    for split in ["train", "valid", "test"]:
        result["splits"][split] = validate_split(root / f"{split}.csv", meta)
    result["qe_manifest"] = validate_manifest(root / "qe_manifest.jsonl")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
