#!/usr/bin/env python3
"""
Prepare Statics 2011 raw student-step logs for MobileKT.

Input root:
    data/datasets/KT/statics2011/

Outputs in the same root by default:
    train.csv, valid.csv, test.csv, meta.json

MobileKT reserves id 0 for padding, so this script assigns question and concept
ids from 1. Multi-concept F2011 labels are encoded as colon-separated concept ids.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


FIRST_ATTEMPT_TO_RESPONSE = {"correct": 1, "incorrect": 0}


def latest_student_step_file(data_root: Path) -> Path:
    candidates = sorted(
        data_root.glob("ds507_student_step_*/ds507_student_step_All_Data_*.txt")
    )
    if not candidates:
        raise FileNotFoundError(f"student-step file not found under {data_root}")
    return candidates[-1]


def clean_token(value: object) -> str:
    text = str(value).strip()
    return "" if text == "." else text


def split_kcs(value: object) -> list[str]:
    text = clean_token(value)
    if not text:
        return []
    return [tok.strip() for tok in text.split("~~") if tok.strip()]


def stable_id_map(values: Iterable[str]) -> dict[str, int]:
    return {value: idx + 1 for idx, value in enumerate(sorted(set(values)))}


def make_question_key(problem_name: str, step_name: str) -> str:
    return f"{problem_name.strip()}----{step_name.strip()}"


def sequence_to_row(uid: str, interactions: list[dict]) -> dict | None:
    if len(interactions) < 2:
        return None
    return {
        "uid": uid,
        "questions": ",".join(str(x["question_id"]) for x in interactions),
        "concepts": ",".join(":".join(map(str, x["concept_ids"])) for x in interactions),
        "responses": ",".join(str(x["response"]) for x in interactions),
    }


def write_split(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows, columns=["uid", "questions", "concepts", "responses"]).to_csv(
        path, index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/datasets/KT/statics2011")
    parser.add_argument("--out_dir", default="data/datasets/KT/statics2011")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step_path = latest_student_step_file(data_root)
    df = pd.read_csv(step_path, sep="\t", dtype=str, keep_default_na=False)
    df = df[df["First Attempt"].isin(FIRST_ATTEMPT_TO_RESPONSE)].copy()

    df["_question_key"] = [
        make_question_key(problem, step)
        for problem, step in zip(df["Problem Name"], df["Step Name"])
    ]
    df["_kc_list"] = [split_kcs(x) for x in df["KC (F2011)"]]

    # Keep rows without F2011 labels usable, but make the fallback explicit.
    fallback_unique = [split_kcs(x) for x in df["KC (Unique-step)"]]
    df["_kc_list"] = [
        f2011 if f2011 else (unique if unique else ["__unknown_kc__"])
        for f2011, unique in zip(df["_kc_list"], fallback_unique)
    ]

    question2id = stable_id_map(df["_question_key"])
    concept2id = stable_id_map(kc for kcs in df["_kc_list"] for kc in kcs)

    df["_time"] = pd.to_datetime(
        df["First Transaction Time"].where(
            df["First Transaction Time"].astype(bool), df["Step Start Time"]
        ),
        errors="coerce",
    )
    df["_row_int"] = pd.to_numeric(df["Row"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["Anon Student Id", "_time", "_row_int"])

    by_user: dict[str, list[dict]] = defaultdict(list)
    for _, row in df.iterrows():
        uid = row["Anon Student Id"]
        by_user[uid].append(
            {
                "question_id": question2id[row["_question_key"]],
                "concept_ids": [concept2id[kc] for kc in row["_kc_list"]],
                "response": FIRST_ATTEMPT_TO_RESPONSE[row["First Attempt"]],
            }
        )

    users = sorted(by_user)
    rng = random.Random(args.seed)
    rng.shuffle(users)

    n_train = int(len(users) * args.train_ratio)
    n_valid = int(len(users) * args.valid_ratio)
    split_users = {
        "train": users[:n_train],
        "valid": users[n_train : n_train + n_valid],
        "test": users[n_train + n_valid :],
    }

    split_rows = {}
    for split, split_uid in split_users.items():
        rows = [sequence_to_row(uid, by_user[uid]) for uid in split_uid]
        split_rows[split] = [row for row in rows if row is not None]
        write_split(out_dir / f"{split}.csv", split_rows[split])

    meta = {
        "dataset": "statics2011",
        "source": str(step_path),
        "format": "MobileKT sequence CSV",
        "split": "student-level random split",
        "seed": args.seed,
        "n_questions": len(question2id),
        "n_concepts": len(concept2id),
        "n_students": len(users),
        "n_interactions": int(len(df)),
        "n_train_sequences": len(split_rows["train"]),
        "n_valid_sequences": len(split_rows["valid"]),
        "n_test_sequences": len(split_rows["test"]),
        "question_id_base": 1,
        "concept_id_base": 1,
        "padding_id": 0,
        "question_key_format": "Problem Name----Step Name",
        "concept_source": "KC (F2011), fallback KC (Unique-step), fallback __unknown_kc__",
        "files": {
            "question2id": "question2id.json",
            "concept2id": "concept2id.json",
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "question2id.json").write_text(json.dumps(question2id, indent=2, sort_keys=True))
    (out_dir / "concept2id.json").write_text(json.dumps(concept2id, indent=2, sort_keys=True))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
