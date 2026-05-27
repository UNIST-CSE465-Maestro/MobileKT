"""
KT Dataset

Loads standard KT benchmark datasets and wraps them in a PyTorch Dataset.
Supports multi-concept questions (each question can map to multiple concepts).

Expected CSV format (compatible with pykt-toolkit preprocessed data):
    - columns: student_id, question_id, concept_ids (comma-separated), correct, timestamp

Multi-concept handling:
    concept_ids is a comma-separated string, e.g. "3,7,12"
    These are padded to max_concepts_per_question with -1.
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Optional


class KTDataset(Dataset):
    def __init__(
        self,
        data: list[dict],
        max_seq_len: int = 200,
        max_concepts: int = 10,
        question_features: torch.Tensor | None = None,
    ):
        """
        Args:
            data:         list of student sequences, each a dict with:
                              'question_ids'  : list[int]
                              'concept_ids'   : list[list[int]]  (multi-concept)
                              'responses'     : list[int]  {0,1}
            max_seq_len:  truncate/pad sequences to this length
            max_concepts: max number of concepts per question (pad with -1)
        """
        self.data = data
        self.max_seq_len = max_seq_len
        self.max_concepts = max_concepts
        self.question_features = question_features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        q_ids = seq["question_ids"]
        c_ids = seq["concept_ids"]      # list of lists
        resp  = seq["responses"]

        L = len(q_ids)
        pad_len = self.max_seq_len - L

        # Pad / truncate question ids
        q_ids = (q_ids + [0] * pad_len)[:self.max_seq_len]
        resp  = (resp  + [0] * pad_len)[:self.max_seq_len]

        # Pad concept ids: (S, max_c)
        c_padded = []
        for cs in c_ids:
            cs = cs[:self.max_concepts]
            cs = cs + [-1] * (self.max_concepts - len(cs))
            c_padded.append(cs)
        c_padded += [[-1] * self.max_concepts] * pad_len
        c_padded = c_padded[:self.max_seq_len]

        item = {
            "question_ids": torch.tensor(q_ids, dtype=torch.long),
            "concept_ids":  torch.tensor(c_padded, dtype=torch.long),
            "responses":    torch.tensor(resp, dtype=torch.float),
            "length":       torch.tensor(min(L, self.max_seq_len), dtype=torch.long),
        }
        if self.question_features is not None:
            item["question_features"] = self.question_features[
                torch.tensor(q_ids, dtype=torch.long)
            ].float()
        return item


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ─── Dataset loaders ──────────────────────────────────────────────────────────

def load_dataset(
    dataset_name: str,
    data_dir: str,
    max_seq_len: int = 200,
    max_concepts: int = 10,
    seed: int = 42,
    question_features_path: str = "",
) -> tuple["KTDataset", "KTDataset", "KTDataset", dict]:
    """
    Load a KT dataset by name and return train/val/test splits + metadata.

    Expects data prepared by data/prepare_data.py:
        data/{dataset}/train.csv, valid.csv, test.csv, meta.json

    Returns:
        train_ds, val_ds, test_ds, meta
        meta: dict with n_questions, n_concepts
    """
    data_path = os.path.join(data_dir, dataset_name)
    assert os.path.isdir(data_path), (
        f"Dataset directory not found: {data_path}\n"
        f"Run: python3 data/prepare_data.py --dataset {dataset_name}"
    )

    meta_path = os.path.join(data_path, "meta.json")
    assert os.path.exists(meta_path), f"meta.json not found in {data_path}"
    with open(meta_path) as f:
        meta = json.load(f)

    question_features = None
    if question_features_path:
        question_features = _load_question_features(question_features_path)
        meta = dict(meta)
        meta["question_feature_dim"] = int(question_features.shape[-1])

    def load_split(split_name):
        csv_path = os.path.join(data_path, f"{split_name}.csv")
        assert os.path.exists(csv_path), f"{split_name}.csv not found in {data_path}"
        return KTDataset(
            _parse_csv(csv_path),
            max_seq_len,
            max_concepts,
            question_features=question_features,
        )

    train_ds = load_split("train")
    val_ds   = load_split("valid")
    test_ds  = load_split("test")

    return train_ds, val_ds, test_ds, meta


def _load_question_features(path: str) -> torch.Tensor:
    if path.endswith(".npy"):
        features = torch.from_numpy(np.load(path)).float()
    else:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            if "features" not in obj:
                raise KeyError(f"question feature file missing 'features': {path}")
            features = obj["features"].float()
        else:
            features = obj.float()
    if features.dim() != 2:
        raise ValueError("question feature matrix must have shape (n_questions+1, feature_dim)")
    return features


def _parse_csv(csv_path: str) -> list[dict]:
    """
    Parse a prepared CSV (train/valid/test) where each row is one student sequence:
        questions : comma-separated question IDs (1-indexed)
        concepts  : comma-separated concept IDs (colon-sep for multi-concept)
        responses : comma-separated {0,1}
    """
    df = pd.read_csv(csv_path)
    sequences = []

    for _, row in df.iterrows():
        q_ids = [int(x) for x in str(row["questions"]).split(",") if x.strip()]
        c_ids = []
        for tok in str(row["concepts"]).split(","):
            concepts = [int(x) for x in tok.split(":") if x.strip().lstrip("-").isdigit() and int(x) >= 0]
            c_ids.append(concepts if concepts else [0])
        resp = [int(x) for x in str(row["responses"]).split(",") if x.strip() in ("0", "1")]

        min_len = min(len(q_ids), len(c_ids), len(resp))
        q_ids, c_ids, resp = q_ids[:min_len], c_ids[:min_len], resp[:min_len]
        if min_len < 2:
            continue

        sequences.append({"question_ids": q_ids, "concept_ids": c_ids, "responses": resp})

    return sequences
