#!/usr/bin/env python3
"""Build Harrier embeddings for MobileKT v4 Question Encoder.

This is the document-aligned authoring-time path:

    Question + Options + Optional Visual Description
        -> microsoft/harrier-oss-v1-0.6b
        -> question_harrier_features.pt
        -> MobileKTV4(question_features=...)

The output contract matches ``KTDataset(question_features_path=...)``:

    {
      "features": FloatTensor[n_questions + 1, 1024],
      "feature_dim": 1024,
      "encoder": "microsoft/harrier-oss-v1-0.6b",
      ...
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


def format_raw_question(row: dict) -> str:
    options = row.get("options") or []
    option_lines = [
        f"{opt.get('label', '')}. {opt.get('text', '')}".strip()
        for opt in options
        if opt
    ]
    pieces = [
        "Question:",
        str(row.get("question") or "").strip(),
    ]
    if option_lines:
        pieces.extend(["", "Options:", *option_lines])
    visual = str(row.get("visual_description") or "").strip()
    if visual:
        pieces.extend(["", "Visual Description:", visual])
    return "\n".join(piece for piece in pieces if piece != "")


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def load_rows(manifest: Path, n_questions: int) -> tuple[list[int], list[str], int]:
    qids: list[int] = []
    texts: list[str] = []
    rows = 0
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
            text = format_raw_question(row)
            if not text.strip():
                continue
            qids.append(qid)
            texts.append(text)
    return qids, texts, rows


@torch.inference_mode()
def encode_texts(
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, dtype="auto")
    model.eval()
    model.to(device)

    chunks = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        chunks.append(embeddings.cpu().float())
        print(f"encoded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/workspace/data/datasets/KT/statics2011")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--question2id", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--model_name", default="microsoft/harrier-oss-v1-0.6b")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    manifest = Path(args.manifest) if args.manifest else data_root / "qe_manifest.jsonl"
    question2id_path = Path(args.question2id) if args.question2id else data_root / "question2id.json"
    out_path = Path(args.out) if args.out else data_root / "question_harrier_features.pt"

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    question2id = json.loads(question2id_path.read_text())
    n_questions = max(int(v) for v in question2id.values()) if question2id else 0
    qids, texts, rows = load_rows(manifest, n_questions)

    row_embeddings = encode_texts(
        texts=texts,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    if row_embeddings.numel() == 0:
        raise RuntimeError(f"No matched question text rows found in {manifest}")

    feature_dim = int(row_embeddings.shape[-1])
    sums = torch.zeros(n_questions + 1, feature_dim, dtype=torch.float32)
    counts = torch.zeros(n_questions + 1, dtype=torch.float32)
    for qid, emb in zip(qids, row_embeddings):
        sums[qid] += emb
        counts[qid] += 1.0

    nonzero = counts > 0
    features = sums.clone()
    features[nonzero] = features[nonzero] / counts[nonzero].unsqueeze(-1)
    features[nonzero] = F.normalize(features[nonzero], p=2, dim=-1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features,
            "feature_dim": feature_dim,
            "source": str(manifest),
            "encoder": args.model_name,
            "rows": rows,
            "matched_rows": len(qids),
            "covered_questions": int(nonzero.sum().item()),
            "max_length": args.max_length,
        },
        out_path,
    )
    print(
        json.dumps(
            {
                "out": str(out_path),
                "encoder": args.model_name,
                "feature_dim": feature_dim,
                "rows": rows,
                "matched_rows": len(qids),
                "covered_questions": int(nonzero.sum().item()),
                "n_questions": n_questions,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
