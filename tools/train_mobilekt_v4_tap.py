#!/usr/bin/env python3
"""Train a post-hoc TAP mastery readout for a frozen MobileKT v4 backbone.

The exported mobile engine stores MIKT concept vectors, not user-facing
mastery probabilities. This script replays a trained MobileKT v4 checkpoint,
extracts concept-aligned states after each observed interaction, and trains a
small TAP readout against future same-concept correctness labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import MobileKTConfig
from datasets import collate_fn, load_dataset
from models import MobileKTV4
from models.tap import (
    EbbinghausTimeAwareProbe,
    TimeAwareProbe,
    TimeAwareProbeConfig,
    build_future_correctness_labels,
    build_timer_features,
    forgetting_monotonic_loss,
    soft_binary_cross_entropy,
)


DEFAULT_CHECKPOINT = (
    ROOT
    / "experiments"
    / "statics2011_qe_e2e_teacher_guided_best_20260528"
    / "qe_seed2024_lr1e-03_dp0p1"
    / "statics2011_qe_trainable_id_seed2024_lr1e-03_dp0p1_seed2024_lr1e-03_dp0.1_q1_d1_logit1_kt1"
    / "qe_distill_best.pt"
)
DEFAULT_OUT_DIR = ROOT / "experiments" / "statics2011_mobilekt4_tap_export"
DEFAULT_DATA_DIR = ROOT.parents[1] / "data" / "datasets" / "KT"
DEFAULT_QUESTION_FEATURES = DEFAULT_DATA_DIR / "statics2011" / "question_harrier_features.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_mobilekt_v4(checkpoint_path: Path, device: torch.device) -> tuple[MobileKTV4, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["student_state_dict"] if "student_state_dict" in ckpt else ckpt
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    cfg = MobileKTConfig(
        dataset=args.get("dataset", "statics2011"),
        data_dir=args.get("data_dir", "data/datasets/KT"),
        max_seq_len=int(args.get("max_seq_len", 200)),
        d=int(args.get("d", 64)),
        qde_hidden=int(args.get("qde_hidden", 128)),
        qe_input_mode="features",
        question_feature_dim=int(meta.get("question_feature_dim", 1024)),
        question_features_path=args.get(
            "question_features_path",
            "data/datasets/KT/statics2011/question_harrier_features.pt",
        ),
        use_diff_bias=True,
        mikt_state_dim=int(args.get("mikt_state_dim", 64)),
        mikt_output_scale=float(args.get("mikt_output_scale", 5.0)),
        dropout=float(args.get("dropout", 0.1)),
    )
    cfg.n_questions = int(meta.get("n_questions", 1223))
    cfg.n_concepts = int(meta.get("n_concepts", 640))
    cfg.model = "mobilekt4"  # type: ignore[attr-defined]
    model = MobileKTV4(cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, {"args": args, "meta": meta}


def make_probe(probe_type: str, cfg: TimeAwareProbeConfig) -> nn.Module:
    if probe_type == "mlp":
        return TimeAwareProbe(cfg)
    if probe_type == "ebbinghaus":
        return EbbinghausTimeAwareProbe(cfg)
    raise ValueError("probe_type must be one of: mlp, ebbinghaus")


def expected_calibration_error(pred: torch.Tensor, label: torch.Tensor, n_bins: int = 10) -> float:
    bins = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = pred.new_tensor(0.0)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (pred >= lo) & (pred <= hi) if i == n_bins - 1 else (pred >= lo) & (pred < hi)
        if mask.any():
            ece = ece + mask.float().mean() * (pred[mask].mean() - label[mask].mean()).abs()
    return float(ece.item())


def binary_auc(pred: torch.Tensor, label: torch.Tensor) -> float:
    """Compute pilot AUC after thresholding soft future labels at 0.5."""
    hard = (label >= 0.5).long()
    pos = int(hard.sum().item())
    neg = int(hard.numel() - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    order = torch.argsort(pred)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, pred.numel() + 1, dtype=torch.float)
    pos_rank_sum = ranks[hard.bool()].sum()
    return float(((pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)).item())


def _sample_label_indices(
    label_mask: torch.Tensor,
    max_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    index = torch.nonzero(label_mask, as_tuple=False)
    if max_samples > 0 and index.shape[0] > max_samples:
        perm = torch.randperm(index.shape[0], generator=generator)
        index = index[perm[:max_samples]]
    return index


@torch.no_grad()
def build_probe_batch(
    model: MobileKTV4,
    batch: dict[str, torch.Tensor],
    cfg: TimeAwareProbeConfig,
    max_samples: int,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    """Build sparse TAP samples without materializing every concept state."""
    q_ids_cpu = batch["question_ids"]
    c_ids_cpu = batch["concept_ids"]
    responses_cpu = batch["responses"]
    lengths_cpu = batch["length"]

    timer = build_timer_features(
        c_ids_cpu,
        responses_cpu,
        lengths=lengths_cpu,
        n_concepts=cfg.n_concepts,
    )
    labels, label_mask, support = build_future_correctness_labels(
        c_ids_cpu,
        responses_cpu,
        lengths=lengths_cpu,
        n_concepts=cfg.n_concepts,
        horizon=cfg.horizon,
        tau=cfg.tau,
        mode=cfg.label_mode,
    )
    index = _sample_label_indices(label_mask, max_samples, generator)
    if index.numel() == 0:
        return None

    q_ids = q_ids_cpu.to(device)
    c_ids = c_ids_cpu.to(device)
    responses = responses_cpu.to(device)
    question_features = batch["question_features"].to(device)
    encoded = model.encode_questions(question_features=question_features)
    batch_size, seq_len = q_ids.shape
    item_embedding = model.backbone.build_item_embedding(
        encoded.embedding.reshape(batch_size * seq_len, -1),
        encoded.difficulty.reshape(batch_size * seq_len),
        c_ids.reshape(batch_size * seq_len, c_ids.shape[-1]),
    ).view(batch_size, seq_len, -1)
    skill_state, all_state, last_skill_time = model.backbone.initial_state(batch_size, device)

    index_device = index.to(device)
    hidden = torch.empty(index.shape[0], 2 * model.backbone.state_d, device=device)
    for step in range(seq_len):
        valid = q_ids[:, step] != 0
        skill_state, all_state, last_skill_time = model.backbone._update_observed(
            skill_state,
            all_state,
            last_skill_time,
            item_embedding[:, step],
            c_ids[:, step],
            responses[:, step],
            step=step,
            valid=valid,
        )
        selected = index_device[:, 1] == step
        if selected.any():
            batch_index = index_device[selected, 0]
            concept_index = index_device[selected, 2]
            hidden[selected] = torch.cat(
                [all_state[batch_index], skill_state[batch_index, concept_index]],
                dim=-1,
            )

    b_idx, t_idx, c_idx = index[:, 0], index[:, 1], index[:, 2]
    return {
        "hidden_state": hidden,
        "concept_ids": c_idx.to(device).long(),
        "timer_features": timer[b_idx, t_idx, c_idx].to(device),
        "labels": labels[b_idx, t_idx, c_idx].to(device),
        "support": support[b_idx, t_idx, c_idx].to(device),
    }


def forgetting_loss(
    probe: nn.Module,
    samples: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.lambda_forgetting <= 0:
        return samples["labels"].new_tensor(0.0)
    timer_short = samples["timer_features"].clone()
    timer_long = samples["timer_features"].clone()
    timer_short[:, 0] = torch.log1p(torch.full_like(timer_short[:, 0], args.forgetting_short_gap))
    timer_long[:, 0] = torch.log1p(torch.full_like(timer_long[:, 0], args.forgetting_long_gap))
    return forgetting_monotonic_loss(
        probe(samples["hidden_state"], samples["concept_ids"], timer_short),
        probe(samples["hidden_state"], samples["concept_ids"], timer_long),
        margin=args.forgetting_margin,
    )


def train_one_epoch(
    probe: nn.Module,
    model: MobileKTV4,
    loader: DataLoader,
    optimizer: Adam,
    cfg: TimeAwareProbeConfig,
    args: argparse.Namespace,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, float]:
    probe.train()
    total_loss, total_samples = 0.0, 0
    for batch_index, batch in enumerate(loader):
        if args.max_batches and batch_index >= args.max_batches:
            break
        samples = build_probe_batch(model, batch, cfg, args.max_samples_per_batch, generator, device)
        if samples is None:
            continue
        pred = probe(samples["hidden_state"], samples["concept_ids"], samples["timer_features"])
        future = soft_binary_cross_entropy(pred, samples["labels"])
        loss = future + args.lambda_forgetting * forgetting_loss(probe, samples, args)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), args.grad_clip)
        optimizer.step()
        n = samples["labels"].numel()
        total_loss += float(loss.item()) * n
        total_samples += n
    return {"loss": total_loss / max(total_samples, 1), "samples": float(total_samples)}


@torch.no_grad()
def evaluate(
    probe: nn.Module,
    model: MobileKTV4,
    loader: DataLoader,
    cfg: TimeAwareProbeConfig,
    args: argparse.Namespace,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, float]:
    probe.eval()
    all_pred, all_label = [], []
    total_loss, total_samples = 0.0, 0
    for batch_index, batch in enumerate(loader):
        if args.eval_max_batches and batch_index >= args.eval_max_batches:
            break
        samples = build_probe_batch(model, batch, cfg, args.max_samples_per_batch, generator, device)
        if samples is None:
            continue
        pred = probe(samples["hidden_state"], samples["concept_ids"], samples["timer_features"])
        loss = soft_binary_cross_entropy(pred, samples["labels"])
        n = samples["labels"].numel()
        total_loss += float(loss.item()) * n
        total_samples += n
        all_pred.append(pred.cpu())
        all_label.append(samples["labels"].cpu())
    if not all_pred:
        return {"loss": math.nan, "samples": 0.0, "auc": math.nan, "brier": math.nan, "ece": math.nan}
    pred = torch.cat(all_pred)
    label = torch.cat(all_label)
    return {
        "loss": total_loss / max(total_samples, 1),
        "samples": float(total_samples),
        "auc": binary_auc(pred, label),
        "brier": float(torch.mean((pred - label) ** 2).item()),
        "ece": expected_calibration_error(pred, label),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset", default="statics2011")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--question_features_path", default=str(DEFAULT_QUESTION_FEATURES))
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--probe_type", choices=["mlp", "ebbinghaus"], default="mlp")
    parser.add_argument("--probe_hidden_dim", type=int, default=128)
    parser.add_argument("--concept_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--tau", type=float, default=20.0)
    parser.add_argument("--label_mode", choices=["next", "average", "decayed_average"], default="decayed_average")
    parser.add_argument("--lambda_forgetting", type=float, default=0.0)
    parser.add_argument("--forgetting_margin", type=float, default=0.0)
    parser.add_argument("--forgetting_short_gap", type=float, default=3.0)
    parser.add_argument("--forgetting_long_gap", type=float, default=50.0)
    parser.add_argument("--max_decay_rate", type=float, default=2.0)
    parser.add_argument("--max_samples_per_batch", type=int, default=8192)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--eval_max_batches", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint_meta = load_mobilekt_v4(args.checkpoint, device)
    train_ds, valid_ds, test_ds, dataset_meta = load_dataset(
        args.dataset,
        args.data_dir,
        args.max_seq_len,
        question_features_path=args.question_features_path,
    )
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    valid_loader = DataLoader(valid_ds, args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
    cfg = TimeAwareProbeConfig(
        state_dim=2 * model.backbone.state_d,
        n_concepts=model.backbone.n_concepts,
        timer_dim=3,
        concept_dim=args.concept_dim,
        hidden_dim=args.probe_hidden_dim,
        dropout=args.dropout,
        horizon=args.horizon,
        tau=args.tau,
        label_mode=args.label_mode,
        lambda_forgetting=args.lambda_forgetting,
        forgetting_margin=args.forgetting_margin,
        forgetting_short_gap=args.forgetting_short_gap,
        forgetting_long_gap=args.forgetting_long_gap,
        max_decay_rate=args.max_decay_rate,
    )
    probe = make_probe(args.probe_type, cfg).to(device)
    optimizer = Adam(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_generator = torch.Generator().manual_seed(args.seed)
    eval_generator = torch.Generator().manual_seed(args.seed + 1)

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")
    print(f"Dataset    : train={len(train_ds)} valid={len(valid_ds)} test={len(test_ds)}")
    print(f"Probe      : {args.probe_type} {asdict(cfg)}")
    start = time.time()
    best_valid = math.inf
    best_state: dict[str, Any] | None = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            probe, model, train_loader, optimizer, cfg, args, device, train_generator
        )
        valid_metrics = evaluate(probe, model, valid_loader, cfg, args, device, eval_generator)
        history.append({"epoch": epoch, "train": train_metrics, "valid": valid_metrics})
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_auc={valid_metrics['auc']:.4f} "
            f"valid_brier={valid_metrics['brier']:.4f} "
            f"valid_ece={valid_metrics['ece']:.4f}"
        )
        if valid_metrics["loss"] < best_valid:
            best_valid = valid_metrics["loss"]
            best_state = {
                "probe_state_dict": probe.state_dict(),
                "probe_config": asdict(cfg),
                "probe_type": args.probe_type,
                "base_model_name": "mobilekt4",
                "base_checkpoint": str(args.checkpoint),
                "base_checkpoint_sha256": sha256_file(args.checkpoint),
                "dataset_name": args.dataset,
                "dataset_meta": dataset_meta,
                "valid_metrics": valid_metrics,
                "training_args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
            }

    if best_state is None:
        raise RuntimeError("No TAP checkpoint was produced")
    checkpoint_path = args.out_dir / "mobilekt_v4_tap_best.pt"
    torch.save(best_state, checkpoint_path)
    probe.load_state_dict(best_state["probe_state_dict"])
    test_metrics = evaluate(probe, model, test_loader, cfg, args, device, eval_generator)
    report = {
        "schema_version": "1.0",
        "model": "MobileKT v4 TAP",
        "base_checkpoint": str(args.checkpoint),
        "base_checkpoint_sha256": best_state["base_checkpoint_sha256"],
        "tap_checkpoint": str(checkpoint_path),
        "probe_type": args.probe_type,
        "probe_config": asdict(cfg),
        "best_valid": best_state["valid_metrics"],
        "test": test_metrics,
        "history": history,
        "elapsed_seconds": time.time() - start,
        "notes": [
            "The MobileKT v4 backbone is frozen. Only the TAP readout is trained.",
            "TAP targets are future same-concept correctness proxies, not ground-truth psychological mastery labels.",
            "The export-compatible timer uses interaction-step gaps, not wall-clock elapsed time.",
        ],
    }
    write_json(args.out_dir / "tap_training_report.json", report)
    print(json.dumps({"checkpoint": str(checkpoint_path), "test": test_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
