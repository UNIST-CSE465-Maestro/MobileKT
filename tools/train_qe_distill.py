#!/usr/bin/env python3
"""Train a MobileKT v4 Question Encoder against a MIKT-ID teacher.

This implements the MIKT-first setting from the architecture note:

    pretrained MIKT-ID teacher -> freeze MIKT backbone -> train QE(feature)
    to reconstruct teacher (q_embedding, difficulty).

The default is the clean frozen ``QE-Distill-q+diff`` experiment. Passing
``--backbone_mode trainable`` turns it into the teacher-guided joint setting:

    L = L_KT + lambda_q L_q + lambda_diff L_diff + lambda_logit L_logit
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import MobileKTConfig
from datasets import collate_fn, load_dataset
from models import MobileKTV4
from models.irt.prediction import IRTPrediction
from utils import compute_acc, compute_auc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_cfg(args: argparse.Namespace, *, qe_input_mode: str, feature_dim: int | None) -> MobileKTConfig:
    cfg = MobileKTConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        max_seq_len=args.max_seq_len,
        d=args.d,
        qde_hidden=args.qde_hidden,
        qe_input_mode=qe_input_mode,
        question_feature_dim=feature_dim,
        question_features_path=args.question_features_path,
        use_diff_bias=True,
        mikt_state_dim=args.mikt_state_dim,
        mikt_output_scale=args.mikt_output_scale,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        dropout=args.dropout,
        patience=args.patience,
        grad_clip=args.grad_clip,
        seed=args.seed,
        device=args.device,
        session=args.session,
    )
    cfg.model = "mobilekt4"  # type: ignore[attr-defined]
    return cfg


def load_teacher(args: argparse.Namespace, meta: dict, device: torch.device) -> MobileKTV4:
    cfg = make_cfg(args, qe_input_mode="id", feature_dim=None)
    cfg.n_questions = int(meta["n_questions"])
    cfg.n_concepts = int(meta["n_concepts"])
    teacher = MobileKTV4(cfg).to(device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def load_student(args: argparse.Namespace, meta: dict, teacher: MobileKTV4, device: torch.device) -> MobileKTV4:
    cfg = make_cfg(args, qe_input_mode="features", feature_dim=int(meta["question_feature_dim"]))
    cfg.n_questions = int(meta["n_questions"])
    cfg.n_concepts = int(meta["n_concepts"])
    student = MobileKTV4(cfg).to(device)
    student.backbone.load_state_dict(teacher.backbone.state_dict())
    if args.backbone_mode == "frozen":
        for param in student.backbone.parameters():
            param.requires_grad_(False)
    return student


def masked_values(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return tensor[mask]


def logit_probs(probs: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(probs.dtype).eps
    return torch.logit(probs.clamp(eps, 1.0 - eps))


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return float("nan")
    x = x.float()
    y = y.float()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = vx.norm() * vy.norm()
    if denom.item() == 0:
        return float("nan")
    return float((vx @ vy / denom).item())


def distill_losses(
    student: MobileKTV4,
    teacher: MobileKTV4,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    q_ids = batch["question_ids"].to(device)
    c_ids = batch["concept_ids"].to(device)
    resp = batch["responses"].to(device)
    q_feat = batch["question_features"].to(device)
    item_mask = q_ids > 0

    with torch.no_grad():
        teacher_encoded = teacher.encode_questions(question_ids=q_ids)

    student_encoded = student.encode_questions(question_features=q_feat)
    q_student = masked_values(student_encoded.embedding, item_mask)
    q_teacher = masked_values(teacher_encoded.embedding, item_mask)
    d_student = masked_values(student_encoded.difficulty, item_mask)
    d_teacher = masked_values(teacher_encoded.difficulty, item_mask)

    q_loss = F.mse_loss(q_student, q_teacher)
    diff_loss = F.mse_loss(d_student, d_teacher)
    loss = args.q_loss_weight * q_loss + args.diff_loss_weight * diff_loss

    stats = {
        "q_loss": float(q_loss.detach().item()),
        "diff_loss": float(diff_loss.detach().item()),
    }

    if args.logit_loss_weight > 0 or args.kt_loss_weight > 0:
        y_student, pred_mask = student(q_ids, c_ids, resp, question_features=q_feat)
        if args.logit_loss_weight > 0:
            with torch.no_grad():
                y_teacher, _ = teacher(q_ids, c_ids, resp)
            logit_loss = F.mse_loss(
                masked_values(logit_probs(y_student), pred_mask),
                masked_values(logit_probs(y_teacher), pred_mask),
            )
            loss = loss + args.logit_loss_weight * logit_loss
            stats["logit_loss"] = float(logit_loss.detach().item())
        if args.kt_loss_weight > 0:
            kt_loss = IRTPrediction.loss(y_student, resp[:, 1:], pred_mask)
            loss = loss + args.kt_loss_weight * kt_loss
            stats["kt_loss"] = float(kt_loss.detach().item())

    stats["loss"] = float(loss.detach().item())
    return loss, stats


def train_one_epoch(
    student: MobileKTV4,
    teacher: MobileKTV4,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    student.train()
    if args.backbone_mode == "frozen":
        student.backbone.eval()
    totals: dict[str, float] = {}
    n_batch = 0

    for batch in loader:
        optimizer.zero_grad()
        loss, stats = distill_losses(student, teacher, batch, device, args)
        loss.backward()
        trainable_params = [param for param in student.parameters() if param.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()

        for key, value in stats.items():
            totals[key] = totals.get(key, 0.0) + value
        n_batch += 1

    return {key: value / max(n_batch, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(student: MobileKTV4, teacher: MobileKTV4, loader: DataLoader, device: torch.device) -> dict[str, float]:
    student.eval()
    teacher.eval()
    pred_all, true_all, mask_all = [], [], []
    q_mse_sum, diff_mse_sum, q_cos_sum, n_items = 0.0, 0.0, 0.0, 0
    logit_mse_sum, n_preds = 0.0, 0
    diff_student_all, diff_teacher_all = [], []

    for batch in loader:
        q_ids = batch["question_ids"].to(device)
        c_ids = batch["concept_ids"].to(device)
        resp = batch["responses"].to(device)
        q_feat = batch["question_features"].to(device)
        item_mask = q_ids > 0

        teacher_encoded = teacher.encode_questions(question_ids=q_ids)
        student_encoded = student.encode_questions(question_features=q_feat)

        q_student = masked_values(student_encoded.embedding, item_mask)
        q_teacher = masked_values(teacher_encoded.embedding, item_mask)
        d_student = masked_values(student_encoded.difficulty, item_mask)
        d_teacher = masked_values(teacher_encoded.difficulty, item_mask)

        batch_items = int(item_mask.sum().item())
        if batch_items:
            q_mse_sum += F.mse_loss(q_student, q_teacher, reduction="sum").item() / q_student.shape[-1]
            diff_mse_sum += F.mse_loss(d_student, d_teacher, reduction="sum").item()
            q_cos_sum += F.cosine_similarity(q_student, q_teacher, dim=-1).sum().item()
            n_items += batch_items
            diff_student_all.append(d_student.cpu())
            diff_teacher_all.append(d_teacher.cpu())

        y_pred, pred_mask = student(q_ids, c_ids, resp, question_features=q_feat)
        y_teacher, _ = teacher(q_ids, c_ids, resp)
        batch_preds = int(pred_mask.sum().item())
        if batch_preds:
            logit_mse_sum += F.mse_loss(
                masked_values(logit_probs(y_pred), pred_mask),
                masked_values(logit_probs(y_teacher), pred_mask),
                reduction="sum",
            ).item()
            n_preds += batch_preds
        pred_all.append(y_pred.cpu())
        true_all.append(resp[:, 1:].cpu())
        mask_all.append(pred_mask.cpu())

    y_pred = torch.cat(pred_all)
    y_true = torch.cat(true_all)
    pred_mask = torch.cat(mask_all)
    diff_student = torch.cat(diff_student_all) if diff_student_all else torch.empty(0)
    diff_teacher = torch.cat(diff_teacher_all) if diff_teacher_all else torch.empty(0)

    return {
        "auc": compute_auc(y_pred, y_true, pred_mask),
        "acc": compute_acc(y_pred, y_true, pred_mask),
        "kt_loss": IRTPrediction.loss(y_pred, y_true, pred_mask).item(),
        "q_mse": q_mse_sum / max(n_items, 1),
        "diff_mse": diff_mse_sum / max(n_items, 1),
        "q_cos": q_cos_sum / max(n_items, 1),
        "diff_pearson": pearson(diff_student, diff_teacher),
        "logit_mse": logit_mse_sum / max(n_preds, 1),
    }


def run_dir(args: argparse.Namespace) -> Path:
    session = args.session or datetime.now().strftime("statics2011_qe_distill_%Y%m%d_%H%M%S")
    teacher_tag = Path(args.teacher_ckpt).parent.parent.name
    loss_tag = f"q{args.q_loss_weight:g}_d{args.diff_loss_weight:g}_logit{args.logit_loss_weight:g}_kt{args.kt_loss_weight:g}"
    name = (
        f"{args.dataset}_qe_{args.backbone_mode}_{teacher_tag}"
        f"_seed{args.seed}_lr{args.lr:.0e}_dp{args.dropout}_{loss_tag}"
    )
    return ROOT / "experiments" / session / name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--dataset", default="statics2011")
    parser.add_argument("--data_dir", default="data/datasets/KT")
    parser.add_argument("--question_features_path", default="data/datasets/KT/statics2011/question_harrier_features.pt")
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--qde_hidden", type=int, default=128)
    parser.add_argument("--mikt_state_dim", type=int, default=64)
    parser.add_argument("--mikt_output_scale", type=float, default=5.0)
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--q_loss_weight", type=float, default=1.0)
    parser.add_argument("--diff_loss_weight", type=float, default=1.0)
    parser.add_argument("--logit_loss_weight", type=float, default=0.0)
    parser.add_argument("--kt_loss_weight", type=float, default=0.0)
    parser.add_argument(
        "--backbone_mode",
        choices=["frozen", "trainable"],
        default="frozen",
        help="frozen=QE-only distillation, trainable=joint QE+MIKT fine-tuning",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--session", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out_dir = run_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    train_ds, val_ds, test_ds, meta = load_dataset(
        args.dataset,
        args.data_dir,
        args.max_seq_len,
        question_features_path=args.question_features_path,
    )
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    teacher = load_teacher(args, meta, device)
    student = load_student(args, meta, teacher, device)
    trainable_params = [param for param in student.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=1e-5)
    ckpt_path = out_dir / "qe_distill_best.pt"

    print(f"Run dir       : {out_dir}")
    print(f"Teacher ckpt  : {args.teacher_ckpt}")
    print(f"Device        : {device}")
    print(f"Dataset       : {args.dataset} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"Backbone mode : {args.backbone_mode}")
    print(f"Loss weights  : q={args.q_loss_weight} diff={args.diff_loss_weight} logit={args.logit_loss_weight} kt={args.kt_loss_weight}")
    print(f"Trainable     : {sum(p.numel() for p in student.parameters() if p.requires_grad):,}")

    best_val = math.inf
    patience_count = 0
    history = []
    for epoch in range(1, args.n_epochs + 1):
        train_stats = train_one_epoch(student, teacher, train_loader, optimizer, device, args)
        val_stats = evaluate(student, teacher, val_loader, device)
        scheduler.step()
        score = (
            args.q_loss_weight * val_stats["q_mse"]
            + args.diff_loss_weight * val_stats["diff_mse"]
            + args.logit_loss_weight * val_stats["logit_mse"]
            + args.kt_loss_weight * val_stats["kt_loss"]
        )
        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})

        print(
            f"Epoch {epoch:3d} | "
            f"loss={train_stats['loss']:.5f} q={train_stats['q_loss']:.5f} diff={train_stats['diff_loss']:.5f} | "
            f"val_auc={val_stats['auc']:.4f} val_acc={val_stats['acc']:.4f} "
            f"q_cos={val_stats['q_cos']:.4f} diff_r={val_stats['diff_pearson']:.4f} "
            f"logit_mse={val_stats['logit_mse']:.4f}"
        )

        if score < best_val:
            best_val = score
            patience_count = 0
            torch.save(
                {
                    "student_state_dict": student.state_dict(),
                    "teacher_ckpt": args.teacher_ckpt,
                    "args": vars(args),
                    "meta": meta,
                    "val": val_stats,
                },
                ckpt_path,
            )
            print(f"  -> Best QE saved (val_score={best_val:.6f})")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    saved = torch.load(ckpt_path, map_location=device)
    student.load_state_dict(saved["student_state_dict"])
    test_stats = evaluate(student, teacher, test_loader, device)
    elapsed = time.time() - start

    result = {
        "run_dir": str(out_dir),
        "teacher_ckpt": args.teacher_ckpt,
        "best_val": saved["val"],
        "test": test_stats,
        "elapsed_sec": elapsed,
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== Test Results ===")
    print(f"  AUC          : {test_stats['auc']:.4f}")
    print(f"  ACC          : {test_stats['acc']:.4f}")
    print(f"  KT loss      : {test_stats['kt_loss']:.4f}")
    print(f"  q MSE        : {test_stats['q_mse']:.6f}")
    print(f"  q cosine     : {test_stats['q_cos']:.4f}")
    print(f"  diff MSE     : {test_stats['diff_mse']:.6f}")
    print(f"  diff Pearson : {test_stats['diff_pearson']:.4f}")
    print(f"  logit MSE    : {test_stats['logit_mse']:.6f}")
    print(f"  Saved        : {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
