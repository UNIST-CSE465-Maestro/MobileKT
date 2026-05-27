"""
MobileKT Training Script

Usage:
    python3 train.py --dataset assist09 --d 128 --n_epochs 100
"""

import argparse
import os
import sys
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import MobileKTConfig
from models import MobileKT, MobileKTV2, MobileKTV3, MobileKTV3b, MobileKTV4
from models.irt.prediction import IRTPrediction
from datasets import load_dataset, collate_fn
from utils import compute_auc, compute_acc

MODEL_REGISTRY = {
    "mobilekt":   MobileKT,
    "mobilekt2":  MobileKTV2,
    "mobilekt3":  MobileKTV3,
    "mobilekt3b": MobileKTV3b,
    "mobilekt4":  MobileKTV4,
}


# ── Logging helpers ───────────────────────────────────────────────────────────

class _Tee:
    """Write to multiple streams simultaneously (terminal + log file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def fileno(self):
        # Fall back to the first real stream for fileno (e.g. subprocess needs it)
        return self.streams[0].fileno()


STATUS_LOG = os.path.join("experiments", "training_status.log")


def _log_status(msg: str):
    """Append a timestamped line to the shared training_status.log."""
    os.makedirs("experiments", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with open(STATUS_LOG, "a") as f:
        f.write(line)


# ── Pretrain transfer ─────────────────────────────────────────────────────────

def load_pretrained_core(model: "MobileKT", ckpt_path: str, device: torch.device):
    """
    Load weights from a checkpoint trained on a different dataset.

    Strategy
    --------
    All fixed-size MLP modules (QDE, ERM, KGF, DRE, SAE, CU, DU) and
    dataset-agnostic parameters (answer_embed, domain_embed,
    init_domain_mastery) are transferred when their shapes match exactly.

    Dataset-specific parameters (question_embed, concept_embed, init_mastery)
    are skipped if the shapes differ — they will remain randomly initialised
    and learned from scratch on the target dataset.
    """
    src_sd = torch.load(ckpt_path, map_location=device)
    dst_sd = model.state_dict()

    loaded, skipped_shape, skipped_missing = [], [], []
    for k, src_v in src_sd.items():
        if k not in dst_sd:
            skipped_missing.append(k)
        elif dst_sd[k].shape != src_v.shape:
            skipped_shape.append(f"{k}  {tuple(src_v.shape)}→{tuple(dst_sd[k].shape)}")
        else:
            dst_sd[k] = src_v
            loaded.append(k)

    model.load_state_dict(dst_sd)
    return loaded, skipped_shape, skipped_missing


# ── Training helpers ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, device, grad_clip):
    model.train()
    total_loss, n_batch = 0.0, 0
    for batch in loader:
        q_ids = batch["question_ids"].to(device)
        c_ids = batch["concept_ids"].to(device)
        resp  = batch["responses"].to(device)
        q_feat = batch.get("question_features")
        if q_feat is not None:
            q_feat = q_feat.to(device)

        optimizer.zero_grad()
        if q_feat is None:
            y_pred, mask = model(q_ids, c_ids, resp)
        else:
            y_pred, mask = model(q_ids, c_ids, resp, question_features=q_feat)
        y_true = resp[:, 1:]

        loss = IRTPrediction.loss(y_pred, y_true, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batch += 1

    return total_loss / max(n_batch, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred, all_true, all_mask = [], [], []
    for batch in loader:
        q_ids = batch["question_ids"].to(device)
        c_ids = batch["concept_ids"].to(device)
        resp  = batch["responses"].to(device)
        q_feat = batch.get("question_features")
        if q_feat is not None:
            q_feat = q_feat.to(device)

        if q_feat is None:
            y_pred, mask = model(q_ids, c_ids, resp)
        else:
            y_pred, mask = model(q_ids, c_ids, resp, question_features=q_feat)
        y_true = resp[:, 1:]

        all_pred.append(y_pred.cpu())
        all_true.append(y_true.cpu())
        all_mask.append(mask.cpu())

    y_pred = torch.cat(all_pred)
    y_true = torch.cat(all_true)
    mask   = torch.cat(all_mask)

    auc  = compute_auc(y_pred, y_true, mask)
    acc  = compute_acc(y_pred, y_true, mask)
    loss = IRTPrediction.loss(y_pred, y_true, mask).item()
    return {"auc": auc, "acc": acc, "loss": loss}


def print_config(cfg: MobileKTConfig, model: "MobileKT"):
    n_total   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_no_qemb = n_total - sum(
        p.numel() for name, p in model.named_parameters()
        if "question_embed" in name
    )
    print(f"\n{'='*52}")
    print(f"  MobileKT — {cfg.dataset}")
    print(f"{'='*52}")
    print(f"  Dataset  : {cfg.dataset}  (Q={cfg.n_questions}  C={cfg.n_concepts}  T={cfg.n_domains})")
    print(f"  d={cfg.d}  dropout={cfg.dropout}  irt_scale={cfg.irt_scale}")
    print(f"  qde_hidden={cfg.qde_hidden}  erm_hidden={cfg.erm_hidden}")
    print(f"  sae_hidden={cfg.sae_hidden}")
    print(f"  cu_hidden={cfg.cu_hidden}  du_hidden={cfg.du_hidden}")
    print(f"  lr={cfg.lr}  weight_decay={cfg.weight_decay}  batch={cfg.batch_size}")
    print(f"  n_epochs={cfg.n_epochs}  patience={cfg.patience}  scheduler=CosineAnnealing")
    print(f"{'─'*52}")
    print(f"  Parameters (total)         : {n_total:>10,}")
    print(f"  Parameters (ex q_embed)    : {n_no_qemb:>10,}")
    print(f"{'='*52}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(cfg: MobileKTConfig):
    set_seed(cfg.seed)
    t_start = time.time()

    # ── Run directory & logging setup ──────────────────────────────────────
    model_name = getattr(cfg, 'model', 'mobilekt')
    pretrain_tag = ""
    if cfg.pretrain_ckpt:
        pt_name = os.path.splitext(os.path.basename(cfg.pretrain_ckpt))[0]
        # Extract source dataset from checkpoint name (e.g. mobilekt_ktbd_... → ktbd)
        parts = pt_name.split("_")
        pt_src = parts[1] if len(parts) > 1 else "pt"
        pretrain_tag = f"_from{pt_src}"

    run_tag = (
        f"{cfg.dataset}"
        f"_{model_name}"
        f"_d{cfg.d}"
        f"_lr{cfg.lr:.0e}"
        f"_wd{cfg.weight_decay:.0e}"
        f"_nd{cfg.n_domains}"
        f"_dp{cfg.dropout}"
        f"{pretrain_tag}"
    )
    # session: shared timestamp folder for all runs launched together.
    # Pass --session YYYYMMDD_HHMMSS to group concurrent experiments.
    session = cfg.session or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("experiments", session, run_tag)
    os.makedirs(run_dir, exist_ok=True)

    # Redirect stdout/stderr → terminal + per-run log file simultaneously
    log_file   = open(os.path.join(run_dir, "train.log"), "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    _log_status(f"START  | run={run_tag} | pid={os.getpid()} | dir={run_dir}")

    # ── Data ──────────────────────────────────────────────────────────────
    question_features_path = cfg.question_features_path
    if (
        getattr(cfg, "model", "") == "mobilekt4"
        and cfg.qe_input_mode == "features"
        and not question_features_path
    ):
        candidate = os.path.join(cfg.data_dir, cfg.dataset, "question_text_features.pt")
        if os.path.exists(candidate):
            question_features_path = candidate

    train_ds, val_ds, test_ds, meta = load_dataset(
        cfg.dataset,
        cfg.data_dir,
        cfg.max_seq_len,
        question_features_path=question_features_path,
    )
    cfg.n_questions = meta.get("n_questions", cfg.n_questions)
    cfg.n_concepts  = meta.get("n_concepts",  cfg.n_concepts)
    cfg.question_feature_dim = meta.get("question_feature_dim", cfg.question_feature_dim)

    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True,  collate_fn=collate_fn, num_workers=4)
    val_loader   = DataLoader(val_ds,   cfg.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader  = DataLoader(test_ds,  cfg.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # ── Model ─────────────────────────────────────────────────────────────
    device     = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model_cls  = MODEL_REGISTRY.get(getattr(cfg, 'model', 'mobilekt'), MobileKT)
    model      = model_cls(cfg).to(device)
    print_config(cfg, model)
    print(f"  Device : {device}")
    print(f"  Run dir: {run_dir}")
    print(f"  Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")

    # ── Pretrained core weight transfer ───────────────────────────────────
    if cfg.pretrain_ckpt:
        loaded, skipped_shape, _ = load_pretrained_core(
            model, cfg.pretrain_ckpt, device
        )
        print(f"\n  [Pretrain] loaded {len(loaded)} weights from: {cfg.pretrain_ckpt}")
        if skipped_shape:
            print(f"  [Pretrain] shape-mismatch skipped ({len(skipped_shape)}):")
            for s in skipped_shape:
                print(f"    ✗ {s}")
        print()
    else:
        print()

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs, eta_min=1e-5)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_auc  = 0.0
    patience_count = 0
    ckpt_path = os.path.join(run_dir, f"mobilekt_{run_tag}_best.pt")

    for epoch in range(1, cfg.n_epochs + 1):
        train_loss  = train_one_epoch(model, train_loader, optimizer, device, cfg.grad_clip)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:3d} | loss={train_loss:.4f} | "
            f"val_auc={val_metrics['auc']:.4f}  val_acc={val_metrics['acc']:.4f} | "
            f"lr={lr_now:.2e}"
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc   = val_metrics["auc"]
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Best model saved  (val_auc={best_val_auc:.4f})")
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # ── Test ──────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    duration = time.time() - t_start
    h, m, s  = int(duration // 3600), int((duration % 3600) // 60), int(duration % 60)

    print(f"\n=== Test Results ({cfg.dataset}) ===")
    print(f"  AUC : {test_metrics['auc']:.4f}")
    print(f"  ACC : {test_metrics['acc']:.4f}")
    print(f"  Time: {h:02d}h {m:02d}m {s:02d}s")

    _log_status(
        f"FINISH | run={run_tag} | "
        f"test_auc={test_metrics['auc']:.4f}  test_acc={test_metrics['acc']:.4f} | "
        f"duration={h:02d}h{m:02d}m{s:02d}s | dir={run_dir}"
    )

    # Restore stdout/stderr and close log file
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    type=str,   default="assist09")
    parser.add_argument("--data_dir",   type=str,   default="data")
    parser.add_argument("--d",          type=int,   default=128)
    parser.add_argument("--n_domains",  type=int,   default=5)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--n_epochs",   type=int,   default=100)
    parser.add_argument("--dropout",    type=float, default=0.2)
    parser.add_argument("--irt_scale",  type=float, default=3.0)
    parser.add_argument("--patience",   type=int,   default=15)
    parser.add_argument("--device",     type=str,   default="cuda")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--weight_decay",  type=float, default=1e-5)
    parser.add_argument("--use_diff_bias", type=int,   default=1,
                        help="v3 only: 1=use per-question bias (default), 0=disable")
    parser.add_argument("--question_feature_dim", type=int, default=None,
                        help="v4 only: cached text/question feature dimension")
    parser.add_argument("--question_features_path", type=str, default="",
                        help="v4 only: cached raw-question feature matrix path")
    parser.add_argument("--qe_input_mode", type=str, default="features",
                        choices=["features", "id"],
                        help="v4 only: features=document path, id=MIKT-ID baseline")
    parser.add_argument("--mikt_state_dim", type=int, default=64,
                        help="v4 only: MIKT per-concept state dimension")
    parser.add_argument("--mikt_output_scale", type=float, default=5.0,
                        help="v4 only: Rasch-style ability-difficulty scale")
    parser.add_argument("--session",       type=str,   default="")
    parser.add_argument("--pretrain_ckpt", type=str,   default="")
    parser.add_argument("--model",         type=str,   default="mobilekt",
                        choices=list(MODEL_REGISTRY.keys()))
    args = parser.parse_args()

    cfg = MobileKTConfig(**{k: v for k, v in vars(args).items()
                            if k in MobileKTConfig.__dataclass_fields__})
    # Attach model name outside dataclass (avoids breaking existing configs)
    cfg.model = args.model  # type: ignore[attr-defined]
    main(cfg)
