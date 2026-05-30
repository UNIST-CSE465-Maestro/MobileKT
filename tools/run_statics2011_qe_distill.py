#!/usr/bin/env python3
"""Launch Statics2011 MobileKT v4 QE distillation/joint runs.

The runner reuses ID-teacher checkpoints from
``tools/run_statics2011_v4_compare.py`` and trains Harrier-feature QE heads on
MIKT backbones.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT.parents[1] / "data" / "datasets" / "KT"
DEFAULT_FEATURES = DEFAULT_DATA_DIR / "statics2011" / "question_harrier_features.pt"
DEFAULT_TEACHER_SESSION = "statics2011_v4_compare_20260526_dropout"


@dataclass(frozen=True)
class RunSpec:
    seed: int
    dropout: float
    lr: float = 1e-3
    d: int = 64
    state_dim: int = 64

    @property
    def name(self) -> str:
        dp_tag = str(self.dropout).replace(".", "p")
        return f"qe_seed{self.seed}_lr{self.lr:.0e}_dp{dp_tag}"


def build_specs(preset: str) -> list[RunSpec]:
    seeds = [42, 2024, 3407]
    if preset == "core":
        dropouts = [0.2]
    elif preset == "best":
        dropouts = [0.1]
        seeds = [2024]
    elif preset == "dropout":
        dropouts = [0.1, 0.2, 0.3]
    else:
        raise ValueError(f"unknown preset: {preset}")
    return [RunSpec(seed=seed, dropout=dropout) for dropout in dropouts for seed in seeds]


def id_spec_name(spec: RunSpec) -> str:
    dp_tag = str(spec.dropout).replace(".", "p")
    return f"id_seed{spec.seed}_lr{spec.lr:.0e}_dp{dp_tag}"


def run_tag(spec: RunSpec) -> str:
    return (
        f"statics2011_mobilekt4_d{spec.d}"
        f"_lr{spec.lr:.0e}"
        f"_wd1e-05_nd5_dp{spec.dropout}"
    )


def teacher_ckpt(args: argparse.Namespace, spec: RunSpec) -> Path:
    return (
        ROOT
        / "experiments"
        / args.teacher_session
        / id_spec_name(spec)
        / run_tag(spec)
        / f"mobilekt_{run_tag(spec)}_best.pt"
    )


def parse_metrics(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    auc = re.search(r"^\s*AUC\s*:\s*([0-9.]+)", text, flags=re.MULTILINE)
    acc = re.search(r"^\s*ACC\s*:\s*([0-9.]+)", text, flags=re.MULTILINE)
    qcos = re.search(r"^\s*q cosine\s*:\s*([-0-9.]+)", text, flags=re.MULTILINE)
    dr = re.search(r"^\s*diff Pearson\s*:\s*([-0-9.]+|nan)", text, flags=re.MULTILINE)
    lm = re.search(r"^\s*logit MSE\s*:\s*([-0-9.]+)", text, flags=re.MULTILINE)
    return {
        "status": "done" if auc and acc else "incomplete",
        "test_auc": auc.group(1) if auc else "",
        "test_acc": acc.group(1) if acc else "",
        "q_cos": qcos.group(1) if qcos else "",
        "diff_pearson": dr.group(1) if dr else "",
        "logit_mse": lm.group(1) if lm else "",
    }


def command(args: argparse.Namespace, spec: RunSpec) -> list[str]:
    cmd = [
        sys.executable,
        "tools/train_qe_distill.py",
        "--teacher_ckpt",
        str(teacher_ckpt(args, spec)),
        "--dataset",
        "statics2011",
        "--data_dir",
        str(args.data_dir),
        "--question_features_path",
        str(args.question_features_path),
        "--d",
        str(spec.d),
        "--mikt_state_dim",
        str(spec.state_dim),
        "--batch_size",
        str(args.batch_size),
        "--n_epochs",
        str(args.n_epochs),
        "--patience",
        str(args.patience),
        "--lr",
        str(args.lr),
        "--dropout",
        str(spec.dropout),
        "--device",
        "cuda",
        "--seed",
        str(spec.seed),
        "--session",
        f"{args.session}/{spec.name}",
        "--q_loss_weight",
        str(args.q_loss_weight),
        "--diff_loss_weight",
        str(args.diff_loss_weight),
        "--logit_loss_weight",
        str(args.logit_loss_weight),
        "--kt_loss_weight",
        str(args.kt_loss_weight),
        "--backbone_mode",
        args.backbone_mode,
    ]
    return cmd


def expected_run_dir(args: argparse.Namespace, spec: RunSpec) -> Path:
    teacher_tag = id_spec_name(spec)
    loss_tag = f"q{args.q_loss_weight:g}_d{args.diff_loss_weight:g}_logit{args.logit_loss_weight:g}_kt{args.kt_loss_weight:g}"
    name = (
        f"statics2011_qe_{args.backbone_mode}_{teacher_tag}"
        f"_seed{spec.seed}_lr{args.lr:.0e}_dp{spec.dropout}_{loss_tag}"
    )
    return ROOT / "experiments" / args.session / spec.name / name


def write_summary(args: argparse.Namespace, specs: list[RunSpec]) -> Path:
    out = ROOT / "experiments" / args.session / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specs:
        run_dir = expected_run_dir(args, spec)
        metrics = parse_metrics(run_dir / "train.log")
        metrics_json = run_dir / "metrics.json"
        if metrics_json.exists():
            try:
                data = json.loads(metrics_json.read_text())
                metrics.update(
                    {
                        "status": "done",
                        "test_auc": f"{data['test']['auc']:.4f}",
                        "test_acc": f"{data['test']['acc']:.4f}",
                        "q_cos": f"{data['test']['q_cos']:.4f}",
                        "diff_pearson": f"{data['test']['diff_pearson']:.4f}",
                        "logit_mse": f"{data['test']['logit_mse']:.6f}",
                    }
                )
            except (KeyError, json.JSONDecodeError, TypeError):
                pass
        rows.append(
            {
                "seed": spec.seed,
                "dropout": spec.dropout,
                "lr": args.lr,
                "backbone_mode": args.backbone_mode,
                "teacher_ckpt": str(teacher_ckpt(args, spec)),
                **metrics,
                "run_dir": str(run_dir),
            }
        )
    fields = ["seed", "dropout", "lr", "backbone_mode", "status", "test_auc", "test_acc", "q_cos", "diff_pearson", "logit_mse", "teacher_ckpt", "run_dir"]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["best", "core", "dropout"], default="core")
    parser.add_argument("--session", default="")
    parser.add_argument("--teacher_session", default=DEFAULT_TEACHER_SESSION)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--question_features_path", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--q_loss_weight", type=float, default=1.0)
    parser.add_argument("--diff_loss_weight", type=float, default=1.0)
    parser.add_argument("--logit_loss_weight", type=float, default=0.0)
    parser.add_argument("--kt_loss_weight", type=float, default=0.0)
    parser.add_argument("--backbone_mode", choices=["frozen", "trainable"], default="frozen")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.session = args.session or datetime.now().strftime("statics2011_qe_distill_%Y%m%d_%H%M%S")
    specs = build_specs(args.preset)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")

    missing = [str(teacher_ckpt(args, spec)) for spec in specs if not teacher_ckpt(args, spec).exists()]
    if missing:
        raise FileNotFoundError("Missing teacher checkpoint(s):\n" + "\n".join(missing))

    print(f"Session : {args.session}")
    print(f"Preset  : {args.preset} ({len(specs)} runs)")
    print(f"Teachers: {ROOT / 'experiments' / args.teacher_session}")
    print(f"GPUs    : {', '.join(gpus)}")

    pending = []
    for spec in specs:
        run_dir = expected_run_dir(args, spec)
        if (run_dir / "metrics.json").exists() and not args.force:
            print(f"skip done: {spec.name}")
        else:
            pending.append(spec)

    if args.dry_run:
        for spec in pending:
            print(" ".join(command(args, spec)))
        write_summary(args, specs)
        return 0

    active: list[tuple[subprocess.Popen[bytes], RunSpec, str, object]] = []
    next_idx = 0
    failed = 0
    while next_idx < len(pending) or active:
        active_gpus = {gpu for _, _, gpu, _ in active}
        while next_idx < len(pending) and len(active) < len(gpus):
            free_gpus = [gpu for gpu in gpus if gpu not in active_gpus]
            if not free_gpus:
                break
            spec = pending[next_idx]
            gpu = free_gpus[0]
            run_dir = expected_run_dir(args, spec)
            run_dir.mkdir(parents=True, exist_ok=True)
            log = (run_dir / "train.log").open("w", buffering=1)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            print(f"\n[start] gpu={gpu} {spec.name}", flush=True)
            proc = subprocess.Popen(command(args, spec), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            active.append((proc, spec, gpu, log))
            active_gpus.add(gpu)
            next_idx += 1

        time.sleep(10)
        still_active = []
        for proc, spec, gpu, log in active:
            code = proc.poll()
            if code is None:
                still_active.append((proc, spec, gpu, log))
                continue
            log.close()
            status = "ok" if code == 0 else f"exit={code}"
            print(f"[done]  gpu={gpu} {spec.name} {status}", flush=True)
            if code != 0:
                failed += 1
            write_summary(args, specs)
        active = still_active

    summary = write_summary(args, specs)
    print(f"\nSummary: {summary}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
