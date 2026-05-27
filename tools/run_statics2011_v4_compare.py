#!/usr/bin/env python3
"""Run MobileKT v4 ID-table vs Harrier-QE comparisons on Statics2011.

The runner keeps each training job in a unique session folder because
``train.py`` run tags do not include seed or QE mode.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/workspace/data/datasets/KT")
HARRIER_FEATURES = DATA_DIR / "statics2011" / "question_harrier_features.pt"


@dataclass(frozen=True)
class RunSpec:
    method: str
    seed: int
    lr: float
    dropout: float
    d: int = 64
    state_dim: int = 64

    @property
    def name(self) -> str:
        lr_tag = f"{self.lr:.0e}".replace("+", "")
        dp_tag = str(self.dropout).replace(".", "p")
        return f"{self.method}_seed{self.seed}_lr{lr_tag}_dp{dp_tag}"


def build_specs(preset: str) -> list[RunSpec]:
    seeds = [42, 2024, 3407]
    if preset == "core":
        lrs = [1e-3]
        dropouts = [0.2]
    elif preset == "dropout":
        lrs = [1e-3]
        dropouts = [0.1, 0.2, 0.3]
    elif preset == "grid":
        lrs = [1e-3, 5e-4]
        dropouts = [0.1, 0.2, 0.3]
    else:
        raise ValueError(f"unknown preset: {preset}")

    return [
        RunSpec(method=method, seed=seed, lr=lr, dropout=dropout)
        for dropout in dropouts
        for lr in lrs
        for seed in seeds
        for method in ("id", "harrier")
    ]


def run_tag(spec: RunSpec) -> str:
    return (
        f"statics2011_mobilekt4_d{spec.d}"
        f"_lr{spec.lr:.0e}"
        f"_wd1e-05_nd5_dp{spec.dropout}"
    )


def run_dir(base_session: str, spec: RunSpec) -> Path:
    return ROOT / "experiments" / base_session / spec.name / run_tag(spec)


def train_log(base_session: str, spec: RunSpec) -> Path:
    return run_dir(base_session, spec) / "train.log"


def parse_log(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    auc = re.search(r"^\s*AUC\s*:\s*([0-9.]+)", text, flags=re.MULTILINE)
    acc = re.search(r"^\s*ACC\s*:\s*([0-9.]+)", text, flags=re.MULTILINE)
    best = re.findall(r"Best model saved\s+\(val_auc=([0-9.]+)\)", text)
    stop = re.search(r"Early stopping at epoch\s+([0-9]+)", text)
    last_epoch = re.findall(r"^Epoch\s+([0-9]+)\s+\|", text, flags=re.MULTILINE)
    duration = re.search(r"^\s*Time:\s*([0-9hms ]+)", text, flags=re.MULTILINE)
    return {
        "status": "done" if auc and acc else "incomplete",
        "best_val_auc": best[-1] if best else "",
        "test_auc": auc.group(1) if auc else "",
        "test_acc": acc.group(1) if acc else "",
        "stop_epoch": stop.group(1) if stop else (last_epoch[-1] if last_epoch else ""),
        "duration": duration.group(1).strip() if duration else "",
    }


def command(spec: RunSpec, session: str) -> list[str]:
    cmd = [
        sys.executable,
        "train.py",
        "--model",
        "mobilekt4",
        "--dataset",
        "statics2011",
        "--data_dir",
        str(DATA_DIR),
        "--d",
        str(spec.d),
        "--mikt_state_dim",
        str(spec.state_dim),
        "--batch_size",
        "32",
        "--n_epochs",
        "100",
        "--patience",
        "15",
        "--lr",
        str(spec.lr),
        "--dropout",
        str(spec.dropout),
        "--device",
        "cuda",
        "--seed",
        str(spec.seed),
        "--session",
        f"{session}/{spec.name}",
    ]
    if spec.method == "id":
        cmd += ["--qe_input_mode", "id"]
    else:
        cmd += [
            "--qe_input_mode",
            "features",
            "--question_features_path",
            str(HARRIER_FEATURES),
        ]
    return cmd


def write_summary(base_session: str, specs: list[RunSpec]) -> Path:
    out = ROOT / "experiments" / base_session / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for spec in specs:
        metrics = parse_log(train_log(base_session, spec))
        rows.append(
            {
                "method": spec.method,
                "seed": str(spec.seed),
                "lr": str(spec.lr),
                "dropout": str(spec.dropout),
                "d": str(spec.d),
                "state_dim": str(spec.state_dim),
                **metrics,
                "run_dir": str(run_dir(base_session, spec)),
            }
        )
    fields = [
        "method",
        "seed",
        "lr",
        "dropout",
        "d",
        "state_dim",
        "status",
        "best_val_auc",
        "test_auc",
        "test_acc",
        "stop_epoch",
        "duration",
        "run_dir",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["core", "dropout", "grid"], default="core")
    parser.add_argument("--session", default="")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    session = args.session or datetime.now().strftime("statics2011_v4_compare_%Y%m%d_%H%M%S")
    specs = build_specs(args.preset)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")

    print(f"Session: {session}")
    print(f"Preset : {args.preset} ({len(specs)} runs)")
    print(f"GPUs   : {', '.join(gpus)}")

    pending: list[RunSpec] = []
    for spec in specs:
        metrics = parse_log(train_log(session, spec))
        if metrics.get("status") == "done" and not args.force:
            print(f"skip done: {spec.name}")
        else:
            pending.append(spec)

    if args.dry_run:
        for spec in pending:
            print(" ".join(command(spec, session)))
        write_summary(session, specs)
        return 0

    active: list[tuple[subprocess.Popen[bytes], RunSpec, str]] = []
    next_idx = 0
    failed = 0
    while next_idx < len(pending) or active:
        active_gpus = {gpu for _, _, gpu in active}
        while next_idx < len(pending) and len(active) < len(gpus):
            free_gpus = [gpu for gpu in gpus if gpu not in active_gpus]
            if not free_gpus:
                break
            spec = pending[next_idx]
            gpu = free_gpus[0]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            print(f"\n[start] gpu={gpu} {spec.name}", flush=True)
            proc = subprocess.Popen(command(spec, session), cwd=ROOT, env=env)
            active.append((proc, spec, gpu))
            active_gpus.add(gpu)
            next_idx += 1

        time.sleep(10)
        still_active: list[tuple[subprocess.Popen[bytes], RunSpec, str]] = []
        for proc, spec, gpu in active:
            code = proc.poll()
            if code is None:
                still_active.append((proc, spec, gpu))
                continue
            status = "ok" if code == 0 else f"exit={code}"
            print(f"[done]  gpu={gpu} {spec.name} {status}", flush=True)
            if code != 0:
                failed += 1
            write_summary(session, specs)
        active = still_active

    summary = write_summary(session, specs)
    print(f"\nSummary: {summary}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
