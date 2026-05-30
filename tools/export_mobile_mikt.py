#!/usr/bin/env python3
"""Export MobileKT v4 MIKT stateful mobile artifacts.

The mobile runtime owns only the KT engine state. The server-side Question
Encoder returns a MIKT-compatible question embedding and difficulty scalar.

This exporter writes:
  - stateful ONNX predict/update models
  - initial student state
  - API/model contracts for the mobile and QE server teams
  - validation fixtures generated from the trained checkpoint
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import MobileKTConfig
from datasets.kt_dataset import _parse_csv
from models import MobileKTV4


DEFAULT_RUN_DIR = (
    ROOT
    / "experiments"
    / "statics2011_qe_e2e_teacher_guided_best_20260528"
    / "qe_seed2024_lr1e-03_dp0p1"
    / "statics2011_qe_trainable_id_seed2024_lr1e-03_dp0p1_seed2024_lr1e-03_dp0.1_q1_d1_logit1_kt1"
)
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "qe_distill_best.pt"
DEFAULT_METRICS = DEFAULT_RUN_DIR / "metrics.json"
DEFAULT_DATA_ROOT = ROOT.parents[1] / "data" / "datasets" / "KT" / "statics2011"
DEFAULT_OUT_DIR = ROOT / "export"


class MobileMIKTStepBase(nn.Module):
    """ONNX-friendly step implementation for MobileKT's MIKT backbone."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.n_concepts = backbone.n_concepts
        self.d = backbone.d
        self.state_d = backbone.state_d
        self.max_seq_len = backbone.max_seq_len
        concept_range = torch.arange(self.n_concepts + 1, dtype=torch.long)
        self.register_buffer("concept_range", concept_range, persistent=False)

    def _valid_concept_mask(self, concept_ids: torch.Tensor) -> torch.Tensor:
        return (concept_ids > 0) & (concept_ids <= self.n_concepts)

    def _related_concepts(self, concept_ids: torch.Tensor) -> torch.Tensor:
        valid = self._valid_concept_mask(concept_ids)
        safe_ids = concept_ids.clamp(min=0, max=self.n_concepts)
        matches = self.concept_range.view(1, -1, 1) == safe_ids.unsqueeze(1)
        related = (matches & valid.unsqueeze(1)).any(dim=-1)
        return related & (self.concept_range.view(1, -1) > 0)

    def _masked_softmax(self, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = scores.masked_fill(~mask, -1.0e9)
        weights = torch.softmax(masked, dim=-1) * mask.to(scores.dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        return weights / denom

    def _build_item_embedding(
        self,
        question_embedding: torch.Tensor,
        difficulty: torch.Tensor,
        concept_ids: torch.Tensor,
    ) -> torch.Tensor:
        valid = self._valid_concept_mask(concept_ids)
        safe_ids = concept_ids.clamp(min=0, max=self.n_concepts)
        skill_emb = self.backbone.skill_embed(safe_ids)

        q_query = self.backbone.pro_linear(question_embedding).unsqueeze(1)
        c_key = self.backbone.skill_linear(skill_emb)
        scores = torch.bmm(q_query, c_key.transpose(1, 2)).squeeze(1) / (float(self.d) ** 0.5)
        alpha = self._masked_softmax(scores, valid)

        skill_attn = torch.bmm(alpha.unsqueeze(1), skill_emb).squeeze(1)
        denom = valid.to(question_embedding.dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
        skill_mean = (skill_emb * valid.unsqueeze(-1).to(skill_emb.dtype)).sum(dim=1) / denom
        diff_direction = difficulty.unsqueeze(-1) * self.backbone.pro_change(skill_mean)
        return question_embedding + skill_attn + diff_direction

    def _decay_for_concepts(
        self,
        skill_state: torch.Tensor,
        all_state: torch.Tensor,
        last_skill_time: torch.Tensor,
        concept_ids: torch.Tensor,
        step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = skill_state.shape[0]
        one_gap = torch.ones(batch_size, dtype=torch.long, device=skill_state.device)
        all_gap_embed = self.backbone.time_state[one_gap]
        decayed_all = all_state * self.backbone.all_forget(torch.cat([all_state, all_gap_embed], dim=-1))

        step = step.to(last_skill_time.dtype).view(-1, 1)
        gap = (step - last_skill_time).clamp(min=0.0, max=float(self.max_seq_len)).long()
        gap_embed = self.backbone.time_state[gap]
        all_effect = decayed_all.unsqueeze(1).expand_as(skill_state)
        forget = self.backbone.skill_forget(torch.cat([skill_state, gap_embed, all_effect], dim=-1))
        related = self._related_concepts(concept_ids).unsqueeze(-1)
        forget = torch.where(related, forget, torch.ones_like(forget))
        return skill_state * forget, decayed_all

    def _predict_from_state(
        self,
        item_embedding: torch.Tensor,
        difficulty: torch.Tensor,
        concept_ids: torch.Tensor,
        skill_state: torch.Tensor,
        all_state: torch.Tensor,
        last_skill_time: torch.Tensor,
        step: torch.Tensor,
    ) -> torch.Tensor:
        decayed_skill, decayed_all = self._decay_for_concepts(
            skill_state, all_state, last_skill_time, concept_ids, step
        )
        related = self._related_concepts(concept_ids)
        q_state = self.backbone.question_to_state(item_embedding).unsqueeze(1)
        scores = torch.bmm(q_state, decayed_skill.transpose(1, 2)).squeeze(1) / (
            float(self.state_d) ** 0.5
        )
        alpha = self._masked_softmax(scores, related)
        need_state = torch.bmm(alpha.unsqueeze(1), decayed_skill).squeeze(1)

        gate = torch.sigmoid(
            self.backbone.predict_attn(torch.cat([need_state, decayed_all, item_embedding], dim=-1))
        )
        fused_state = torch.cat([(1.0 - gate) * need_state, gate * decayed_all], dim=-1)
        ability = torch.sigmoid(
            self.backbone.pro_ability(torch.cat([fused_state, item_embedding], dim=-1)).squeeze(-1)
        )
        return torch.sigmoid(self.backbone.output_scale * (ability - torch.sigmoid(difficulty)))

    def _update_state(
        self,
        item_embedding: torch.Tensor,
        concept_ids: torch.Tensor,
        response: torch.Tensor,
        skill_state: torch.Tensor,
        all_state: torch.Tensor,
        last_skill_time: torch.Tensor,
        step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decayed_skill, decayed_all = self._decay_for_concepts(
            skill_state, all_state, last_skill_time, concept_ids, step
        )
        related = self._related_concepts(concept_ids)
        valid = related.any(dim=-1)
        x = item_embedding + self.backbone.answer_embed(response.long().clamp(0, 1))
        next_all = decayed_all + torch.tanh(self.backbone.all_obtain(x))

        to_get = self.backbone.now_obtain(x).unsqueeze(1)
        scores = torch.bmm(to_get, decayed_skill.transpose(1, 2)).squeeze(1) / (
            float(self.state_d) ** 0.5
        )
        alpha = self._masked_softmax(scores, related)
        delta = alpha.unsqueeze(-1) * to_get
        next_skill = decayed_skill + delta

        skill_state = torch.where(valid.view(-1, 1, 1), next_skill, skill_state)
        all_state = torch.where(valid.view(-1, 1), next_all, all_state)
        next_last = torch.where(related & valid.view(-1, 1), step.view(-1, 1), last_skill_time)
        return skill_state, all_state, next_last


class MobileMIKTPredict(MobileMIKTStepBase):
    def forward(
        self,
        question_embedding: torch.Tensor,
        difficulty: torch.Tensor,
        concept_ids: torch.Tensor,
        skill_state: torch.Tensor,
        all_state: torch.Tensor,
        last_skill_time: torch.Tensor,
        step: torch.Tensor,
    ) -> torch.Tensor:
        item_embedding = self._build_item_embedding(question_embedding, difficulty, concept_ids)
        return self._predict_from_state(
            item_embedding,
            difficulty,
            concept_ids,
            skill_state,
            all_state,
            last_skill_time,
            step,
        )


class MobileMIKTUpdate(MobileMIKTStepBase):
    def forward(
        self,
        question_embedding: torch.Tensor,
        difficulty: torch.Tensor,
        concept_ids: torch.Tensor,
        response: torch.Tensor,
        skill_state: torch.Tensor,
        all_state: torch.Tensor,
        last_skill_time: torch.Tensor,
        step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item_embedding = self._build_item_embedding(question_embedding, difficulty, concept_ids)
        return self._update_state(
            item_embedding,
            concept_ids,
            response,
            skill_state,
            all_state,
            last_skill_time,
            step,
        )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_digest(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def load_model(checkpoint_path: Path) -> tuple[MobileKTV4, dict[str, Any], dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["student_state_dict"] if "student_state_dict" in ckpt else ckpt
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}

    cfg = MobileKTConfig(
        dataset=ckpt_args.get("dataset", "statics2011"),
        data_dir=ckpt_args.get("data_dir", str(DEFAULT_DATA_ROOT.parent)),
        max_seq_len=int(ckpt_args.get("max_seq_len", 200)),
        d=int(ckpt_args.get("d", 64)),
        qde_hidden=int(ckpt_args.get("qde_hidden", 128)),
        qe_input_mode="features",
        question_feature_dim=int(meta.get("question_feature_dim", 1024)),
        question_features_path=ckpt_args.get(
            "question_features_path", str(DEFAULT_DATA_ROOT / "question_harrier_features.pt")
        ),
        use_diff_bias=True,
        mikt_state_dim=int(ckpt_args.get("mikt_state_dim", 64)),
        mikt_output_scale=float(ckpt_args.get("mikt_output_scale", 5.0)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
    )
    cfg.n_questions = int(meta.get("n_questions", 1223))
    cfg.n_concepts = int(meta.get("n_concepts", 640))
    cfg.model = "mobilekt4"  # type: ignore[attr-defined]

    model = MobileKTV4(cfg)
    model.load_state_dict(state_dict)
    model.eval()
    return model, ckpt_args, meta


def load_question_features(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    return obj["features"].float() if isinstance(obj, dict) else obj.float()


def choose_validation_sample(data_root: Path, max_concepts: int) -> dict[str, Any]:
    sequences = _parse_csv(str(data_root / "valid.csv"))
    for seq in sequences:
        if seq["question_ids"]:
            qid = int(seq["question_ids"][0])
            concepts = [int(c) for c in seq["concept_ids"][0] if int(c) > 0]
            response = int(seq["responses"][0])
            concepts = (concepts + [-1] * max_concepts)[:max_concepts]
            return {"question_id": qid, "concept_ids": concepts, "response": response, "step": 0}
    raise RuntimeError("No validation sample found")


def export_onnx(
    model: nn.Module,
    args: tuple[torch.Tensor, ...],
    path: Path,
    input_names: list[str],
    output_names: list[str],
) -> None:
    torch.onnx.export(
        model,
        args,
        str(path),
        input_names=input_names,
        output_names=output_names,
        opset_version=18,
        external_data=False,
        do_constant_folding=True,
        dynamic_axes={name: {0: "batch"} for name in input_names + output_names},
    )


def build_concept_catalog(data_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    concept2id = read_json(data_root / "concept2id.json")
    kc_sources: dict[str, set[str]] = {}
    kc_list = data_root / "kc_list.csv"
    if kc_list.exists():
        with kc_list.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    model_name, _, name = row[0].strip(), row[1].strip(), row[2].strip()
                    if name:
                        kc_sources.setdefault(name, set()).add(model_name)

    concepts = []
    for key, cid in sorted(concept2id.items(), key=lambda kv: int(kv[1])):
        source_models = sorted(kc_sources.get(key, []))
        concepts.append(
            {
                "id": int(cid),
                "key": key,
                "display_name": key.replace("_", " "),
                "source_models": source_models,
                "is_f2011_atomic": "F2011" in source_models,
                "is_unique_step_fallback": "Unique-step" in source_models or key.startswith("KC"),
            }
        )

    return concept2id, {
        "schema_version": "1.0",
        "padding_id": 0,
        "unknown_or_padding_concept_id": -1,
        "n_concepts": len(concepts),
        "concepts": concepts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max_concepts", type=int, default=10)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ["mobile_mikt_predict.onnx.data", "mobile_mikt_update.onnx.data"]:
        stale_path = args.out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    model, ckpt_args, meta = load_model(args.checkpoint)
    backbone = model.backbone
    predict_model = MobileMIKTPredict(backbone).eval()
    update_model = MobileMIKTUpdate(backbone).eval()

    batch = 1
    d = backbone.d
    state_d = backbone.state_d
    n_concepts = backbone.n_concepts
    k1 = n_concepts + 1
    max_c = args.max_concepts

    with torch.no_grad():
        initial_skill, initial_all, initial_last = backbone.initial_state(batch, torch.device("cpu"))

    sample = choose_validation_sample(args.data_root, max_c)
    features = load_question_features(args.data_root / "question_harrier_features.pt")
    qid = int(sample["question_id"])
    with torch.no_grad():
        encoded = model.encode_questions(question_features=features[qid].view(1, -1))
    q_embedding = encoded.embedding.detach().float()
    difficulty = encoded.difficulty.detach().float()
    concept_ids = torch.tensor([sample["concept_ids"]], dtype=torch.long)
    response = torch.tensor([sample["response"]], dtype=torch.long)
    step = torch.tensor([float(sample["step"])], dtype=torch.float32)

    predict_path = args.out_dir / "mobile_mikt_predict.onnx"
    update_path = args.out_dir / "mobile_mikt_update.onnx"
    export_onnx(
        predict_model,
        (q_embedding, difficulty, concept_ids, initial_skill, initial_all, initial_last, step),
        predict_path,
        [
            "question_embedding",
            "difficulty",
            "concept_ids",
            "skill_state",
            "all_state",
            "last_skill_time",
            "step",
        ],
        ["pred_correct"],
    )
    export_onnx(
        update_model,
        (q_embedding, difficulty, concept_ids, response, initial_skill, initial_all, initial_last, step),
        update_path,
        [
            "question_embedding",
            "difficulty",
            "concept_ids",
            "response",
            "skill_state",
            "all_state",
            "last_skill_time",
            "step",
        ],
        ["next_skill_state", "next_all_state", "next_last_skill_time"],
    )

    onnx_check = {"schema_version": "1.0", "models": {}}
    for name, path in {"predict": predict_path, "update": update_path}.items():
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        onnx_check["models"][name] = {
            "file": path.name,
            "checker": "passed",
            "opsets": [
                {"domain": opset.domain, "version": int(opset.version)}
                for opset in onnx_model.opset_import
            ],
            "inputs": [node.name for node in onnx_model.graph.input],
            "outputs": [node.name for node in onnx_model.graph.output],
        }
    write_json(args.out_dir / "onnx_check_report.json", onnx_check)

    with torch.no_grad():
        pred = predict_model(q_embedding, difficulty, concept_ids, initial_skill, initial_all, initial_last, step)
        next_skill, next_all, next_last = update_model(
            q_embedding, difficulty, concept_ids, response, initial_skill, initial_all, initial_last, step
        )

    initial_state_path = args.out_dir / "mobile_mikt_initial_state.npz"
    np.savez_compressed(
        initial_state_path,
        skill_state=initial_skill.numpy().astype(np.float32),
        all_state=initial_all.numpy().astype(np.float32),
        last_skill_time=initial_last.numpy().astype(np.float32),
    )
    expected_update_path = args.out_dir / "validation_expected_update_state.npz"
    np.savez_compressed(
        expected_update_path,
        next_skill_state=next_skill.numpy().astype(np.float32),
        next_all_state=next_all.numpy().astype(np.float32),
        next_last_skill_time=next_last.numpy().astype(np.float32),
    )

    sample_input = {
        "schema_version": "1.0",
        "state_source": "mobile_mikt_initial_state.npz",
        "question_id": qid,
        "question_embedding": q_embedding.squeeze(0).tolist(),
        "difficulty": float(difficulty.item()),
        "concept_ids": sample["concept_ids"],
        "response": int(sample["response"]),
        "step": float(sample["step"]),
    }
    write_json(args.out_dir / "validation_sample_input.json", sample_input)

    validation = {
        "schema_version": "1.0",
        "purpose": "Golden fixture for app-side ONNX Runtime wiring.",
        "sample_input": "validation_sample_input.json",
        "initial_state": "mobile_mikt_initial_state.npz",
        "expected_update_state": "validation_expected_update_state.npz",
        "expected": {
            "pred_correct": float(pred.item()),
            "next_skill_state_sha256": array_digest(next_skill.numpy()),
            "next_all_state_sha256": array_digest(next_all.numpy()),
            "next_last_skill_time_sha256": array_digest(next_last.numpy()),
            "next_skill_state_l2_norm": float(next_skill.norm().item()),
            "next_all_state_l2_norm": float(next_all.norm().item()),
            "updated_concept_ids": [cid for cid in sample["concept_ids"] if cid > 0],
        },
        "tolerance": {
            "pred_correct_abs": 1e-5,
            "state_element_abs": 1e-5,
        },
    }
    write_json(args.out_dir / "export_validation.json", validation)

    concept2id, concept_catalog = build_concept_catalog(args.data_root)
    write_json(args.out_dir / "concept_id_map.json", concept2id)
    write_json(args.out_dir / "concept_catalog.json", concept_catalog)

    state_contract = {
        "schema_version": "1.0",
        "stateful_mobile_engine": True,
        "state_files": {
            "initial_state": "mobile_mikt_initial_state.npz",
        },
        "state_tensors": {
            "skill_state": {"dtype": "float32", "shape": ["batch", k1, state_d]},
            "all_state": {"dtype": "float32", "shape": ["batch", state_d]},
            "last_skill_time": {"dtype": "float32", "shape": ["batch", k1]},
        },
        "state_semantics": {
            "skill_state": "Per-concept latent knowledge vector. This is not directly user-facing mastery.",
            "all_state": "Global student latent state used by MIKT prediction/update.",
            "last_skill_time": "Per-concept last update step for elapsed-step forgetting.",
        },
        "client_storage": {
            "one_state_per_user_profile": True,
            "increment_step_after_each_answer": True,
            "reset_by_reloading_initial_state": True,
        },
    }
    write_json(args.out_dir / "mikt_state_contract.json", state_contract)

    predict_contract = {
        "schema_version": "1.0",
        "engine_type": "MobileKT v4 stateful MIKT KT-Engine",
        "onnx_models": {
            "predict": {
                "file": "mobile_mikt_predict.onnx",
                "onnx_opset": 18,
                "inputs": {
                    "question_embedding": {"dtype": "float32", "shape": ["batch", d]},
                    "difficulty": {"dtype": "float32", "shape": ["batch"]},
                    "concept_ids": {"dtype": "int64", "shape": ["batch", max_c], "padding_value": -1},
                    "skill_state": {"dtype": "float32", "shape": ["batch", k1, state_d]},
                    "all_state": {"dtype": "float32", "shape": ["batch", state_d]},
                    "last_skill_time": {"dtype": "float32", "shape": ["batch", k1]},
                    "step": {"dtype": "float32", "shape": ["batch"]},
                },
                "outputs": {
                    "pred_correct": {"dtype": "float32", "shape": ["batch"], "range": [0.0, 1.0]},
                },
            },
            "update": {
                "file": "mobile_mikt_update.onnx",
                "onnx_opset": 18,
                "inputs": {
                    "question_embedding": {"dtype": "float32", "shape": ["batch", d]},
                    "difficulty": {"dtype": "float32", "shape": ["batch"]},
                    "concept_ids": {"dtype": "int64", "shape": ["batch", max_c], "padding_value": -1},
                    "response": {"dtype": "int64", "shape": ["batch"], "values": {"incorrect": 0, "correct": 1}},
                    "skill_state": {"dtype": "float32", "shape": ["batch", k1, state_d]},
                    "all_state": {"dtype": "float32", "shape": ["batch", state_d]},
                    "last_skill_time": {"dtype": "float32", "shape": ["batch", k1]},
                    "step": {"dtype": "float32", "shape": ["batch"]},
                },
                "outputs": {
                    "next_skill_state": {"dtype": "float32", "shape": ["batch", k1, state_d]},
                    "next_all_state": {"dtype": "float32", "shape": ["batch", state_d]},
                    "next_last_skill_time": {"dtype": "float32", "shape": ["batch", k1]},
                },
            },
        },
        "call_order": [
            "Load or initialize local user state.",
            "Call QE server or local cache to get question_embedding, difficulty, and concept_ids.",
            "Run mobile_mikt_predict.onnx before the learner answers.",
            "After the answer is known, run mobile_mikt_update.onnx and persist returned state tensors.",
            "Increment the app-side step counter by one.",
        ],
    }
    write_json(args.out_dir / "mikt_predict_contract.json", predict_contract)

    kc_mapping_contract = {
        "schema_version": "1.0",
        "concept_id_map": "concept_id_map.json",
        "concept_catalog": "concept_catalog.json",
        "concept_id_base": 1,
        "padding_values": {"concept_ids_input_padding": -1, "state_padding_concept_column": 0},
        "max_concepts_per_question": max_c,
        "rules": [
            "Every concept id sent to the mobile ONNX engine must exist in concept_id_map.json.",
            "Pad unused concept slots with -1.",
            "State column 0 is reserved and must not be displayed as a real concept.",
            "Generated-question services should return concept keys and ids together for debugging.",
        ],
    }
    write_json(args.out_dir / "kc_mapping_contract.json", kc_mapping_contract)

    qe_contract = {
        "schema_version": "1.0",
        "service_name": "MobileKT Question Representation Service",
        "endpoint": {
            "method": "POST",
            "path": "/v1/question/encode",
        },
        "request": {
            "content_type": "application/json",
            "body": {
                "question": "string",
                "options": [{"label": "string", "text": "string"}],
                "visual_description": "string optional",
                "concept_keys": ["string optional"],
                "question_type": "string optional",
                "client_question_id": "string optional",
            },
        },
        "response": {
            "content_type": "application/json",
            "body": {
                "question_hash": "string",
                "qe_model_version": "string",
                "mikt_compatibility_version": "string",
                "embedding_dim": d,
                "embedding_dtype": "float32",
                "question_embedding": ["float32 length 64"],
                "difficulty": "float32 scalar",
                "concept_ids": ["int64, length <= max_concepts_per_question"],
                "concept_keys": ["string"],
            },
        },
        "caching": {
            "cache_key": "question_hash + qe_model_version + mikt_compatibility_version",
            "client_can_cache": True,
        },
    }
    write_json(args.out_dir / "qe_server_api_contract.json", qe_contract)

    compatibility = {
        "schema_version": "1.0",
        "mikt_compatibility_version": "mobilekt4-stat-20260528-qe-e2e-teacher-seed2024-dp0.1",
        "mobile_backbone_checkpoint": str(args.checkpoint),
        "qe_input_feature_encoder": "microsoft/harrier-oss-v1-0.6b",
        "qe_server_question_encoder_checkpoint": "qe_server_question_encoder.pt",
        "embedding_dim": d,
        "difficulty_dtype": "float32",
        "difficulty_semantics": "MIKT internal latent difficulty, not Bloom difficulty.",
        "n_concepts": n_concepts,
        "state_dim": state_d,
        "max_concepts_per_question": max_c,
        "concept_vocab": {
            "file": "concept_id_map.json",
            "source": str(args.data_root / "concept2id.json"),
            "sha256": sha256_file(args.data_root / "concept2id.json"),
        },
        "question_feature_cache": {
            "source": str(args.data_root / "question_harrier_features.pt"),
        },
        "training_args": ckpt_args,
        "meta": meta,
    }
    write_json(args.out_dir / "qe_mikt_compatibility.json", compatibility)

    metrics = read_json(args.metrics) if args.metrics.exists() else {}
    evaluation = {
        "schema_version": "1.0",
        "model": "MobileKT v4 QE-E2E+Teacher",
        "dataset": "statics2011",
        "split": "student-level random split, seed 42",
        "checkpoint": str(args.checkpoint),
        "metrics_source": str(args.metrics),
        "test": metrics.get("test", {}),
        "best_val": metrics.get("best_val", {}),
        "notes": [
            "This is not yet an unseen-question split result.",
            "pred_correct is calibrated only to the current student-level split.",
            "App-facing mastery should come from TAP/readout in a later export, not directly from skill_state vector values.",
        ],
    }
    write_json(args.out_dir / "evaluation_report.json", evaluation)

    torch.save(model.question_encoder.state_dict(), args.out_dir / "qe_server_question_encoder.pt")
    write_json(
        args.out_dir / "qe_server_config.json",
        {
            "schema_version": "1.0",
            "question_encoder_class": "MIKTQuestionEncoder",
            "input_mode": "features",
            "feature_dim": int(meta.get("question_feature_dim", 1024)),
            "output_embedding_dim": d,
            "output_difficulty_shape": ["batch"],
            "checkpoint": "qe_server_question_encoder.pt",
        },
    )
    if args.metrics.exists():
        shutil.copy2(args.metrics, args.out_dir / "source_metrics.json")

    manifest_files = sorted(
        p for p in args.out_dir.iterdir() if p.is_file() and p.name != "mobile_export_manifest.json"
    )
    manifest = {
        "schema_version": "1.0",
        "created_by": "tools/export_mobile_mikt.py",
        "source_checkpoint": str(args.checkpoint),
        "files": [
            {
                "name": p.name,
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
            for p in manifest_files
        ],
    }
    write_json(args.out_dir / "mobile_export_manifest.json", manifest)

    print(json.dumps({"out_dir": str(args.out_dir), "files": [x["name"] for x in manifest["files"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
