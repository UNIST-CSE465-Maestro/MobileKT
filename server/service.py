"""Question representation service for MobileKT mobile runtime.

This module intentionally keeps the model/runtime logic independent from any
specific web framework. ``app.py`` exposes it through a small stdlib HTTP
server, and a FastAPI wrapper can use the same service later.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def format_raw_question(payload: dict[str, Any]) -> str:
    """Format a payload exactly like the Harrier feature builder."""
    options = payload.get("options") or []
    option_lines = [
        f"{opt.get('label', '')}. {opt.get('text', '')}".strip()
        for opt in options
        if isinstance(opt, dict)
    ]
    pieces = [
        "Question:",
        str(payload.get("question") or "").strip(),
    ]
    if option_lines:
        pieces.extend(["", "Options:", *option_lines])
    visual = str(payload.get("visual_description") or "").strip()
    if visual:
        pieces.extend(["", "Visual Description:", visual])
    return "\n".join(piece for piece in pieces if piece != "")


def canonical_question_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Build the stable payload used for request hashing and caching."""
    options = []
    for opt in payload.get("options") or []:
        if not isinstance(opt, dict):
            continue
        options.append(
            {
                "label": _clean_text(opt.get("label")),
                "text": _clean_text(opt.get("text")),
            }
        )
    return {
        "question": _clean_text(payload.get("question")),
        "options": options,
        "answer": _clean_text(payload.get("answer")),
        "solution": _clean_text(payload.get("solution")),
        "visual_description": _clean_text(payload.get("visual_description")),
        "question_type": _clean_text(payload.get("question_type")),
        "concept_keys": sorted(_clean_text(x) for x in payload.get("concept_keys") or [] if _clean_text(x)),
        "concept_ids": sorted(int(x) for x in payload.get("concept_ids") or []),
    }


@dataclass
class ServiceConfig:
    export_dir: Path = ROOT / "export"
    device: str = os.environ.get("MOBILEKT_QE_DEVICE", "cuda")
    feature_mode: str = os.environ.get("MOBILEKT_QE_FEATURE_MODE", "harrier")
    harrier_model_name: str | None = None
    max_length: int = int(os.environ.get("MOBILEKT_QE_MAX_LENGTH", "2048"))
    local_files_only: bool = os.environ.get("MOBILEKT_QE_LOCAL_FILES_ONLY", "1") != "0"


class ServiceError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

    def to_response(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message}}


class HarrierFeatureEncoder:
    def __init__(
        self,
        model_name: str,
        device: str,
        max_length: int,
        local_files_only: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.local_files_only = local_files_only
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            dtype="auto",
            local_files_only=self.local_files_only,
        )
        self._model.eval()
        self._model.to(self.device)

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        import torch

        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    def encode(self, texts: list[str]):
        import torch
        import torch.nn.functional as F

        with self._lock:
            self._load()
            batch = self._tokenizer(
                texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.inference_mode():
                outputs = self._model(**batch)
                features = self._last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
                features = F.normalize(features, p=2, dim=1)
            return features.cpu().float()


class HashFeatureEncoder:
    """Deterministic dev fallback. Not valid for research metrics."""

    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim

    def encode(self, texts: list[str]):
        import numpy as np
        import torch

        rows = []
        for text in texts:
            values = np.zeros(self.feature_dim, dtype=np.float32)
            seed = hashlib.sha256(text.encode("utf-8")).digest()
            for idx in range(self.feature_dim):
                digest = hashlib.sha256(seed + idx.to_bytes(4, "little")).digest()
                value = int.from_bytes(digest[:4], "little") / 0xFFFFFFFF
                values[idx] = (value * 2.0) - 1.0
            norm = np.linalg.norm(values)
            if norm > 0:
                values = values / norm
            rows.append(values)
        return torch.from_numpy(np.stack(rows, axis=0)).float()


class QuestionRepresentationService:
    def __init__(self, config: ServiceConfig | None = None):
        self.config = config or ServiceConfig()
        self.export_dir = self.config.export_dir
        if not self.export_dir.exists():
            raise FileNotFoundError(f"export_dir not found: {self.export_dir}")

        self.qe_config = _read_json(self.export_dir / "qe_server_config.json")
        self.compatibility = _read_json(self.export_dir / "qe_mikt_compatibility.json")
        self.concept_id_map = _read_json(self.export_dir / "concept_id_map.json")
        self.id_to_concept = {int(v): k for k, v in self.concept_id_map.items()}

        self.feature_dim = int(self.qe_config["feature_dim"])
        self.embedding_dim = int(self.qe_config["output_embedding_dim"])
        self.mikt_compatibility_version = str(self.compatibility["mikt_compatibility_version"])
        self.harrier_model_name = (
            self.config.harrier_model_name
            or self.compatibility.get("qe_input_feature_encoder")
            or "microsoft/harrier-oss-v1-0.6b"
        )
        qe_ckpt = self.export_dir / self.qe_config["checkpoint"]
        self.qe_checkpoint_sha256 = _sha256_file(qe_ckpt)
        self.qe_model_version = f"qe-harrier-mobilekt4-{self.qe_checkpoint_sha256[:12]}"

        self._question_encoder = None
        self._feature_encoder = None
        self._model_lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "mobilekt-qe",
            "qe_model_version": self.qe_model_version,
            "mikt_compatibility_version": self.mikt_compatibility_version,
            "feature_mode": self.config.feature_mode,
            "feature_dim": self.feature_dim,
            "embedding_dim": self.embedding_dim,
            "n_concepts": len(self.concept_id_map),
        }

    def _load_question_encoder(self):
        if self._question_encoder is not None:
            return self._question_encoder
        import torch
        from models.qe.question_encoder import MIKTQuestionEncoder

        if self.config.device == "cuda" and not torch.cuda.is_available():
            self.config.device = "cpu"
        encoder = MIKTQuestionEncoder(
            n_questions=0,
            d=self.embedding_dim,
            hidden=128,
            dropout=0.0,
            feature_dim=self.feature_dim,
            use_diff_bias=False,
            input_mode="features",
        )
        state = torch.load(self.export_dir / self.qe_config["checkpoint"], map_location="cpu")
        encoder.load_state_dict(state)
        encoder.eval()
        encoder.to(self.config.device)
        self._question_encoder = encoder
        return encoder

    def _load_feature_encoder(self):
        if self._feature_encoder is not None:
            return self._feature_encoder
        mode = self.config.feature_mode.lower()
        if mode == "harrier":
            self._feature_encoder = HarrierFeatureEncoder(
                model_name=self.harrier_model_name,
                device=self.config.device,
                max_length=self.config.max_length,
                local_files_only=self.config.local_files_only,
            )
        elif mode == "hash":
            self._feature_encoder = HashFeatureEncoder(self.feature_dim)
        else:
            raise ServiceError(500, "bad_feature_mode", f"Unsupported feature mode: {mode}")
        return self._feature_encoder

    def _resolve_concepts(self, payload: dict[str, Any]) -> tuple[list[int], list[str]]:
        concept_ids: list[int] = []
        concept_keys: list[str] = []

        for cid in payload.get("concept_ids") or []:
            try:
                cid_int = int(cid)
            except (TypeError, ValueError):
                raise ServiceError(422, "invalid_concept_id", f"Invalid concept id: {cid!r}")
            if cid_int not in self.id_to_concept:
                raise ServiceError(422, "unknown_concept_id", f"Unknown concept id: {cid_int}")
            if cid_int not in concept_ids:
                concept_ids.append(cid_int)
                concept_keys.append(self.id_to_concept[cid_int])

        for key in payload.get("concept_keys") or []:
            key = str(key).strip()
            if not key:
                continue
            if key not in self.concept_id_map:
                raise ServiceError(422, "unknown_concept_key", f"Unknown concept key: {key}")
            cid_int = int(self.concept_id_map[key])
            if cid_int not in concept_ids:
                concept_ids.append(cid_int)
                concept_keys.append(key)

        max_c = int(self.compatibility.get("max_concepts_per_question", 10))
        if len(concept_ids) > max_c:
            raise ServiceError(
                422,
                "too_many_concepts",
                f"Received {len(concept_ids)} concepts, max is {max_c}",
            )
        if not concept_ids:
            raise ServiceError(
                422,
                "missing_concepts",
                "concept_keys or concept_ids are required; current QE does not infer concepts.",
            )
        return concept_ids, concept_keys

    def _validate_payload(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ServiceError(400, "invalid_json", "Request body must be a JSON object.")
        if not str(payload.get("question") or "").strip():
            raise ServiceError(422, "missing_question", "question is required.")
        requested = payload.get("mikt_compatibility_version")
        if requested and requested != self.mikt_compatibility_version:
            raise ServiceError(
                409,
                "incompatible_mikt_version",
                f"Requested {requested}, server serves {self.mikt_compatibility_version}",
            )

    def encode_one(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._validate_payload(payload)
        concept_ids, concept_keys = self._resolve_concepts(payload)
        canonical = canonical_question_payload(payload)
        question_hash = hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()
        text = format_raw_question(payload)

        with self._model_lock:
            feature_encoder = self._load_feature_encoder()
            question_encoder = self._load_question_encoder()
            features = feature_encoder.encode([text])
            import torch

            if features.shape[-1] != self.feature_dim:
                raise ServiceError(
                    500,
                    "feature_dim_mismatch",
                    f"Feature dim {features.shape[-1]} != expected {self.feature_dim}",
                )
            features = features.to(self.config.device)
            with torch.inference_mode():
                encoded = question_encoder(question_features=features)
            embedding = encoded.embedding[0].detach().cpu().float().tolist()
            difficulty = float(encoded.difficulty[0].detach().cpu().float().item())

        representation_id = hashlib.sha256(
            f"{question_hash}:{self.qe_model_version}:{self.mikt_compatibility_version}".encode("utf-8")
        ).hexdigest()
        return {
            "question_hash": question_hash,
            "representation_id": representation_id,
            "qe_model_version": self.qe_model_version,
            "mikt_compatibility_version": self.mikt_compatibility_version,
            "embedding_dim": self.embedding_dim,
            "embedding_dtype": "float32",
            "question_embedding": embedding,
            "difficulty": difficulty,
            "concept_ids": concept_ids,
            "concept_keys": concept_keys,
            "max_concepts_per_question": int(self.compatibility.get("max_concepts_per_question", 10)),
            "feature_encoder": self.harrier_model_name if self.config.feature_mode == "harrier" else "hash-dev-fallback",
            "feature_mode": self.config.feature_mode,
        }

    def encode_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        requested = payload.get("mikt_compatibility_version")
        if requested and requested != self.mikt_compatibility_version:
            raise ServiceError(
                409,
                "incompatible_mikt_version",
                f"Requested {requested}, server serves {self.mikt_compatibility_version}",
            )
        questions = payload.get("questions")
        if not isinstance(questions, list) or not questions:
            raise ServiceError(422, "missing_questions", "questions must be a non-empty list.")

        items = []
        for item in questions:
            if not isinstance(item, dict):
                items.append({"status": "error", "error": {"code": "invalid_item", "message": "item is not an object"}})
                continue
            merged = dict(item)
            if requested:
                merged["mikt_compatibility_version"] = requested
            try:
                encoded = self.encode_one(merged)
                encoded["client_question_id"] = item.get("client_question_id")
                encoded["status"] = "ok"
                items.append(encoded)
            except ServiceError as exc:
                items.append(
                    {
                        "client_question_id": item.get("client_question_id"),
                        "status": "error",
                        "error": {"code": exc.code, "message": exc.message},
                    }
                )
        return {
            "items": items,
            "qe_model_version": self.qe_model_version,
            "mikt_compatibility_version": self.mikt_compatibility_version,
        }
