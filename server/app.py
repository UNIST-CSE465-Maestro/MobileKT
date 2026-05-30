#!/usr/bin/env python3
"""Stdlib HTTP API for the MobileKT server-side Question Encoder.

Run inside the Maestro Docker container:

    cd /workspace/maestro/MobileKT
    python3 -m server.app --host 0.0.0.0 --port 8091
"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .service import QuestionRepresentationService, ServiceConfig, ServiceError


def _json_bytes(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class MobileKTRequestHandler(BaseHTTPRequestHandler):
    service: QuestionRepresentationService

    server_version = "MobileKTQE/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def _send_json(self, status: int, data: Any) -> None:
        body = _json_bytes(data)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            raise ServiceError(415, "unsupported_media_type", "Content-Type must be application/json.")
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            raise ServiceError(400, "empty_body", "Request body is empty.")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ServiceError(400, "invalid_json", f"Invalid JSON: {exc.msg}")
        if not isinstance(data, dict):
            raise ServiceError(400, "invalid_json", "Request JSON must be an object.")
        return data

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send_json(200, self.service.health())
            return
        self._send_json(404, {"error": {"code": "not_found", "message": self.path}})

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/v1/question/encode":
                self._send_json(200, self.service.encode_one(payload))
            elif self.path == "/v1/question/encode-batch":
                self._send_json(200, self.service.encode_batch(payload))
            else:
                self._send_json(404, {"error": {"code": "not_found", "message": self.path}})
        except ServiceError as exc:
            self._send_json(exc.status, exc.to_response())
        except Exception as exc:  # keep server alive and return useful diagnostics
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": {"code": "internal_error", "message": str(exc)}},
            )


def build_server(args: argparse.Namespace) -> ThreadingHTTPServer:
    config = ServiceConfig(
        export_dir=Path(args.export_dir),
        device=args.device,
        feature_mode=args.feature_mode,
        harrier_model_name=args.harrier_model_name or None,
        max_length=args.max_length,
        local_files_only=not args.allow_model_download,
    )
    service = QuestionRepresentationService(config)
    MobileKTRequestHandler.service = service
    return ThreadingHTTPServer((args.host, args.port), MobileKTRequestHandler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--export_dir", default=str(Path(__file__).resolve().parents[1] / "export"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature_mode", choices=["harrier", "hash"], default="harrier")
    parser.add_argument("--harrier_model_name", default="")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--allow_model_download", action="store_true")
    args = parser.parse_args()

    server = build_server(args)
    print(
        json.dumps(
            {
                "service": "mobilekt-qe",
                "host": args.host,
                "port": args.port,
                "export_dir": args.export_dir,
                "feature_mode": args.feature_mode,
                "device": args.device,
            },
            indent=2,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
