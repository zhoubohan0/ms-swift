#!/usr/bin/env python3
"""Cloth stage 2-6 verifier server: VLM (ms-swift) based folding stage classifier.

Receives a JPEG-encoded image over TCP, runs a fine-tuned Qwen3-VL model via
ms-swift to predict the detailed folding stage (1-6).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os, sys
import socket
import struct
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configurable defaults (override via CLI args or env vars)
# ---------------------------------------------------------------------------
HOST = "0.0.0.0"
DEFAULT_PORT = 5051
DEFAULT_CKPT = str(
    Path("/mnt/nas/zbh/__backup/REPO/ms-swift/outputs")
    / "stage_20260330_sft" / "v0-20260331-111104" / "checkpoint-1700"
)
DEFAULT_CUDA_DEVICE = "1"

# swift inference parameters
SWIFT_MAX_PIXELS = "1003520"
SWIFT_VIDEO_MAX_PIXELS = "50176"

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append("/mnt/nas/zbh/__backup/REPO/ms-swift")
from swift.infer_engine import InferRequest, RequestConfig
from swift.pipelines.infer import SwiftInfer

# ---------------------------------------------------------------------------
# Model path resolution
# ---------------------------------------------------------------------------
def _looks_like_merged(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    return (
        (path / "model.safetensors").is_file()
        or (path / "model.safetensors.index.json").is_file()
        or (path / "pytorch_model.bin").is_file()
    )


def resolve_infer_model_path(ckpt: Path) -> tuple[Path, str]:
    """Return (load_path, mode) where mode is ``merged`` or ``lora``."""
    ckpt = ckpt.resolve()
    if _looks_like_merged(ckpt):
        return ckpt, "merged"
    sibling = ckpt.parent / f"{ckpt.name}-merged"
    if _looks_like_merged(sibling):
        return sibling, "merged"
    return ckpt, "lora"


# ---------------------------------------------------------------------------
# TCP helpers (length-prefixed JSON, same as stage12)
# ---------------------------------------------------------------------------
def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)


def _recv_message(conn: socket.socket) -> dict[str, Any]:
    header = _recv_exact(conn, 4)
    (msg_len,) = struct.unpack("!I", header)
    if msg_len <= 0 or msg_len > 256 * 1024 * 1024:
        raise ValueError(f"bad message length {msg_len}")
    payload = _recv_exact(conn, msg_len)
    obj = json.loads(payload.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("json root must be object")
    return obj


def _send_message(conn: socket.socket, obj: dict[str, Any]) -> None:
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    conn.sendall(struct.pack("!I", len(data)) + data)


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------
class ClothStage26VerifierService:
    """Loads ms-swift VLM model once, provides ``predict(pil_rgb)``."""

    def __init__(self, ckpt: str = DEFAULT_CKPT) -> None:
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_dir():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        load_path, load_mode = resolve_infer_model_path(ckpt_path)
        print(f"Loading model: {load_path}  mode={load_mode}", flush=True)

        self._prompt_parser_cls = self._load_prompt_parser()
        self._prompt_text = self._prompt_parser_cls()._build_prompt().strip()
        self._swift_infer = self._build_swift_infer(load_path, load_mode)

        self._InferRequest = InferRequest
        self._SwiftInfer = SwiftInfer
        self._request_config = RequestConfig(
            max_tokens=512, temperature=0.0, stream=False,
        )
        print("ClothStage26VerifierService ready", flush=True)

    @staticmethod
    def _load_prompt_parser():
        path = REPO_ROOT / "datasets" / "prompt-parse.py"
        spec = importlib.util.spec_from_file_location("cloth_prompt_parse", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.PromptParserClothFolding

    @staticmethod
    def _build_swift_infer(model_path: Path, mode: str):
        from swift.pipelines.infer import SwiftInfer

        common = [
            "--infer_backend", "transformers",
            "--torch_dtype", "bfloat16",
            "--template", "qwen3_vl",
            "--max_length", "2048",
            "--load_data_args", "false",
            "--temperature", "0",
            "--max_new_tokens", "512",
            "--stream", "false",
        ]
        if mode == "merged":
            argv = ["--model", str(model_path), "--merge_lora", "false", *common]
        elif mode == "lora":
            argv = [
                "--model", "Qwen/Qwen3-VL-4B-Instruct",
                "--adapters", str(model_path),
                "--merge_lora", "true",
                *common,
            ]
        else:
            raise ValueError(f"unknown mode: {mode}")
        return SwiftInfer(argv)

    def predict(self, pil_rgb) -> dict[str, Any]:
        """Run VLM inference on a single PIL RGB image.

        Returns ``{"success": True, "result": <int 1-6>, ...}`` on success.
        """
        infer_request = self._InferRequest(
            messages=[{"role": "user", "content": f"<image>\n{self._prompt_text}"}],
            images=[pil_rgb],
        )
        res_list = self._swift_infer.infer(
            [infer_request], self._request_config, use_tqdm=False,
        )
        raw = self._SwiftInfer.parse_data_from_response(res_list[0])

        parser = self._prompt_parser_cls()
        parsed = dict(parser._parse_llm_stage_response(raw))
        parsed["stage"] = int(parsed.get("stage", 1))

        return {
            "success": True,
            "result": parsed["stage"],
            "shape": str(parsed.get("shape", "")),
            "orientation": str(parsed.get("orientation", "")),
            "completion": str(parsed.get("completion", "")),
            "raw_response": raw,
        }


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Cloth stage 2-6 verifier TCP server")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--cuda", type=str, default=DEFAULT_CUDA_DEVICE)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.cuda)
    os.environ.setdefault("MAX_PIXELS", SWIFT_MAX_PIXELS)
    os.environ.setdefault("VIDEO_MAX_PIXELS", SWIFT_VIDEO_MAX_PIXELS)

    service = ClothStage26VerifierService(ckpt=args.ckpt)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.listen(128)
    print(
        f"Listening on {args.host}:{args.port}  "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )

    from PIL import Image

    while True:
        conn, addr = sock.accept()
        try:
            conn.settimeout(600)
            req = _recv_message(conn)
            if req.get("command") != "verify":
                _send_message(conn, {"success": False, "error": f"unknown command: {req.get('command')!r}"})
                continue

            nbytes = int(req.get("image_nbytes", 0))
            if nbytes <= 0 or nbytes > 64 * 1024 * 1024:
                _send_message(conn, {"success": False, "error": "invalid image_nbytes"})
                continue

            jpeg = _recv_exact(conn, nbytes)
            rgb = Image.open(BytesIO(jpeg)).convert("RGB")
            result = service.predict(rgb)
            _send_message(conn, result)
        except Exception as e:
            try:
                _send_message(conn, {"success": False, "error": str(e), "traceback": traceback.format_exc()})
            except OSError:
                pass
        finally:
            conn.close()


if __name__ == "__main__":
    main()
