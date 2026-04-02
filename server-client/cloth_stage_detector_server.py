#!/usr/bin/env python3
"""在 GPU 上常驻 Swift ``transformers`` 推理，对外提供与 ``cloth_flatness_detector`` 相同 socket 协议。

启动前请设置（或在下方默认值中修改）::

    export CUDA_VISIBLE_DEVICES=1
    export MAX_PIXELS=1003520

单图推理路径与 ``swift/pipelines/infer/infer.py`` 中 ``SwiftInfer`` + ``InferEngine.infer`` 一致。

本脚本位于 ``server-client/``；默认 checkpoint 与 ``datasets/prompt-parse.py`` 均相对
**ms-swift 仓库根目录**（即 ``server-client`` 的上一级）解析::

    <repo>/outputs/stage_20260325_sft/v2-20260325-213558/checkpoint-950
    <repo>/datasets/prompt-parse.py

若同级存在 ``checkpoint-950-merged``（``swift export`` / ``merge_lora`` 产物，含完整
``config.json`` + 分片权重），则**自动改用 merged 目录**为 ``--model``，不再每次
CPU merge LoRA。也可直接把 ``--ckpt`` 指到 ``.../checkpoint-950-merged``。

监听 ``0.0.0.0:4452``（可用 ``--port`` 修改）。客户端见同目录 ``cloth_stage_detector_client.py``。
若仅能通过 SSH（如 ``Host aliyun-4090-2-2`` ``Port 8903``）登录推理机，请在本地::

    ssh -p 8903 -L 4452:127.0.0.1:4452 user@<HostName>

再在客户端连接 ``127.0.0.1:4452``。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import struct
import sys
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any

# 在 import torch / swift 之前设置可见 GPU 与视觉分辨率（与 bash/infer_sft.sh 对齐）
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("MAX_PIXELS", "1003520")
os.environ.setdefault("VIDEO_MAX_PIXELS", "50176")

# 本文件在 server-client/；仓库根为上一级（swift、outputs、datasets 等所在目录）
SERVER_CLIENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_CLIENT_DIR.parent
_repo = str(REPO_ROOT)
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _looks_like_merged_full_model(path: Path) -> bool:
    """判断是否为已合并的完整权重目录（而非仅 LoRA adapter 目录）。"""
    if not path.is_dir():
        return False
    cfg = path / "config.json"
    if not cfg.is_file():
        return False
    return (
        (path / "model.safetensors").is_file()
        or (path / "model.safetensors.index.json").is_file()
        or (path / "pytorch_model.bin").is_file()
    )


def resolve_infer_model_path(ckpt: Path) -> tuple[Path, str]:
    """返回实际加载路径及模式 ``merged`` | ``lora``。

    - 若 ``ckpt`` 本身为完整 merged 目录 → 直接使用。
    - 否则若存在 ``{ckpt.name}-merged`` 且为完整权重 → 优先使用（与 ms-swift 导出目录名一致）。
    - 否则按 LoRA：``--model`` 基座 + ``--adapters`` + ``--merge_lora true``。
    """
    ckpt = ckpt.resolve()
    if _looks_like_merged_full_model(ckpt):
        return ckpt, "merged"
    sibling = ckpt.parent / f"{ckpt.name}-merged"
    if _looks_like_merged_full_model(sibling):
        return sibling, "merged"
    return ckpt, "lora"


def _load_prompt_parser():
    path = REPO_ROOT / "datasets" / "prompt-parse.py"
    spec = importlib.util.spec_from_file_location("cloth_prompt_parse", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PromptParserClothFolding


def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)


def recv_message(conn: socket.socket) -> dict[str, Any]:
    header = recv_exact(conn, 4)
    (msg_len,) = struct.unpack("!I", header)
    if msg_len <= 0 or msg_len > 256 * 1024 * 1024:
        raise ValueError(f"bad message length {msg_len}")
    payload = recv_exact(conn, msg_len)
    obj = json.loads(payload.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("json root must be object")
    return obj


def send_message(conn: socket.socket, obj: dict[str, Any]) -> None:
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    conn.sendall(struct.pack("!I", len(data)) + data)


def build_swift_infer(model_or_adapter: Path, *, mode: str):
    from swift.pipelines.infer import SwiftInfer

    model_or_adapter = model_or_adapter.resolve()
    common = [
        "--infer_backend",
        "transformers",
        "--torch_dtype",
        "bfloat16",
        "--template",
        "qwen3_vl",
        "--max_length",
        "2048",
        "--load_data_args",
        "false",
        "--temperature",
        "0",
        "--max_new_tokens",
        "512",
        "--stream",
        "false",
    ]
    if mode == "merged":
        argv = [
            "--model",
            str(model_or_adapter),
            "--merge_lora",
            "false",
            *common,
        ]
    elif mode == "lora":
        argv = [
            "--model",
            "Qwen/Qwen3-VL-4B-Instruct",
            "--adapters",
            str(model_or_adapter),
            "--merge_lora",
            "true",
            *common,
        ]
    else:
        raise ValueError(f"unknown mode: {mode}")
    return SwiftInfer(argv)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4452)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default="/mnt/nas/zbh/__backup/REPO/ms-swift/outputs/stage_20260330_sft/v0-20260331-111104/checkpoint-1700",
        help="LoRA adapter 目录，或 merged 完整权重目录；若传入 adapter 且存在同名 *-merged 则自动用 merged",
    )
    args = parser.parse_args()
    if not args.ckpt.is_dir():
        print(f"Checkpoint not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    load_path, load_mode = resolve_infer_model_path(args.ckpt)
    if load_mode == "merged" and load_path != args.ckpt.resolve():
        print(
            f"检测到 merged 权重，使用: {load_path}（未使用 LoRA 目录 {args.ckpt}）",
            flush=True,
        )
    elif load_mode == "merged":
        print(f"使用 merged 模型目录: {load_path}", flush=True)
    else:
        print(
            f"未找到同级 *-merged 完整权重，使用 LoRA + merge_lora: {load_path}",
            flush=True,
        )

    PromptParserClothFolding = _load_prompt_parser()
    prompt_text = PromptParserClothFolding()._build_prompt().strip()

    if load_mode == "lora":
        print("Loading SwiftInfer（首次可能在 CPU 上 merge LoRA，较慢）...", flush=True)
    else:
        print("Loading SwiftInfer（直接加载 merged 权重）...", flush=True)
    swift_infer = build_swift_infer(load_path, mode=load_mode)
    from swift.infer_engine import InferRequest, RequestConfig
    from swift.pipelines.infer import SwiftInfer

    # InferArguments 在 task_type 非 causal_lm 时 get_request_config 可能为 None，这里显式指定
    request_config = RequestConfig(max_tokens=512, temperature=0.0, stream=False)

    from PIL import Image

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.listen(128)
    print(
        f"cloth_stage_detector_server listening on {args.host}:{args.port} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"load_mode={load_mode} model_path={load_path}",
        flush=True,
    )

    while True:
        conn, addr = sock.accept()
        try:
            conn.settimeout(600)
            req = recv_message(conn)
            cmd = req.get("command")
            if cmd != "cloth_stage_detector":
                send_message(
                    conn,
                    {"success": False, "error": f"unknown command: {cmd!r}"},
                )
                continue
            nbytes = int(req.get("image_nbytes", 0))
            if nbytes <= 0 or nbytes > 64 * 1024 * 1024:
                send_message(conn, {"success": False, "error": "invalid image_nbytes"})
                continue
            jpeg = recv_exact(conn, nbytes)
            rgb = Image.open(BytesIO(jpeg)).convert("RGB")
            user_content = f"<image>\n{prompt_text}"
            infer_request = InferRequest(
                messages=[{"role": "user", "content": user_content}],
                images=[rgb],
            )
            res_list = swift_infer.infer(
                [infer_request], request_config, use_tqdm=False
            )
            raw = SwiftInfer.parse_data_from_response(res_list[0])
            parser_inst = PromptParserClothFolding()
            parsed = dict(parser_inst._parse_llm_stage_response(raw))
            st = parsed.get("stage", 1)
            if isinstance(st, str) and st.isdigit():
                parsed["stage"] = int(st)
            else:
                parsed["stage"] = int(st)

            send_message(
                conn,
                {
                    "success": True,
                    "raw_response": raw,
                    "parsed": parsed,
                },
            )
        except Exception as e:
            try:
                send_message(
                    conn,
                    {
                        "success": False,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    },
                )
            except OSError:
                pass
        finally:
            conn.close()


if __name__ == "__main__":
    main()
