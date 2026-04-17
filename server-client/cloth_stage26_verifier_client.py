"""Socket client for the cloth stage 2-6 verifier server."""
from __future__ import annotations

import json
import socket
import struct
from typing import Any

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Configurable defaults
# ---------------------------------------------------------------------------
DEFAULT_HOST = "101.132.143.105"
DEFAULT_PORT = 5052


# ---------------------------------------------------------------------------
# TCP helpers (length-prefixed JSON, same as stage12)
# ---------------------------------------------------------------------------
def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)


def _recv_message(sock: socket.socket) -> dict[str, Any]:
    header = _recv_exact(sock, 4)
    (msg_len,) = struct.unpack("!I", header)
    if msg_len <= 0 or msg_len > 256 * 1024 * 1024:
        raise ValueError(f"bad message length {msg_len}")
    payload = _recv_exact(sock, msg_len)
    obj = json.loads(payload.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("json root must be object")
    return obj


def _send_message(sock: socket.socket, obj: dict[str, Any]) -> None:
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    sock.sendall(struct.pack("!I", len(data)) + data)


def _encode_image(rgb_img: Image.Image) -> bytes:
    bgr = cv2.cvtColor(np.asarray(rgb_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr)
    if not ok:
        raise ValueError("failed to encode image")
    return encoded.tobytes()


# ---------------------------------------------------------------------------
# Client class
# ---------------------------------------------------------------------------
class ClothStage26VerifierClient:
    """Client for the remote cloth stage 2-6 verifier server.

    Returns ``{"result": <int 1-6>}`` with optional detail fields.
    """

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout_s: int = 120,
    ) -> None:
        self._host = host
        self._port = port
        self._timeout_s = timeout_s

    def verify(self, rgb_img: Image.Image) -> dict[str, Any]:
        image_bytes = _encode_image(rgb_img)
        header = {
            "command": "verify",
            "image_nbytes": len(image_bytes),
        }
        with socket.create_connection(
            (self._host, self._port),
            timeout=self._timeout_s,
        ) as sock:
            _send_message(sock, header)
            sock.sendall(image_bytes)
            response = _recv_message(sock)

        if not response.get("success", False):
            raise RuntimeError(response.get("error", "server error"))

        return {
            "result": int(response["result"]),
            "shape": str(response.get("shape", "")),
            "orientation": str(response.get("orientation", "")),
            "completion": str(response.get("completion", "")),
        }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import os
    import time

    _DEFAULT_DEMO_IMAGES = [
        "/mnt/nas/zbh/verifier/clothes-verifier/datasets/case/neg/rgb/1_IMG_1657.png",
        "/mnt/nas/zbh/verifier/clothes-verifier/datasets/case/neg/rgb/1_IMG_1658.png",
    ]

    ap = argparse.ArgumentParser(description="Cloth stage 2-6 verifier client demo")
    ap.add_argument("--host", default=os.environ.get("CLOTH_STAGE26_HOST", DEFAULT_HOST))
    ap.add_argument("--port", type=int, default=int(os.environ.get("CLOTH_STAGE26_PORT", str(DEFAULT_PORT))))
    ap.add_argument("images", nargs="*", default=_DEFAULT_DEMO_IMAGES, help="RGB image paths")
    ns = ap.parse_args()

    client = ClothStage26VerifierClient(host=ns.host, port=ns.port)
    for img_path in ns.images:
        rgb = Image.open(img_path).convert("RGB")
        t0 = time.perf_counter()
        out = client.verify(rgb)
        elapsed = time.perf_counter() - t0
        print(f"{img_path} => {json.dumps(out, ensure_ascii=False)}  ({elapsed:.3f}s)")
