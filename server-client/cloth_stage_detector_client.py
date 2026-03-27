"""Socket client for the remote cloth folding stage detector (Qwen3-VL SFT).

与 ``cloth_flatness_detector_node.py`` 相同 framing：4 字节大端长度 + JSON / 原始 JPEG。

**直连（不做 SSH 端口转发）**

服务端已 ``--host 0.0.0.0 --port 4452`` 时，本地只要网络能访问到推理机的 **公网 IP / 内网 IP**
（与 SSH 的 ``Host`` 别名无关，除非该名字在本机能被 DNS 或 ``/etc/hosts`` 解析到那台机器）::

    # 将 <IP> 换成推理机实际地址（如云厂商控制台里的公网 EIP）
    python cloth_stage_detector_client.py --host <IP> --port 4452 /path/to.jpg

或环境变量::

    export CLOTH_STAGE_HOST=<IP>
    export CLOTH_STAGE_PORT=4452
    python cloth_stage_detector_client.py /path/to.jpg

须在云安全组 / 本机防火墙 **放行入站 TCP 4452**（仅 SSH 放行 8903 不够）。

**经 SSH 转发（无公网暴露 4452 时）**

``~/.ssh/config`` 里 ``Port 8903`` 是 **sshd**，与业务端口无关::

    ssh -p 8903 -L 4452:127.0.0.1:4452 user@<HostName>

本地再 ``--host 127.0.0.1 --port 4452``。
"""

from __future__ import annotations

import json
import os
import socket
from typing import Any

import cv2
import numpy as np
from PIL import Image

# 与 ~/.ssh/config 中 Host 名一致；业务端口非 SSH 的 8903，见模块文档
DEFAULT_CLOTH_STAGE_DETECTOR_HOST = "101.132.143.105"
DEFAULT_CLOTH_STAGE_DETECTOR_PORT = 5052

CLOTH_STAGE_DETECTOR_METHODS = ["cloth_stage_detector"]


def _recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    chunks: list[bytes] = []
    bytes_recd = 0
    while bytes_recd < num_bytes:
        chunk = sock.recv(min(65536, num_bytes - bytes_recd))
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        chunks.append(chunk)
        bytes_recd += len(chunk)
    return b"".join(chunks)


def _recv_message(sock: socket.socket) -> dict[str, Any]:
    header = _recv_exact(sock, 4)
    msg_len = int.from_bytes(header, byteorder="big", signed=False)
    if msg_len <= 0:
        raise ValueError("Invalid message length")
    payload = _recv_exact(sock, msg_len)
    response = json.loads(payload.decode("utf-8"))
    if not isinstance(response, dict):
        raise ValueError("Invalid response payload")
    return response


def _send_message(sock: socket.socket, obj: dict[str, Any]) -> None:
    data = json.dumps(obj).encode("utf-8")
    header = len(data).to_bytes(4, byteorder="big", signed=False)
    sock.sendall(header + data)


def _encode_image(rgb_img: Image.Image) -> bytes:
    rgb_array = np.asarray(rgb_img.convert("RGB"))
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", bgr_array)
    if not success:
        raise ValueError("Failed to encode image for cloth stage detector")
    return encoded.tobytes()


class ClothStageDetectorClient:
    """远端 ``cloth_stage_detector_server`` 的客户端：单张 RGB -> 解析后的 stage 结构。"""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        timeout_s: int = 120,
    ) -> None:
        self._host = host
        self._port = port
        self._timeout_s = timeout_s

    def cloth_stage_detector(self, rgb_img: Image.Image) -> dict[str, Any]:
        """对单件衣物 RGB 图请求折叠阶段分类。

        Returns:
            ``success`` 时服务端已解析；本方法返回
            ``{"shape", "orientation", "completion", "stage", "raw_response"}``，
            其中 ``stage`` 为 int 1..6。
        """
        image_bytes = _encode_image(rgb_img)
        header = {
            "command": "cloth_stage_detector",
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
            raise RuntimeError(response.get("error", "Cloth stage detector failed"))

        parsed = response.get("parsed")
        if not isinstance(parsed, dict):
            raise ValueError("Missing or invalid 'parsed' in response")

        for k in ("shape", "orientation", "completion", "stage"):
            if k not in parsed:
                raise ValueError(f"Missing key in parsed: {k}")

        # TODO
        '''
        out = {
            "shape": str(parsed["shape"]),
            "orientation": str(parsed["orientation"]),
            "completion": str(parsed["completion"]),
            "result": int(parsed["stage"]),
            "raw_response": str(response.get("raw_response", "")),
        }
        '''
        if "stage" not in parsed:
            out = dict(raw_response=response.get("raw_response", ""))
        else:
            out = {
                "shape": str(parsed["shape"]),
                "orientation": str(parsed["orientation"]),
                "completion": str(parsed["completion"]),
                "result": int(parsed["stage"]),
            }
        return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Call remote cloth_stage_detector_server (single image).")
    ap.add_argument(
        "--host",
        default=os.environ.get("CLOTH_STAGE_HOST", DEFAULT_CLOTH_STAGE_DETECTOR_HOST),
        help="推理机可达地址（公网/内网 IP 或能解析的主机名），默认 env CLOTH_STAGE_HOST 或 aliyun-4090-2-2",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("CLOTH_STAGE_PORT", str(DEFAULT_CLOTH_STAGE_DETECTOR_PORT))),
        help="服务端口，默认 4452 或 env CLOTH_STAGE_PORT",
    )
    ap.add_argument(
        "image",
        nargs="?",
        default=os.environ.get("CLOTH_STAGE_DEMO_IMAGE", ""),
        help="输入 RGB 图像路径；也可用 env CLOTH_STAGE_DEMO_IMAGE",
    )
    ns = ap.parse_args()
    if not ns.image:
        ap.error("请提供 image 路径，或设置 CLOTH_STAGE_DEMO_IMAGE")

    client = ClothStageDetectorClient(host=ns.host, port=ns.port)
    out = client.cloth_stage_detector(Image.open(ns.image).convert("RGB"))
    print(json.dumps(out, ensure_ascii=False, indent=2))
