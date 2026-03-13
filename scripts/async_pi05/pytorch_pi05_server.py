"""
PyTorch Pi0.5 WebSocket server — drop-in replacement for async_pi05_websocket_server.py.

Same protocol as the JAX server:
  1. On connect: sends metadata JSON
  2. Receives request JSON with {"mode": "extract_latent"|null, "images": {...},
                                  "high_level_prompt": ..., "state": [...]}
  3. Responds with:
     - extract_latent: {"status": "success", "s2_latent": [...2048 floats...], "timing": {...}}
     - full inference: {"status": "success", "actions": [...], "subtask": "...", "timing": {...}}

Usage:
    cd /home/feit/Documents/openpi_subtask
    .venv/bin/python scripts/async_pi05/pytorch_pi05_server.py \\
        --checkpoint ~/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch \\
        --norm-stats ~/.cache/openpi/checkpoints/soarm-pi05-state-11997/assets/thewisp/cylinder_ring_assembly/norm_stats.json \\
        --port 8765 --device cuda:0
"""

import argparse
import asyncio
import base64
import json
import logging
import sys
from pathlib import Path

import numpy as np
import websockets

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from pytorch_pi05_inference import (  # noqa: E402
    SOARM_IMAGE_KEYS,
    PyTorchPi05Inference,
)

logger = logging.getLogger(__name__)

# Key mapping: various input keys → canonical model keys
INPUT_TO_MODEL_IMAGE_KEYS = {
    "agentview_rgb": "base_0_rgb",
    "wrist_rgb_left": "left_wrist_0_rgb",
    "wrist_rgb": "left_wrist_0_rgb",
    "front": "base_0_rgb",
    "top": "base_1_rgb",
    "left_wrist": "left_wrist_0_rgb",
    "right_wrist": "right_wrist_0_rgb",
    "base_0_rgb": "base_0_rgb",
    "base_1_rgb": "base_1_rgb",
    "left_wrist_0_rgb": "left_wrist_0_rgb",
    "right_wrist_0_rgb": "right_wrist_0_rgb",
}

TRAINING_KEY_ORDER = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "base_1_rgb")


def _decode_image(img_data) -> np.ndarray:
    """Decode image from shm_path, base64 dict, or nested list."""
    if isinstance(img_data, dict) and "shm_path" in img_data:
        return np.load(img_data["shm_path"])
    if isinstance(img_data, dict) and "base64" in img_data:
        raw = base64.b64decode(img_data["base64"])
        return np.frombuffer(raw, dtype=np.uint8).reshape(img_data["shape"])
    return np.array(img_data, dtype=np.uint8)


def _map_image_keys(images: dict) -> dict:
    """Rename keys to model keys and enforce training key order."""
    renamed = {INPUT_TO_MODEL_IMAGE_KEYS.get(k, k): v for k, v in images.items()}
    if "right_wrist_0_rgb" not in renamed and renamed:
        renamed["right_wrist_0_rgb"] = np.zeros_like(next(iter(renamed.values())))
    ordered = {k: renamed[k] for k in TRAINING_KEY_ORDER if k in renamed}
    for k in renamed:
        if k not in ordered:
            ordered[k] = renamed[k]
    return ordered


class PyTorchPi05Server:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        host: str = "0.0.0.0",
        port: int = 8765,
        norm_stats_path: str | None = None,
    ):
        self.host = host
        self.port = port
        self.engine = PyTorchPi05Inference(
            checkpoint_path=checkpoint_path,
            device=device,
            image_keys=SOARM_IMAGE_KEYS,
            norm_stats_path=norm_stats_path,
        )

    async def handle_client(self, websocket, path: str | None = None):
        logger.info("Client connected: %s", websocket.remote_address)
        metadata = {
            "model": "pytorch_pi05",
            "server_type": "PyTorchPi05Server",
            "checkpoint": self.engine.checkpoint_path,
            "device": str(self.engine.device),
        }
        await websocket.send(json.dumps(metadata))

        try:
            async for message in websocket:
                try:
                    request = json.loads(message)
                    response = await self._process(request)
                except Exception as e:
                    logger.exception("Error processing request")
                    response = {"status": "error", "error": str(e)}
                await websocket.send(json.dumps(response))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            logger.info("Client disconnected: %s", websocket.remote_address)

    async def _process(self, request: dict) -> dict:
        mode = request.get("mode")  # None = full inference, "extract_latent" = latent only

        images_data = request.get("images", {})
        high_level_prompt = request.get("high_level_prompt", request.get("task", ""))
        state = request.get("state")

        images = {k: _decode_image(v) for k, v in images_data.items()}
        images = _map_image_keys(images)
        state_arr = np.array(state, dtype=np.float32) if state is not None else None

        if mode == "extract_latent":
            result = await self.engine.extract_latent(images, high_level_prompt, state=state_arr)
            return {
                "status": "success",
                "s2_latent": result["s2_latent"].tolist(),
                "timing": result["timing"],
            }
        else:
            result = await self.engine.infer(images, high_level_prompt, state=state_arr)
            return {
                "status": "success",
                "actions": result["actions"],
                "subtask": result.get("subtask", ""),
                "timing": result["timing"],
            }

    async def start(self):
        logger.info("Initializing PyTorch Pi0.5 engine...")
        await self.engine.initialize()
        logger.info("Starting WebSocket server on %s:%d", self.host, self.port)
        try:
            async with websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                ping_interval=60,
                ping_timeout=60,
                max_size=50 * 1024 * 1024,
            ) as server:
                logger.info("Server ready on ws://%s:%d", self.host, self.port)
                await server.wait_closed()
        except asyncio.CancelledError:
            pass


async def main():
    parser = argparse.ArgumentParser(description="PyTorch Pi0.5 WebSocket server")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to converted checkpoint directory or model.safetensors file")
    parser.add_argument("--norm-stats", default=None,
                        help="Path to norm_stats.json for state/action quantile normalization")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    server = PyTorchPi05Server(
        checkpoint_path=args.checkpoint,
        device=args.device,
        host=args.host,
        port=args.port,
        norm_stats_path=args.norm_stats,
    )
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())
