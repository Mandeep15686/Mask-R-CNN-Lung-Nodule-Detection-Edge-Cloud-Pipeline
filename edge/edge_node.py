"""
Edge Node Service
=================
Runs on the hospital edge device (e.g. NVIDIA Jetson / PACS workstation).

Responsibilities:
  1. Accept CT scan patches (from scanner / DICOM PACS)
  2. HU windowing + normalisation (preprocessor)
  3. Early-layer feature extraction (layer1 + layer2 of ResNet-50)
  4. Feature compression (INT8 quantisation or learned bottleneck)
  5. Transmit compressed features to cloud API
  6. Receive + cache nodule detection results
  7. Heartbeat + network monitoring
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np
import torch
from PIL import Image

from preprocessor import CTPreprocessor
from feature_extractor import EdgeFeatureExtractor
from compressor import FeatureCompressor
from network_monitor import NetworkMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EDGE] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CLOUD_API_URL   = os.getenv("CLOUD_API_URL",  "http://localhost:8000")
NODE_ID         = os.getenv("NODE_ID",         "edge-node-01")
HOSTNAME        = os.getenv("HOSTNAME",         "edge-01.hospital.local")
NODE_IP         = os.getenv("NODE_IP",          "192.168.1.10")
EDGE_API_TOKEN  = os.getenv("EDGE_API_TOKEN",    "")
SPLIT_LAYER     = os.getenv("SPLIT_LAYER",      "layer2")       # layer1 | layer2 | layer3
COMPRESSION     = os.getenv("COMPRESSION",      "int8")         # none | int8 | learned
PATCH_SIZE      = int(os.getenv("PATCH_SIZE",   "64"))
BATCH_SIZE      = int(os.getenv("BATCH_SIZE",   "1"))
HEARTBEAT_SEC   = int(os.getenv("HEARTBEAT_SEC","30"))
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT    = Path(__file__).resolve().parents[1]
EDGE_MODEL_PATH = Path(
    os.getenv("EDGE_MODEL_PATH")
    or os.getenv("MODEL_PATH")
    or (PROJECT_ROOT / "experiments" / "best_model.pth")
).expanduser()
if not EDGE_MODEL_PATH.is_absolute():
    EDGE_MODEL_PATH = (PROJECT_ROOT / EDGE_MODEL_PATH).resolve()


def _serialize_feature_payload(features: dict, metadata: dict) -> bytes:
    """
    Encode named features as a JSON header followed by packed binary data.

    Version 2 wire format:
    - INT8/QINT8 tensors are serialised as raw int8 bytes with per-tensor
      scale and zero_point stored in the header.  This achieves the intended
      ~4× bandwidth reduction over float32.
    - Float tensors are serialised as little-endian float32 (unchanged).

    Prior to this fix the serialiser dequantised INT8 tensors immediately and
    wrote them as float32, negating all compression benefit.
    """
    keys = list(features)
    shapes = []
    dtypes = []
    scales = []
    zero_points = []
    arrays = []

    for key in keys:
        value = features[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Feature {key!r} must be a tensor")

        if value.is_quantized:
            # Serialise as int8 — actual 4× wire savings vs float32
            int_repr = value.int_repr()          # torch.int8 tensor
            q_scale  = float(value.q_scale())
            q_zp     = int(value.q_zero_point())
            array    = int_repr.detach().cpu().contiguous().numpy()
            shapes.append(list(array.shape))
            dtypes.append("int8")
            scales.append(q_scale)
            zero_points.append(q_zp)
            arrays.append(array.tobytes())
        else:
            array = value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
            if not np.isfinite(array).all():
                raise ValueError(f"Feature {key!r} contains non-finite values")
            shapes.append(list(array.shape))
            dtypes.append("float32")
            scales.append(1.0)
            zero_points.append(0)
            arrays.append(array.astype("<f4", copy=False).tobytes(order="C"))

    header = {
        **metadata,
        "version": 2,
        "keys": keys,
        "shapes": shapes,
        "dtypes": dtypes,
        "scales": scales,
        "zero_points": zero_points,
    }
    return (
        json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8")
        + b"\x00"
        + b"".join(arrays)
    )


class EdgeNode:
    """Main edge node controller."""

    def __init__(self):
        if COMPRESSION == "learned" and SPLIT_LAYER != "layer2":
            raise ValueError("Learned compression checkpoint currently supports only layer2 features")
        self.preprocessor  = CTPreprocessor(patch_size=PATCH_SIZE)
        self.extractor     = EdgeFeatureExtractor(
            split_layer=SPLIT_LAYER,
            device=DEVICE,
            checkpoint_path=EDGE_MODEL_PATH,
        )
        self.compressor    = FeatureCompressor(method=COMPRESSION, device=DEVICE)
        self.net_monitor   = NetworkMonitor(cloud_url=CLOUD_API_URL)
        self.session: Optional[aiohttp.ClientSession] = None
        self._running      = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self.result_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    async def start(self):
        if self.session is not None and not self.session.closed:
            return
        logger.info(f"Starting Edge Node [{NODE_ID}] | split={SPLIT_LAYER} | compress={COMPRESSION}")
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        self._running = True
        await self._register()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("Edge node ready. Waiting for scan submissions...")

    async def stop(self):
        self._running = False
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("Edge node stopped.")

    # ------------------------------------------------------------------
    async def _register(self):
        if self.session is None or self.session.closed:
            return
        try:
            async with self.session.post(
                f"{CLOUD_API_URL}/api/edge/nodes/register",
                json={
                    "node_id": NODE_ID,
                    "hostname": HOSTNAME,
                    "ip_address": NODE_IP,
                    "split_layer": SPLIT_LAYER,
                    "compression_method": COMPRESSION,
                },
            ) as resp:
                if resp.status in (200, 201):
                    logger.info("Registered with cloud successfully")
                else:
                    logger.warning(f"Registration returned HTTP {resp.status}")
        except Exception as e:
            logger.warning(f"Could not register (cloud offline?): {e}")

    async def _heartbeat_loop(self):
        while self._running and self.session is not None and not self.session.closed:
            try:
                async with self.session.post(
                    f"{CLOUD_API_URL}/api/edge/nodes/{NODE_ID}/heartbeat"
                ):
                    pass
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Edge heartbeat failed", exc_info=True)
            await asyncio.sleep(HEARTBEAT_SEC)

    # ------------------------------------------------------------------
    async def process_scan(
        self,
        image_bytes: bytes,
        scan_uid: str,
        source_format: str = "png",
    ) -> dict:
        """
        Full edge-cloud pipeline for one CT patch.
        Returns detection results dict.
        """
        t_start = time.perf_counter()

        # Preprocess once for the local fallback and, if the link is healthy,
        # for edge feature extraction.
        patch_tensor = self.preprocessor.preprocess_bytes(image_bytes, source_format)

        # Probe before spending time extracting/compressing features. A failed
        # or poor link should select local fallback instead of a doomed upload.
        bw_mbps, latency_ms = await self.net_monitor.measure()
        if not self.net_monitor.should_use_cloud():
            logger.info(
                "[%s] Network below cloud policy (%.2f Mbps, %.1f ms); using local fallback",
                scan_uid,
                bw_mbps,
                latency_ms,
            )
            return await self.predict_local(image_bytes, scan_uid)

        # Extract and compress only when the cloud path is viable.
        with torch.no_grad():
            features = self.extractor.extract(patch_tensor.unsqueeze(0).to(DEVICE))
        compressed_payload = self.compressor.compress(features)
        t_compress = time.perf_counter()

        raw_feature_bytes = sum(v.numel() * v.element_size() for v in features.values())
        wire_feature_bytes = sum(
            v.numel() * v.element_size() for v in compressed_payload.values()
        )
        compression_ratio = raw_feature_bytes / max(wire_feature_bytes, 1)

        payload_bytes = _serialize_feature_payload(
            compressed_payload,
            {
                "scan_uid": scan_uid,
                "edge_processing_ms": (t_compress - t_start) * 1000,
                "bandwidth_mbps": bw_mbps,
                "compression_method": COMPRESSION,
                "compression_ratio": compression_ratio,
                "checkpoint_sha256": self.extractor.checkpoint_sha256,
                "image_mean": self.extractor.image_mean,
                "image_std": self.extractor.image_std,
            },
        )
        payload_kb = len(payload_bytes) / 1024

        try:
            result = await self._send_features(payload_bytes, scan_uid)
        except Exception as exc:
            logger.warning("[%s] Cloud feature inference failed; using local fallback: %s", scan_uid, exc)
            result = await self.predict_local(image_bytes, scan_uid)
            result["fallback_reason"] = str(exc)

        total_ms = (time.perf_counter() - t_start) * 1000

        # 6. Log to cloud
        await self._log_network_event({
            "node_id": NODE_ID,
            "latency_ms": latency_ms,
            "bandwidth_mbps": bw_mbps,
            "payload_size_kb": payload_kb,
            "inference_time_ms": result.get("cloud_inference_ms", 0),
            "split_layer": SPLIT_LAYER,
            "compression_method": COMPRESSION,
            "compression_ratio": compression_ratio,
        })

        logger.info(
            f"[{scan_uid}] edge={((t_compress-t_start)*1000):.1f}ms | "
            f"payload={payload_kb:.1f}KB (ratio={compression_ratio:.1f}x) | "
            f"cloud={result.get('cloud_inference_ms',0):.1f}ms | "
            f"total={total_ms:.1f}ms"
        )

        result["total_pipeline_ms"] = total_ms
        result["edge_processing_ms"] = (t_compress - t_start) * 1000
        result["compression_ratio"] = compression_ratio
        self.result_cache[scan_uid] = result
        return result

    async def _send_features(self, payload_bytes: bytes, scan_uid: str) -> dict:
        if self.session is None or self.session.closed:
            raise RuntimeError("Edge HTTP session is not started")
        try:
            async with self.session.post(
                f"{CLOUD_API_URL}/api/edge/infer-features",
                data=payload_bytes,
                params={"node_id": NODE_ID, "scan_uid": scan_uid, "split_layer": SPLIT_LAYER},
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Edge-Token": EDGE_API_TOKEN,
                },
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    text = await resp.text()
                    logger.error(f"Cloud inference failed: HTTP {resp.status}: {text}")
                    raise RuntimeError(f"Cloud inference failed with HTTP {resp.status}: {text}")
        except aiohttp.ClientError as e:
            logger.error(f"Network error sending features: {e}")
            raise RuntimeError(f"Network error sending features: {e}") from e

    async def _log_network_event(self, data: dict):
        if self.session is None or self.session.closed:
            return
        try:
            async with self.session.post(
                f"{CLOUD_API_URL}/api/edge/network-log", json=data
            ):
                pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Fallback: full local inference when cloud is unreachable
    # ------------------------------------------------------------------
    async def predict_local(self, image_bytes: bytes, scan_uid: str) -> dict:
        """
        Run lightweight detection locally (edge-only mode).
        Uses a MobileNetV3-based detector — much smaller than full Mask R-CNN.
        """
        logger.info(f"[{scan_uid}] Running local edge-only inference (cloud unreachable)")
        patch_tensor = self.preprocessor.preprocess_bytes(image_bytes)
        # Simple thresholding-based candidate detection as fallback
        patch_np = patch_tensor[1].cpu().numpy()  # middle slice
        from scipy import ndimage
        binary = patch_np > 0.6
        labeled, n = ndimage.label(binary)
        nodules = []
        slices = ndimage.find_objects(labeled)
        for i in range(1, n + 1):
            region = slices[i - 1]
            if region is None:
                continue
            y_sl, x_sl = region
            nodules.append({
                "box": [x_sl.start, y_sl.start, x_sl.stop, y_sl.stop],
                "score": 0.5,
                "source": "edge_fallback",
                "diameter_estimate_mm": ((x_sl.stop - x_sl.start) + (y_sl.stop - y_sl.start)) / 2 * 0.7,
            })
        return {"scan_uid": scan_uid, "nodules": nodules[:5], "source": "edge_fallback"}


# ---------------------------------------------------------------------------
# Simple HTTP server for receiving scan submissions
# ---------------------------------------------------------------------------
async def _http_server(edge_node: EdgeNode, host: str = "0.0.0.0", port: int = 9090):
    """Minimal aiohttp HTTP server to accept scan submissions from PACS."""
    from aiohttp import web

    async def handle_scan(request: web.Request) -> web.Response:
        scan_uid = request.query.get("scan_uid", f"scan_{int(time.time())}")
        image_bytes = await request.read()
        try:
            result = await edge_node.process_scan(image_bytes, scan_uid)
        except Exception as e:
            # Fallback to local
            logger.warning(f"Cloud unavailable, using local: {e}")
            result = await edge_node.predict_local(image_bytes, scan_uid)
        return web.json_response(result)

    async def handle_health(request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "node_id": NODE_ID,
            "split_layer": SPLIT_LAYER,
            "compression": COMPRESSION,
        })

    app = web.Application()
    app.router.add_post("/scan", handle_scan)
    app.router.add_get("/health", handle_health)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info(f"Edge HTTP server listening on {host}:{port}")
    return runner


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main():
    edge = EdgeNode()
    await edge.start()
    runner = await _http_server(edge)
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down...")
        await edge.stop()
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
