"""
Edge Node Service
=================
Runs on the hospital edge device (e.g. NVIDIA Jetson / PACS workstation).

Responsibilities:
  1. Accept CT scan patches (from scanner / DICOM PACS)
  2. Decode a model-ready windowed RGB patch (not raw HU/DICOM reconstruction)
  3. Early-layer feature extraction (layer1 + layer2 of ResNet-50)
  4. Feature compression (none or INT8; learned codec is not wired into live inference)
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
COMPRESSION     = os.getenv("COMPRESSION",      "int8")         # live: none | int8
PATCH_SIZE      = int(os.getenv("PATCH_SIZE",   "64"))
BATCH_SIZE      = int(os.getenv("BATCH_SIZE",   "1"))
HEARTBEAT_SEC   = int(os.getenv("HEARTBEAT_SEC","30"))
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Differential privacy: set DP_EPSILON > 0 to add Gaussian noise to features
# before compression. Set to 0 (default) to disable live DP noise injection.
_DP_EPSILON_STR = os.getenv("DP_EPSILON", "0").strip()
DP_EPSILON      = float(_DP_EPSILON_STR) if _DP_EPSILON_STR else 0.0
DP_DELTA        = float(os.getenv("DP_DELTA", "1e-5"))
DP_SENSITIVITY  = float(os.getenv("DP_SENSITIVITY", "1.0"))
PROJECT_ROOT    = Path(__file__).resolve().parents[1]
EDGE_MODEL_PATH = Path(
    os.getenv("EDGE_MODEL_PATH")
    or os.getenv("MODEL_PATH")
    or (PROJECT_ROOT / "experiments" / "best_model.pth")
).expanduser()
if not EDGE_MODEL_PATH.is_absolute():
    EDGE_MODEL_PATH = (PROJECT_ROOT / EDGE_MODEL_PATH).resolve()


class _DPAccountant:
    """
    Per-node privacy accountant using the moments/Rényi composition rule.

    Each call to ``record_release`` accumulates privacy spend.  When the
    total exceeds the configured budget (DP_EPSILON_BUDGET), the accountant
    returns False from ``can_release`` and the edge node refuses to send
    more features until reset (e.g. at the start of a new patient session).

    This is a simplified RDP accountant (alpha=10 Rényi order); a full
    implementation would use google/dp-accounting or autodp.
    """

    _RENYI_ALPHA = 10  # Rényi order

    def __init__(self, epsilon_budget: float, delta: float):
        self.epsilon_budget = epsilon_budget
        self.delta = delta
        self._epsilon_spent: float = 0.0
        self._n_releases: int = 0

    def record_release(self, sigma: float, sensitivity: float) -> None:
        """Add one Gaussian mechanism release to the privacy ledger."""
        # RDP ε(α) for Gaussian: α * sensitivity² / (2σ²)
        rdp = self._RENYI_ALPHA * (sensitivity ** 2) / (2.0 * sigma ** 2)
        # Convert RDP → (ε, δ)-DP via standard bound
        eps_dp = rdp + (
            np.log(1.0 / self.delta) / (self._RENYI_ALPHA - 1)
        )
        self._epsilon_spent += eps_dp
        self._n_releases += 1

    def can_release(self) -> bool:
        return self._epsilon_spent < self.epsilon_budget

    def reset(self) -> None:
        self._epsilon_spent = 0.0
        self._n_releases = 0

    @property
    def epsilon_spent(self) -> float:
        return self._epsilon_spent

    @property
    def n_releases(self) -> int:
        return self._n_releases


# Module-level accountant; shared across all scans in a session.
_dp_accountant: "_DPAccountant | None" = None
if DP_EPSILON > 0.0:
    _dp_budget = float(os.getenv("DP_EPSILON_BUDGET", str(DP_EPSILON * 100)))
    _dp_accountant = _DPAccountant(epsilon_budget=_dp_budget, delta=DP_DELTA)


def _add_dp_noise(features: dict, epsilon: float, delta: float = 1e-5, sensitivity: float = 1.0) -> dict:
    """
    Inject calibrated Gaussian noise into feature tensors for (ε, δ)-DP.

    Steps:
      1. L2-clip each feature tensor to bound the sensitivity (per-sample).
      2. Add Gaussian noise scaled to the clipped sensitivity.
      3. Record the release in the module-level privacy accountant.

    σ = (sensitivity / epsilon) * sqrt(2 * ln(1.25 / delta))

    The clipping norm equals ``sensitivity``, which the caller supplies via
    DP_SENSITIVITY.  Set this to the empirical 95th-percentile L2 norm of
    your feature tensors for a tight but safe bound.
    """
    if _dp_accountant is not None and not _dp_accountant.can_release():
        raise RuntimeError(
            f"DP budget exhausted after {_dp_accountant.n_releases} releases "
            f"(ε_spent={_dp_accountant.epsilon_spent:.4f} ≥ budget={_dp_accountant.epsilon_budget:.4f}). "
            "Call _dp_accountant.reset() to start a new privacy epoch."
        )

    sigma = (sensitivity / epsilon) * (2.0 * np.log(1.25 / delta)) ** 0.5
    noised = {}
    for k, v in features.items():
        fp = v.dequantize().float() if v.is_quantized else v.float()
        # Per-sample L2 clipping: scale each tensor so its L2 norm ≤ sensitivity
        l2_norm = fp.norm(p=2)
        fp = fp * (sensitivity / (l2_norm + 1e-8)).clamp(max=1.0)
        noised[k] = fp + torch.randn_like(fp) * sigma

    if _dp_accountant is not None:
        _dp_accountant.record_release(sigma=sigma, sensitivity=sensitivity)

    return noised


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


def _wire_compression_ratio(features: dict, payload_bytes: bytes) -> float:
    """Return float32-baseline bytes divided by the complete serialized body."""
    raw_float32_bytes = sum(value.numel() for value in features.values()) * 4
    return raw_float32_bytes / max(len(payload_bytes), 1)


class EdgeNode:
    """Main edge node controller."""

    def __init__(self):
        if DP_EPSILON > 0.0:
            raise ValueError(
                "Live DP feature release is disabled: the experimental clipping/noise "
                "and accountant have not been validated as an end-to-end "
                "(epsilon, delta)-DP mechanism. Use scripts/cn5_differential_privacy.py "
                "for offline research only."
            )
        if COMPRESSION == "learned":
            raise ValueError(
                "Learned compression is not supported by the live edge/cloud protocol: "
                "the cloud decoder and rate-distortion-trained checkpoint integration "
                "are not implemented. Use none or int8."
            )
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
            # Record the scheduling decision so the dashboard can show fallback frequency.
            await self._log_network_event({
                "node_id": NODE_ID,
                "latency_ms": latency_ms,
                "bandwidth_mbps": bw_mbps,
                "payload_size_kb": 0.0,
                "inference_time_ms": 0.0,
                "split_layer": SPLIT_LAYER,
                "compression_method": "local_fallback",
                "compression_ratio": 0.0,
            })
            result = await self.predict_local(image_bytes, scan_uid)
            result["adaptive_decision"] = "local_fallback"
            result["probe_bandwidth_mbps"] = bw_mbps
            result["probe_latency_ms"] = latency_ms
            return result

        # Extract and optionally apply live DP noise before compression.
        with torch.no_grad():
            features = self.extractor.extract(patch_tensor.unsqueeze(0).to(DEVICE))

        if DP_EPSILON > 0.0:
            features = _add_dp_noise(
                features, epsilon=DP_EPSILON, delta=DP_DELTA, sensitivity=DP_SENSITIVITY
            )
            logger.info(
                "[%s] DP noise applied (ε=%.4f, δ=%g, sensitivity=%.3f) | "
                "accountant: ε_spent=%.4f, releases=%d",
                scan_uid, DP_EPSILON, DP_DELTA, DP_SENSITIVITY,
                _dp_accountant.epsilon_spent if _dp_accountant else 0.0,
                _dp_accountant.n_releases if _dp_accountant else 1,
            )

        compressed_payload = self.compressor.compress(features)
        t_compress = time.perf_counter()

        payload_bytes = _serialize_feature_payload(
            compressed_payload,
            {
                "scan_uid": scan_uid,
                "edge_processing_ms": (t_compress - t_start) * 1000,
                "bandwidth_mbps": bw_mbps,
                "compression_method": COMPRESSION,
                "dp_epsilon": DP_EPSILON if DP_EPSILON > 0.0 else None,
                "dp_epsilon_spent": _dp_accountant.epsilon_spent if _dp_accountant else None,
                "dp_releases": _dp_accountant.n_releases if _dp_accountant else None,
                "checkpoint_sha256": self.extractor.checkpoint_sha256,
                "image_mean": self.extractor.image_mean,
                "image_std": self.extractor.image_std,
            },
        )
        payload_kb = len(payload_bytes) / 1024

        # Use the complete serialized request body, including its JSON header.
        compression_ratio = _wire_compression_ratio(features, payload_bytes)

        try:
            result = await self._send_features(payload_bytes, scan_uid)
            result.setdefault("adaptive_decision", "cloud")
        except Exception as exc:
            logger.warning("[%s] Cloud feature inference failed; using local fallback: %s", scan_uid, exc)
            result = await self.predict_local(image_bytes, scan_uid)
            result["fallback_reason"] = str(exc)
            result["adaptive_decision"] = "local_fallback_cloud_error"

        result["probe_bandwidth_mbps"] = bw_mbps
        result["probe_latency_ms"] = latency_ms

        total_ms = (time.perf_counter() - t_start) * 1000

        # 6. Log to cloud — use wire-verified compression_ratio
        await self._log_network_event({
            "node_id": NODE_ID,
            "latency_ms": latency_ms,
            "bandwidth_mbps": bw_mbps,
            "payload_size_kb": payload_kb,
            "inference_time_ms": result.get("cloud_inference_ms", 0),
            "split_layer": SPLIT_LAYER,
            "compression_method": (
                "local_fallback_cloud_error"
                if result.get("adaptive_decision") == "local_fallback_cloud_error"
                else COMPRESSION
            ),
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
