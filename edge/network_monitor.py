"""
Network monitor: measures real-time bandwidth and RTT to the cloud.
Used by the edge node to decide whether to use cloud or local inference.
"""
import asyncio
import time
import logging
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# Thresholds for adaptive scheduling
MIN_BANDWIDTH_MBPS  = 5.0    # below this → use local inference
MAX_LATENCY_MS      = 500.0  # above this → use local inference
PROBE_PAYLOAD_BYTES = 65536  # 64 KB probe


class NetworkMonitor:
    def __init__(self, cloud_url: str):
        self.cloud_url = cloud_url
        self._last_bw: float = 100.0
        self._last_latency: float = 10.0
        self._history: list[dict] = []

    async def measure(self) -> tuple[float, float]:
        """
        Probe the cloud endpoint and return (bandwidth_mbps, latency_ms).
        Uses a small ping + timed payload upload.
        """
        t0 = time.perf_counter()
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                # RTT probe: GET /health
                t_ping = time.perf_counter()
                async with session.get(f"{self.cloud_url}/health") as r:
                    await r.text()
                latency_ms = (time.perf_counter() - t_ping) * 1000

                # Bandwidth probe: upload a fixed payload
                payload = bytes(PROBE_PAYLOAD_BYTES)
                t_bw = time.perf_counter()
                async with session.post(
                    f"{self.cloud_url}/health",
                    data=payload,
                    headers={"Content-Type": "application/octet-stream"},
                ) as r:
                    await r.read()
                elapsed_bw = max(time.perf_counter() - t_bw, 1e-6)
                bandwidth_mbps = (PROBE_PAYLOAD_BYTES * 8) / (elapsed_bw * 1e6)

        except Exception as e:
            logger.warning(f"Network probe failed: {e}")
            latency_ms  = 9999.0
            bandwidth_mbps = 0.0

        self._last_bw      = bandwidth_mbps
        self._last_latency = latency_ms
        self._history.append({
            "timestamp": time.time(),
            "bandwidth_mbps": bandwidth_mbps,
            "latency_ms": latency_ms,
        })
        if len(self._history) > 1000:
            self._history = self._history[-500:]

        logger.debug(f"Network: bw={bandwidth_mbps:.1f}Mbps lat={latency_ms:.1f}ms")
        return bandwidth_mbps, latency_ms

    def should_use_cloud(self) -> bool:
        """Adaptive scheduling decision."""
        return (
            self._last_bw >= MIN_BANDWIDTH_MBPS
            and self._last_latency <= MAX_LATENCY_MS
        )

    def stats(self) -> dict:
        if not self._history:
            return {}
        bws = [h["bandwidth_mbps"] for h in self._history]
        lats = [h["latency_ms"] for h in self._history]
        return {
            "avg_bandwidth_mbps": sum(bws) / len(bws),
            "avg_latency_ms":     sum(lats) / len(lats),
            "min_latency_ms":     min(lats),
            "max_latency_ms":     max(lats),
            "n_probes":           len(self._history),
        }
