"""
Thread-safe telemetry and latency metrics tracking.
"""

import threading
import time
from collections import deque
import numpy as np

class MetricsTracker:
    def __init__(self, max_history: int = 5000):
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.latencies = deque(maxlen=max_history)

    def record_request(self, latency_ms: float, cached: bool):
        with self._lock:
            self.total_requests += 1
            if cached:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            self.latencies.append(latency_ms)

    def get_stats(self, current_cache_size: int = 0) -> dict:
        with self._lock:
            total = self.total_requests
            hits = self.cache_hits
            misses = self.cache_misses
            hit_rate = (hits / total * 100.0) if total > 0 else 0.0

            if self.latencies:
                arr = list(self.latencies)
                avg_lat = float(np.mean(arr))
                p50 = float(np.percentile(arr, 50))
                p95 = float(np.percentile(arr, 95))
            else:
                avg_lat = p50 = p95 = 0.0

            return {
                "total_requests": total,
                "cache_hits": hits,
                "cache_misses": misses,
                "cache_hit_rate_pct": round(hit_rate, 2),
                "avg_latency_ms": round(avg_lat, 2),
                "p50_latency_ms": round(p50, 2),
                "p95_latency_ms": round(p95, 2),
                "cache_size": current_cache_size,
            }

    @property
    def uptime_seconds(self) -> float:
        return round(time.time() - self.start_time, 2)
