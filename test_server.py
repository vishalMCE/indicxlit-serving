"""
Smoke tests for the FastAPI transliteration service.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.app import app

def test_endpoints():
    with TestClient(app) as client:
        # 1. Health check
        print("Testing GET /health...")
        res = client.get("/health")
        assert res.status_code == 200, f"Health check failed: {res.text}"
        health_data = res.json()
        print("Health status:", health_data)
        assert health_data["status"] == "healthy"
        assert health_data["model_loaded"] is True

        # 2. Single word transliteration
        print("\nTesting POST /transliterate (single word)...")
        res = client.post("/transliterate", json={"word": "nyayalaya", "topk": 4})
        assert res.status_code == 200, f"Transliterate failed: {res.text}"
        data = res.json()
        print("Transliterate response:", data)
        assert data["input_word"] == "nyayalaya"
        assert "न्यायालय" in data["predictions"]
        assert data["latency_ms"] < 100.0  # Must be well under 100ms SLA

        # 3. Batch transliteration
        print("\nTesting POST /transliterate_batch (multiple words)...")
        test_words = ["dastavej", "kanoon", "vakeel", "faisla"]
        res = client.post("/transliterate_batch", json={"words": test_words, "topk": 4})
        assert res.status_code == 200, f"Batch transliterate failed: {res.text}"
        batch_data = res.json()
        print(f"Batch response ({batch_data['total_words']} words, {batch_data['latency_ms']} ms):", batch_data["predictions"])
        assert batch_data["total_words"] == len(test_words)
        assert len(batch_data["predictions"]) == len(test_words)

        # 4. Telemetry stats
        print("\nTesting GET /stats...")
        res = client.get("/stats")
        assert res.status_code == 200
        stats_data = res.json()
        print("Stats response:", stats_data)
        assert stats_data["total_requests"] > 0

    print("\n[SUCCESS] All FastAPI endpoints verified successfully!")

if __name__ == "__main__":
    test_endpoints()
