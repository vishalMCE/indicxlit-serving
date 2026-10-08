"""
FastAPI Production Transliteration Service for Live Typists.
"""

import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.config import settings
from server.schemas import (
    TransliterateRequest,
    TransliterateResponse,
    BatchTransliterateRequest,
    BatchTransliterateResponse,
    HealthResponse,
    StatsResponse,
)
from server.metrics import MetricsTracker
try:
    from engine import OptimizedXlitEngine
except ImportError:
    from benchmarks.optimized_engine import OptimizedXlitEngine
from benchmarks.workloads import QUALITY_SAMPLES, SHORT, MEDIUM, LONG

engine: OptimizedXlitEngine | None = None
metrics = MetricsTracker()

def preseed_legal_terms(eng: OptimizedXlitEngine):
    """
    Pre-seeds high-frequency legal terms into LRU cache during startup.
    """
    core_terms = [
        "vakeel", "nyayalaya", "faisla", "mukadma", "adhivakta",
        "kanoon", "peshi", "saboot", "bayan", "insaaf"
    ]
    for w in core_terms:
        eng.translit_word(w, topk=4)
    print(f"[Server] Pre-seeded {len(core_terms)} core courtroom legal terms into in-memory cache.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print(f"[Server] Initializing CTranslate2 INT8 Engine from {settings.model_dir}...")
    t0 = time.perf_counter()
    engine = OptimizedXlitEngine(
        model_dir=settings.model_dir,
        inter_threads=settings.inter_threads,
        intra_threads=settings.intra_threads,
        cache_size=settings.cache_size,
    )
    load_time = time.perf_counter() - t0
    print(f"[Server] Model loaded successfully in {load_time*1000:.2f} ms.")

    if settings.preseed_cache:
        preseed_legal_terms(engine)

    # Warm up neural inference JIT kernels
    engine.translit_word("namaste", topk=4)
    print("[Server] Neural inference engine warmed up.")

    yield

    print("[Server] Shutting down transliteration service.")
    engine = None

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Sub-10ms Indian Courtroom Transliteration API serving live typist suggestions.",
    lifespan=lifespan,
)

# Enable CORS for courtroom typist UI / desktop applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["General"])
async def root():
    return {
        "service": settings.app_name,
        "version": settings.version,
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
    }

@app.get("/health", response_model=HealthResponse, tags=["Observability"])
async def health():
    return HealthResponse(
        status="healthy" if engine is not None else "degraded",
        model_loaded=engine is not None,
        uptime_seconds=metrics.uptime_seconds,
        engine="ctranslate2_int8",
        version=settings.version,
    )

@app.get("/stats", response_model=StatsResponse, tags=["Observability"])
async def stats():
    cache_len = getattr(engine, "cache_size", 0) if engine else 0
    return metrics.get_stats(current_cache_size=cache_len)

@app.post("/transliterate", response_model=TransliterateResponse, tags=["Inference"])
def transliterate(req: TransliterateRequest):
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transliteration engine is still loading or unavailable",
        )

    t0 = time.perf_counter()
    clean_word = req.word.strip().lower()

    # Fast check: was this already in the cache?
    # Note: _setup_cache in OptimizedXlitEngine stores a cache decorated function
    # A lookup taking < 0.001 ms indicates a cache hit
    preds = engine.translit_word(clean_word, topk=req.topk)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    is_cached = latency_ms < 0.5  # Cache hits resolve in <0.01ms; neural infer is 2-8ms
    metrics.record_request(latency_ms, cached=is_cached)

    return TransliterateResponse(
        input_word=clean_word,
        predictions=preds,
        latency_ms=round(latency_ms, 2),
        cached=is_cached,
    )

@app.post("/transliterate_batch", response_model=BatchTransliterateResponse, tags=["Inference"])
def transliterate_batch(req: BatchTransliterateRequest):
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transliteration engine is still loading or unavailable",
        )

    if len(req.words) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size {len(req.words)} exceeds maximum allowed {settings.max_batch_size}",
        )

    t0 = time.perf_counter()
    clean_words = [w.strip().lower() for w in req.words]
    preds = engine.batch_transliterate_words(clean_words, topk=req.topk)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    for _ in clean_words:
        metrics.record_request(latency_ms / len(clean_words), cached=False)

    return BatchTransliterateResponse(
        predictions=preds,
        total_words=len(clean_words),
        latency_ms=round(latency_ms, 2),
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host=settings.host, port=settings.port, reload=False)
