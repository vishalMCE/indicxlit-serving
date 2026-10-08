"""
Pydantic request and response schemas for Transliteration Server.
"""

from typing import List
from pydantic import BaseModel, Field

class TransliterateRequest(BaseModel):
    word: str = Field(..., min_length=1, max_length=100, description="Romanized input word")
    topk: int = Field(default=4, ge=1, le=10, description="Number of suggestions to return")
    lang: str = Field(default="hi", description="Target Indic language code")

class TransliterateResponse(BaseModel):
    input_word: str
    predictions: List[str]
    latency_ms: float
    cached: bool

class BatchTransliterateRequest(BaseModel):
    words: List[str] = Field(..., min_items=1, max_items=256, description="List of romanized words")
    topk: int = Field(default=4, ge=1, le=10, description="Number of suggestions per word")
    lang: str = Field(default="hi", description="Target Indic language code")

class BatchTransliterateResponse(BaseModel):
    predictions: List[List[str]]
    total_words: int
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    engine: str = "ctranslate2_int8"
    version: str

class StatsResponse(BaseModel):
    total_requests: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate_pct: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    cache_size: int
