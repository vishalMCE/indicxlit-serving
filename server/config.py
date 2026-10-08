"""
Configuration settings for IndicXlit Transliteration Server.
"""

import os
from pathlib import Path
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parent.parent


def _find_model_dir() -> str:
    if "MODEL_DIR" in os.environ:
        return os.environ["MODEL_DIR"]
    candidates = [
        ROOT_DIR / "models" / "ct2_indicxlit_int8",
        ROOT_DIR.parent / "models" / "ct2_indicxlit_int8",
        Path("/app/models/ct2_indicxlit_int8"),
        Path("models/ct2_indicxlit_int8"),
    ]
    for c in candidates:
        if (c / "model.bin").is_file():
            return str(c)
    return str(ROOT_DIR / "models" / "ct2_indicxlit_int8")


class ServerConfig(BaseModel):
    app_name: str = "IndicXlit Transliteration Serving Engine"
    version: str = "1.0.0"
    model_dir: str = os.getenv("MODEL_DIR", _find_model_dir())
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    inter_threads: int = int(os.getenv("INTER_THREADS", "4"))
    intra_threads: int = int(os.getenv("INTRA_THREADS", "1"))
    cache_size: int = int(os.getenv("CACHE_SIZE", "25000"))
    preseed_cache: bool = os.getenv("PRESEED_CACHE", "true").lower() in ("true", "1", "yes")
    default_lang: str = "hi"
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "256"))

settings = ServerConfig()
