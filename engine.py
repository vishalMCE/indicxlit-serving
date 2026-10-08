"""
IndicXlit inference runtime with INT8 quantization, LRU caching, and baseline support.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Sequence, Union

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _silence_io():
    """
    Silences Python and OS stdout/stderr during third-party library initialization.
    """
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            null_fd = os.open(os.devnull, os.O_RDWR)
            save_stdout = os.dup(1)
            save_stderr = os.dup(2)
            os.dup2(null_fd, 1)
            os.dup2(null_fd, 2)
            try:
                yield
            finally:
                os.dup2(save_stdout, 1)
                os.dup2(save_stderr, 2)
                os.close(null_fd)
                os.close(save_stdout)
                os.close(save_stderr)
        except Exception:
            yield

ROOT = Path(__file__).resolve().parent.parent


def _find_default_model_dir() -> Path:
    if "MODEL_DIR" in os.environ:
        p = Path(os.environ["MODEL_DIR"])
        if (p / "model.bin").is_file():
            return p
    candidates = [
        Path(__file__).resolve().parent / "models" / "ct2_indicxlit_int8",
        Path(__file__).resolve().parent.parent / "models" / "ct2_indicxlit_int8",
        Path("/app/models/ct2_indicxlit_int8"),
        Path("models/ct2_indicxlit_int8"),
    ]
    for c in candidates:
        if (c / "model.bin").is_file():
            return c
    return Path(__file__).resolve().parent / "models" / "ct2_indicxlit_int8"


DEFAULT_MODEL_DIR = _find_default_model_dir()


class OptimizedXlitEngine:
    """
    High-performance CTranslate2 INT8 transliteration engine with optional LRU caching.
    """

    def __init__(
        self,
        model_dir: Path | str = DEFAULT_MODEL_DIR,
        device: str = "cpu",
        inter_threads: int = 4,
        intra_threads: int = 1,
        cache_size: int = 10000,
    ) -> None:
        import ctranslate2

        self.model_dir = Path(model_dir)
        if not (self.model_dir / "model.bin").is_file():
            raise FileNotFoundError(f"Missing CTranslate2 model under {self.model_dir}")

        self.translator = ctranslate2.Translator(
            str(self.model_dir),
            device=device,
            inter_threads=inter_threads,
            intra_threads=intra_threads,
        )
        self.cache_size = cache_size
        self._setup_cache(cache_size)

    def _setup_cache(self, cache_size: int) -> None:
        @lru_cache(maxsize=cache_size)
        def _cached_translit(word: str, topk: int) -> tuple[str, ...]:
            tokens = ["__hi__"] + list(word)
            results = self.translator.translate_batch(
                [tokens],
                beam_size=topk,
                num_hypotheses=topk,
            )
            return tuple("".join(h) for h in results[0].hypotheses)

        self._cached_translit_fn = _cached_translit

    def translit_word(self, word: str, topk: int = 4) -> list[str]:
        """
        Transliterate a single word with optional LRU cache lookup.
        """
        cleaned = word.lower().strip()
        if not cleaned:
            return []
        if self.cache_size > 0:
            return list(self._cached_translit_fn(cleaned, topk))

        tokens = ["__hi__"] + list(cleaned)
        results = self.translator.translate_batch(
            [tokens],
            beam_size=topk,
            num_hypotheses=topk,
        )
        return ["".join(h) for h in results[0].hypotheses]

    def batch_transliterate_words(
        self,
        words: Sequence[str],
        topk: int = 4,
    ) -> list[list[str]]:
        """
        Transliterate a batch of words directly via vectorized C++ execution.
        """
        token_batches = [["__hi__"] + list(w.lower().strip()) for w in words]
        results = self.translator.translate_batch(
            token_batches,
            beam_size=topk,
            num_hypotheses=topk,
        )
        return [["".join(h) for h in r.hypotheses] for r in results]


class StockFairseqEngine:
    """
    Wrapper around baseline PyTorch Fairseq FP32 engine.
    """

    @staticmethod
    def _link_local_model_if_available() -> None:
        local_dir = ROOT / "IndicXlit" / "indicxlit-en-indic-v1.0"
        if not (local_dir / "transformer" / "indicxlit.pt").is_file():
            return
        try:
            import ai4bharat.transliteration.transformer as transformer_pkg
            pkg_root = Path(transformer_pkg.__file__).resolve().parent
            dest = pkg_root / "models" / "en2indic" / "v1.0"
            dest.mkdir(parents=True, exist_ok=True)
            for name in ("transformer", "corpus-bin"):
                target = dest / name
                source = local_dir / name
                if target.is_symlink() or target.exists():
                    if target.is_symlink() and target.resolve() == source.resolve():
                        continue
                    if target.is_dir() and not target.is_symlink():
                        continue
                    target.unlink()
                target.symlink_to(source)
        except Exception:
            pass

    def __init__(self, lang: str = "hi", beam_width: int = 4, rescore: bool = False) -> None:
        try:
            self._link_local_model_if_available()
            with _silence_io():
                from ai4bharat.transliteration import XlitEngine
                self.engine = XlitEngine(lang, beam_width=beam_width, rescore=rescore)
        except Exception as exc:
            raise RuntimeError(f"Stock Fairseq engine unavailable: {exc}") from exc

    def translit_word(self, word: str, topk: int = 4) -> list[str]:
        """
        Transliterate a single word via Stock Fairseq inference.
        """
        cleaned = word.lower().strip()
        if not cleaned:
            return []
        res = self.engine.translit_word(cleaned, topk=topk)
        if isinstance(res, dict):
            return res.get("hi", [])
        return res

    def batch_transliterate_words(
        self,
        words: Sequence[str],
        topk: int = 4,
    ) -> list[list[str]]:
        """
        Transliterate a batch of words via sequential Fairseq calls.
        """
        return [self.translit_word(w, topk=topk) for w in words]


def get_engine(
    backend: str = "ct2",
    model_dir: Path | str | None = None,
    cache_size: int = 10000,
    device: str = "cpu",
    inter_threads: int = 4,
    intra_threads: int = 1,
    beam_width: int = 4,
) -> Union[OptimizedXlitEngine, StockFairseqEngine]:
    """
    Factory to instantiate the requested transliteration backend.
    """
    b = backend.lower().strip()
    target_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR

    if b in ("ct2", "optimized"):
        return OptimizedXlitEngine(
            model_dir=target_dir,
            device=device,
            inter_threads=inter_threads,
            intra_threads=intra_threads,
            cache_size=cache_size,
        )
    if b == "cache":
        return OptimizedXlitEngine(
            model_dir=target_dir,
            device=device,
            inter_threads=inter_threads,
            intra_threads=intra_threads,
            cache_size=max(cache_size, 20000),
        )
    if b in ("stock", "baseline"):
        return StockFairseqEngine(beam_width=beam_width)

    raise ValueError(f"Unknown backend '{backend}'. Choose from: 'ct2', 'cache', 'stock'")
