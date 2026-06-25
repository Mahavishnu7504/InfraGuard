import logging
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3] / "models"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", _DEFAULT_MODEL_DIR))


# ---------------------------------------------------------------------------
# Registry definition
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    """Descriptor for a registered model."""
    key: str
    filename: str
    description: str
    input_size: tuple[int, int] = (640, 640)   # (height, width)
    _model: Optional[YOLO] = field(default=None, repr=False)
    _metadata: Optional[dict] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path(self) -> Path:
        path = MODEL_DIR / self.filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                f"Set the MODEL_DIR environment variable to the folder that "
                f"contains your .pt files."
            )
        return path

    # ------------------------------------------------------------------
    # Loading & warm-up
    # ------------------------------------------------------------------

    def load(self) -> YOLO:
        """Load the model (idempotent — returns cached instance on repeat calls)."""
        if self._model is not None:
            return self._model

        path = self._path()
        logger.info("[%s] Loading model from %s", self.key, path)
        t0 = time.perf_counter()
        model = YOLO(str(path))
        elapsed = time.perf_counter() - t0
        logger.info("[%s] Model loaded in %.2f s", self.key, elapsed)

        self._model = model
        self._build_metadata(elapsed)
        self._warmup()
        return self._model

    def _warmup(self) -> None:
        """Run one dummy inference to pre-compile CUDA kernels / load weights into RAM."""
        if self._model is None:
            return
        h, w = self.input_size
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        logger.info("[%s] Running warm-up inference …", self.key)
        try:
            self._model.predict(dummy, verbose=False)
            logger.info("[%s] Warm-up complete", self.key)
        except Exception as exc:                        # never crash the server
            logger.warning("[%s] Warm-up failed (non-fatal): %s", self.key, exc)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _build_metadata(self, load_time_s: float) -> None:
        model = self._model
        device = next(model.model.parameters()).device if model else "unknown"
        class_names: list[str] = []
        if hasattr(model, "names") and isinstance(model.names, dict):
            class_names = [model.names[i] for i in sorted(model.names)]

        self._metadata = {
            "name": self.key,
            "filename": self.filename,
            "description": self.description,
            "version": getattr(model, "ckpt", {}).get("version", "n/a")
                        if hasattr(model, "ckpt") and model.ckpt else "n/a",
            "classes": class_names,
            "num_classes": len(class_names),
            "input_size": list(self.input_size),
            "device": str(device),
            "loaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "load_time_s": round(load_time_s, 3),
        }

        # Keep the public MODEL_INFO dict in sync so the dashboard always has
        # fresh data without calling any function.
        MODEL_INFO[self.key] = {
            "status":      "online",
            "version":     self._metadata["version"],
            "classes":     class_names,
            "num_classes": len(class_names),
            "device":      "GPU" if str(device).startswith("cuda") else "CPU",
            "loaded_time": self._metadata["loaded_at"],
        }

    @property
    def metadata(self) -> dict:
        """Return model metadata; loads the model first if not already loaded."""
        if self._metadata is None:
            self.load()
        return dict(self._metadata)   # shallow copy — callers shouldn't mutate

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def health(self) -> dict:
        """Return a per-model health snapshot."""
        status = "online" if self.is_loaded else "not_loaded"
        result: dict = {"model": self.key, "status": status}
        if self.is_loaded and self._metadata:
            result["device"] = self._metadata.get("device", "unknown")
            result["loaded_at"] = self._metadata.get("loaded_at")
        return result


# ---------------------------------------------------------------------------
# MODEL_INFO — live dashboard dict, populated automatically on first load
# ---------------------------------------------------------------------------
#
# Shape (once a model is loaded):
#
#   MODEL_INFO = {
#       "infraguard": {
#           "status":      "online",       # "online" | "not_loaded"
#           "version":     "8.0.0",
#           "classes":     ["hardhat", "vest", ...],
#           "num_classes": 21,
#           "device":      "CPU" | "GPU",  # human-readable label
#           "loaded_time": "2024-06-25T10:42:01",
#       },
#       "crack": { ... },
#   }
#
# Before a model is loaded the entry holds sentinel values so the dashboard
# can render immediately without waiting for model initialisation.

MODEL_INFO: dict[str, dict] = {}


def _sentinel(key: str) -> dict:
    """Placeholder entry shown before the model has been loaded."""
    return {
        "status":      "not_loaded",
        "version":     "—",
        "classes":     [],
        "num_classes": 0,
        "device":      "—",
        "loaded_time": "—",
    }


# ---------------------------------------------------------------------------
# Registry — add future models here, no other changes required
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, ModelEntry] = {
    "infraguard": ModelEntry(
        key="infraguard",
        filename="infraguard.pt",
        description="PPE compliance & general safety detection",
        input_size=(640, 640),
    ),
    "crack": ModelEntry(
        key="crack",
        filename="crack.pt",
        description="Structural crack & surface defect detection",
        input_size=(640, 640),
    ),
    # Future models — uncomment & fill in:
    # "fire": ModelEntry(
    #     key="fire",
    #     filename="fire.pt",
    #     description="Fire and smoke detection",
    # ),
    # "fall": ModelEntry(
    #     key="fall",
    #     filename="fall.pt",
    #     description="Worker fall detection",
    # ),
}

# Seed MODEL_INFO with sentinels so callers never hit a KeyError before load
MODEL_INFO.update({key: _sentinel(key) for key in _REGISTRY})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model(key: str) -> YOLO:
    """Return the loaded YOLO model for *key* (loads on first call)."""
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown model key '{key}'. "
            f"Available models: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[key].load()


def get_model_metadata(key: str) -> dict:
    """Return metadata dict for *key* (loads model if not already loaded)."""
    if key not in _REGISTRY:
        raise KeyError(f"Unknown model key '{key}'.")
    return _REGISTRY[key].metadata


def check_models() -> dict:
    """
    Health-check all registered models.

    Returns a dict shaped like::

        {
            "overall": "ok" | "degraded" | "offline",
            "device":  "cuda:0" | "cpu",
            "memory":  {"allocated_mb": ..., "reserved_mb": ...},   # GPU only
            "models": {
                "infraguard": {"status": "online", "device": "cuda:0", ...},
                "crack":      {"status": "not_loaded", ...},
            }
        }
    """
    model_statuses = {key: entry.health() for key, entry in _REGISTRY.items()}

    loaded = [s for s in model_statuses.values() if s["status"] == "online"]
    if len(loaded) == len(_REGISTRY):
        overall = "ok"
    elif loaded:
        overall = "degraded"
    else:
        overall = "offline"

    # Device & memory
    if torch.cuda.is_available():
        device_str = torch.cuda.get_device_name(0)
        memory = {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024 ** 2, 1),
            "reserved_mb":  round(torch.cuda.memory_reserved()  / 1024 ** 2, 1),
        }
    else:
        device_str = "cpu"
        memory = {}

    return {
        "overall": overall,
        "device": device_str,
        "memory": memory,
        "models": model_statuses,
    }


# ---------------------------------------------------------------------------
# Backwards-compatible convenience accessors (existing call-sites unchanged)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_ppe_model() -> YOLO:
    """Backwards-compatible accessor — delegates to the registry."""
    return get_model("infraguard")


@lru_cache(maxsize=1)
def get_crack_model() -> YOLO:
    """Backwards-compatible accessor — delegates to the registry."""
    return get_model("crack")