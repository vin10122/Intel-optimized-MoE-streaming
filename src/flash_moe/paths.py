"""Percorsi del modello e della DLL: una sola fonte di verita.

Il repository ha attraversato un cambio di modello a meta' lavoro, e i path
erano rimasti hardcoded in ogni script con il nome del modello precedente.
Qui il nome vive in un posto solo e si puo' puntare altrove con la variabile
d'ambiente FLASH_MOE_GGUF, senza toccare il codice.
"""

from __future__ import annotations

import os
from pathlib import Path

MODEL_FILENAME = "Hermes3.6-35B-A3B-Uncensored-Genesis-Final-APEX-Compact.gguf"

# src/flash_moe/paths.py -> src/flash_moe -> src -> root
ROOT = Path(__file__).resolve().parent.parent.parent

_ENV_GGUF = "FLASH_MOE_GGUF"
_ENV_DLL = "FLASH_MOE_DLL"

# L'unica dequantizzazione validata: tools/test_dequant_k.py (gate 162/162).
DLL_NAME = "fast_dequant_k.dll"


def model_path() -> Path:
    """Percorso del GGUF: variabile d'ambiente, altrimenti il default."""
    override = os.environ.get(_ENV_GGUF)
    if override:
        return Path(override).expanduser().resolve()
    return ROOT / MODEL_FILENAME


def dll_path() -> Path:
    """Percorso della DLL K-quant, accanto a questo modulo."""
    override = os.environ.get(_ENV_DLL)
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent / DLL_NAME


def require_model() -> Path:
    """Come model_path(), ma fallisce subito con un messaggio utile."""
    path = model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"modello non trovato: {path}\n"
            f"imposta {_ENV_GGUF} per puntare a un altro GGUF"
        )
    return path
