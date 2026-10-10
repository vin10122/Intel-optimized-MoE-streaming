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


def _gguf_candidates() -> list[Path]:
    """Ritorna i possibili GGUF in ordine di preferenza.

    Se il modello gia' esiste nella cartella del progetto, viene usato senza
    richiedere un nome file hardcoded; se invece il nome cambia, si cerca in
    tutto il tree del progetto e si preferiscono i file con quantizzazione Q4_K.
    """
    candidates: list[Path] = []
    seen: set[Path] = set()

    for path in [ROOT / MODEL_FILENAME, *sorted(ROOT.rglob("*.gguf"))]:
        resolved = path.expanduser().resolve()
        if resolved.exists() and resolved not in seen:
            candidates.append(resolved)
            seen.add(resolved)

    def score(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        score = 0
        if "q4_k" in name:
            score += 100
        elif "q4" in name:
            score += 50
        if "m" in name and "q4_k" in name:
            score += 10
        if path.name == MODEL_FILENAME:
            score += 20
        return (score, name)

    return sorted(candidates, key=lambda p: (-score(p)[0], score(p)[1]))


def model_path() -> Path:
    """Percorso del GGUF: override ambiente, poi default storico, poi scan della cartella."""
    override = os.environ.get(_ENV_GGUF)
    if override:
        return Path(override).expanduser().resolve()

    default_path = ROOT / MODEL_FILENAME
    if default_path.exists():
        return default_path

    candidates = _gguf_candidates()
    if candidates:
        return candidates[0]

    return default_path


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
