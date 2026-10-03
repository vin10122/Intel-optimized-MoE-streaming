"""Dequantizzazione dei tensor GGUF con dispatch sul tipo ggml.

Perche' questo modulo esiste
---------------------------
Gli script precedenti facevano::

    dequant = gguf.quants.dequantize(raw, type).astype(np.float32)
    return dequant.astype(np.uint16)          # <-- troncamento a intero

I pesi reali di un expert valgono 0..0,0855, quindi ``astype(np.uint16)``
produce 0..0: una matrice di pesi **interamente zero**, non una approssimazione
grossolana. Il cast giusto e' reinterpretare i bit in virgola mobile mobile, non
convertire a intero.

Nessun kernel C++ e' collegato di default. ``flash_moe/fast_dequant_q8_0.dll`` e'
noto essere numericamente errato (dichiara ``scales[8], qs[256]`` mentre il
layout reale e' interleaved ``d, qs[32], d, qs[32], ...``, e applica un offset
``-128`` che Q8_0 non ha: il suo formato e' ``y = qs[j] * d``). Va abilitato solo
dopo aver passato il gate numerico in ``tools/test_phase1.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ggml type -> (block_size, bytes_per_block). Serve per validare che una slice
# di expert e' allineata al blocco di quantizzazione prima di dequantizzarla.
GGML_BLOCK_INFO: dict[str, tuple[int, int]] = {
    "F32": (1, 4),
    "F16": (1, 2),
    "BF16": (1, 2),
    "Q4_0": (32, 18),
    "Q4_1": (32, 20),
    "Q5_0": (32, 22),
    "Q5_1": (32, 24),
    "Q8_0": (32, 34),
    "Q2_K": (256, 84),
    "Q3_K": (256, 110),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "Q6_K": (256, 210),
    "Q8_K": (256, 292),
    "IQ4_NL": (32, 18),
    "IQ4_XS": (256, 136),
    "IQ3_XXS": (256, 98),
    "IQ2_XXS": (256, 66),
    "IQ1_S": (256, 50),
}

_DTYPES: dict[str, np.dtype] = {
    "f32": np.dtype(np.float32),
    "f16": np.dtype(np.float16),
}


def _gguf():
    import gguf

    return gguf


def type_name_map() -> dict[int, str]:
    """Map ggml type id -> nome leggibile."""
    gguf = _gguf()
    names: dict[int, str] = {}
    for key, value in vars(gguf.GGMLQuantizationType).items():
        if isinstance(value, int) and not key.startswith("_"):
            names.setdefault(value, key)
    return names


def name_of(ggml_type: int) -> str:
    return type_name_map().get(int(ggml_type), f"type_{int(ggml_type)}")


def is_aligned(ggml_type: int, nbytes: int) -> bool:
    """True se ``nbytes`` e' un multiplo della dimensione del blocco."""
    info = GGML_BLOCK_INFO.get(name_of(ggml_type))
    if info is None:
        return False
    return nbytes % info[1] == 0


def dequantize(raw: bytes | bytearray | np.ndarray, ggml_type: int,
               shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Dequantizza in float32. Non troncare mai il risultato a intero."""
    gguf = _gguf()
    if isinstance(raw, (bytes, bytearray)):
        data = np.frombuffer(raw, dtype=np.uint8)
    else:
        data = np.ascontiguousarray(raw, dtype=np.uint8)
    if data.size == 0:
        raise ValueError("empty buffer passed to dequantize")
    if not is_aligned(ggml_type, data.nbytes):
        info = GGML_BLOCK_INFO.get(name_of(ggml_type), (0, 0))
        raise ValueError(
            f"{data.nbytes} bytes is not a whole number of "
            f"{name_of(ggml_type)} blocks ({info[1]} B per blocco)"
        )
    out = gguf.quants.dequantize(data, ggml_type)
    if shape is not None:
        out = out.reshape(*shape)
    return np.ascontiguousarray(out, dtype=np.float32)


def dequantize_to(raw: bytes | bytearray | np.ndarray, ggml_type: int,
                  dtype: str = "f16",
                  shape: tuple[int, ...] | None = None,
                  chunk_bytes: int = 32 * 1024 * 1024) -> np.ndarray:
    """Dequantizza direttamente nel dtype richiesto ("f16" o "f32").

    ``astype`` su un array float e' una reinterpretazione con perdita di
    precisione, non un troncamento a intero: e' il percorso corretto.

    In f16 il dequantizzato viene scritto direttamente nel buffer finale, a
    blocchi di quantizzazione, senza materializzare mai l'array f32. Su
    ``token_embd`` (508M elementi) il percorso f32 materializza 1960 MiB di
    picco; questo ne usa 971 MiB, che sono l'output. Il risultato e' bit
    identico: la dequantizzazione e' indipendente per blocco, quindi spezzare
    la sorgente all'allineamento del blocco non cambia un bit.
    """
    np_dtype = _DTYPES.get(dtype)
    if np_dtype is None:
        raise ValueError(f"unsupported dtype {dtype!r}, expected f32 or f16")
    if dtype == "f32":
        return dequantize(raw, ggml_type, shape)

    info = GGML_BLOCK_INFO.get(name_of(ggml_type))
    if info is None:
        raise ValueError(f"unknown ggml type: {name_of(ggml_type)}")
    block_size, bytes_per_block = info

    if isinstance(raw, (bytes, bytearray)):
        data = np.frombuffer(raw, dtype=np.uint8)
    else:
        data = np.ascontiguousarray(raw, dtype=np.uint8)
    if data.size == 0:
        raise ValueError("empty buffer passed to dequantize_to")
    if data.nbytes % bytes_per_block != 0:
        raise ValueError(
            f"{data.nbytes} bytes is not a whole number of "
            f"{name_of(ggml_type)} blocks ({bytes_per_block} B per blocco)"
        )

    # F16 e' gia' in virgola mobile: si reinterpretano i bit senza passare da
    # gguf.quants, che riporterebbe tutto a f32. Solo F16: BF16 ha un esponente
    # diverso e non puo' essere letto come float16.
    if name_of(ggml_type) == "F16":
        out = np.frombuffer(data, dtype=np.float16)
        return _finish(out, shape)

    gguf = _gguf()
    n_blocks = data.nbytes // bytes_per_block
    blocks_per_chunk = max(1, chunk_bytes // bytes_per_block)
    out = np.empty(n_blocks * block_size, dtype=np.float16)

    written = 0
    for start in range(0, n_blocks, blocks_per_chunk):
        count = min(blocks_per_chunk, n_blocks - start)
        chunk = data[start * bytes_per_block:(start + count) * bytes_per_block]
        piece = np.asarray(gguf.quants.dequantize(chunk, ggml_type),
                           dtype=np.float16).reshape(-1)
        # Il reshape serve perche' gguf.quants.dequantize restituisce una
        # colonna (n, 1) invece di un vettore: senza, l'assegnazione in un
        # buffer 1-D fallisce con "could not broadcast (8192,1) into (8192,)".
        # Il conteggio viene controllato perche' con il reshape un disaccordo
        # passerebbe silenzioso e scriverebbe fuori dal buffer.
        want = count * block_size
        if piece.size != want:
            raise ValueError(
                f"la reference gguf ha restituito {piece.size} elementi "
                f"per {want} attesi ({count} blocchi da {block_size}): i due "
                f"non concordano"
            )
        out[written:written + want] = piece
        written += want

    return _finish(out, shape)


def _finish(out: np.ndarray, shape: tuple[int, ...] | None) -> np.ndarray:
    """Applica lo shape richiesto e garantisce un array contiguo."""
    if shape is not None:
        out = out.reshape(*shape)
    return np.ascontiguousarray(out, dtype=np.float16)


def f32_to_f16(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float16)


def f16_to_f32(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float32)


def max_relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    scale = float(np.abs(reference).max())
    if scale == 0.0:
        return float(np.abs(actual).max())
    return float(np.abs(actual.astype(np.float64) - reference).max() / scale)
