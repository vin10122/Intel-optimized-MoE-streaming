"""Expert store con memory-mapped I/O per file GGUF.

Per modelli single-file GGUF (es. Hermes3.6-35B), questa implementazione:
1. Mappa l'intero file GGUF in memoria una volta sola
2. Legge i tensor degli expert direttamente dalla mappatura
3. Evita completamente le syscall di I/O

Vantaggi:
- Nessun overhead di apertura/chiusura file
- Nessun seek() costoso su file grandi
- Letture istantanee (solo memcpy dalla memoria mappata)
- Prefetch del sistema operativo gestito automaticamente

Svantaggi:
- Richiede abbastanza RAM per il file + working set
- Su Windows, necessita di spazio addressabile
"""

import mmap
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator
import threading

from .key import ExpertKey, StoreStats
from .store import AsyncPrefetchMixin


class MemoryMappedGGUF:
    """Memory-mapped wrapper per file GGUF."""
    
    def __init__(self, path: Path):
        self.path = path
        self._file = None
        self._mapping = None
        self._lock = threading.Lock()
        self._refcount = 0
    
    def open(self):
        """Open and memory-map the file."""
        with self._lock:
            if self._mapping is not None:
                return
            
            self._file = open(self.path, "rb")
            self._mapping = mmap.mmap(
                self._file.fileno(),
                0,
                access=mmap.ACCESS_READ,
                offset=0
            )
            self._refcount = 1
    
    def close(self):
        """Close the mapping and file."""
        with self._lock:
            if self._mapping is None:
                return
            
            try:
                self._mapping.close()
            except Exception:
                pass
            self._mapping = None
            if self._file:
                try:
                    self._file.close()
                except Exception:
                    pass
                self._file = None
            self._refcount = 0
    
    def read(self, offset: int, size: int) -> bytes:
        """Read bytes from memory mapping."""
        if self._mapping is None:
            self.open()
        return self._mapping[offset:offset + size]
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MemoryMappedExpertStore:
    """Expert store che usa memory-mapped I/O per file GGUF single-file."""

    def __init__(self, root: str | Path, capacity: int = 1, manifest: Any = None) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.root = Path(root)
        self.capacity = capacity
        self.manifest = manifest
        self._cache: OrderedDict[ExpertKey, dict[str, bytes]] = OrderedDict()
        self.stats = StoreStats()
        self._gguf_map: MemoryMappedGGUF | None = None
        self._gguf_path: Path | None = None

    def _get_gguf_map(self) -> MemoryMappedGGUF:
        """Get or create memory-mapped GGUF file."""
        if self.manifest is None:
            raise ValueError("MemoryMappedExpertStore requires a manifest")
        
        # Get the GGUF path from manifest root
        from pathlib import Path
        gguf_path = None
        for f in self.manifest.root.iterdir():
            if f.suffix == '.gguf':
                gguf_path = f
                break
        if gguf_path is None:
            raise ValueError(f"No GGUF file found in {self.manifest.root}")
        
        if self._gguf_map is not None and self._gguf_path == gguf_path:
            return self._gguf_map
        
        # Create new mapping
        if self._gguf_map is not None:
            self._gguf_map.close()
        
        self._gguf_map = MemoryMappedGGUF(gguf_path)
        self._gguf_map.open()
        self._gguf_path = gguf_path
        return self._gguf_map

    def _path_for(self, key: ExpertKey) -> Path:
        if self.manifest is not None:
            return self.manifest.path_for(key)
        return self.root / f"layer_{key.layer:03d}" / f"expert_{key.expert:03d}.bin"

    def _iter_slices(self, key: ExpertKey) -> Iterator[tuple[str, int, int]]:
        if self.manifest is None:
            yield "payload", 0, self._path_for(key).stat().st_size
            return
        record = self.manifest.records[key]
        for name, tensor in record.tensors.items():
            yield name, tensor.offset, tensor.nbytes

    def _read_expert(self, key: ExpertKey) -> dict[str, bytes]:
        """Read expert tensors from memory-mapped file."""
        gguf_map = self._get_gguf_map()
        tensors: dict[str, bytes] = {}
        
        for name, offset, nbytes in self._iter_slices(key):
            # Direct read from memory mapping
            data = gguf_map.read(offset, nbytes)
            if len(data) != nbytes:
                raise IOError(
                    f"short read for {key}/{name}: "
                    f"expected {nbytes} bytes, got {len(data)}"
                )
            tensors[name] = data
        
        return tensors

    def _admit(self, key: ExpertKey, tensors: dict[str, bytes]) -> dict[str, bytes]:
        self._cache[key] = tensors
        self._cache.move_to_end(key)
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return tensors

    def read_tensor(self, key: ExpertKey, name: str) -> bytes:
        if self.manifest is None:
            raise ValueError("read_tensor requires a manifest with tensor slices")

        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.stats.hits += 1
            self.stats.tensor_reads += 1
            return cached[name]

        tensors = self._read_expert(key)
        self.stats.misses += 1
        self.stats.tensor_reads += 1
        self.stats.bytes_read += sum(len(v) for v in tensors.values())
        return self._admit(key, tensors)[name]

    def get(self, key: ExpertKey) -> bytes:
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.stats.hits += 1
            return b"".join(cached.values())

        tensors = self._read_expert(key)
        self.stats.misses += 1
        self.stats.bytes_read += sum(len(v) for v in tensors.values())
        return b"".join(self._admit(key, tensors).values())

    def read_all(self, key: ExpertKey) -> dict[str, bytes]:
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.stats.hits += 1
            return cached
        tensors = self._read_expert(key)
        self.stats.misses += 1
        self.stats.bytes_read += sum(len(v) for v in tensors.values())
        return self._admit(key, tensors)

    def clear(self) -> None:
        self._cache.clear()

    def close(self) -> None:
        self.clear()
        if self._gguf_map is not None:
            self._gguf_map.close()
            self._gguf_map = None
        self._gguf_path = None

    def resident_keys(self) -> tuple[ExpertKey, ...]:
        return tuple(self._cache)


class MemoryMappedAsyncExpertStore(AsyncPrefetchMixin, MemoryMappedExpertStore):
    """Memory-mapped reads with the same bounded, synchronized cache policy."""

    DEFAULT_WORKERS = 8

    def __init__(self, root, capacity=2, manifest=None, workers=None):
        super().__init__(root, capacity, manifest)
        self._init_async(workers)

    def _read_expert(self, key):
        # Mapping creation must not race between reader threads.
        with self._cache_lock:
            self._get_gguf_map()
        return super()._read_expert(key)

    def close(self):
        self.executor.shutdown(wait=True)
        super().close()
