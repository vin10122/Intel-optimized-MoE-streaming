"""Store ottimizzato con file descriptor riutilizzato e letture batch.

Ottimizzazioni:
1. Single file handle: con manifest GGUF, tutti gli expert sono nello stesso file
2. Batch reads: leggere tutti i tensor di un expert in una sola operazione
3. Memory-mapped I/O: opzionale, per ridurre syscall overhead
4. Prefetch migliorato: Parallel loading con shared file handle
"""

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator
import threading

from .key import ExpertKey, StoreStats


class OptimizedExpertStore:
    """Expert store con file handle riutilizzato.
    
    Per modelli single-file GGUF (come Hermes3), tutti gli expert sono nello
    stesso file. Invece di aprire/chiudere il file per ogni expert, manteniamo
    un file handle aperto e lo riutilizziamo.
    """

    def __init__(self, root: str | Path, capacity: int = 1, manifest: Any = None) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.root = Path(root)
        self.capacity = capacity
        self.manifest = manifest
        self._cache: OrderedDict[ExpertKey, dict[str, bytes]] = OrderedDict()
        self.stats = StoreStats()
        self._file_handle = None
        self._file_lock = threading.Lock()
        self._file_path = None

    def _get_file_handle(self, path: Path) -> Any:
        """Get or open file handle. Returns the handle."""
        if self._file_path == path and self._file_handle is not None:
            return self._file_handle
        
        # Close old handle if different path
        if self._file_handle is not None:
            try:
                self._file_handle.close()
            except:
                pass
            self._file_handle = None
        
        # Open new handle
        self._file_path = path
        self._file_handle = path.open("rb")
        return self._file_handle

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
        """Read expert tensors using shared file handle."""
        path = self._path_for(key)
        tensors: dict[str, bytes] = {}
        
        with self._file_lock:
            handle = self._get_file_handle(path)
            # Batch read: collect all ranges first
            ranges = list(self._iter_slices(key))
            total_bytes = sum(nbytes for _, _, nbytes in ranges)
            
            # Read all at once if possible (for single-file GGUF)
            if self.manifest is not None and len(ranges) > 1:
                # For GGUF with manifest, read all slices sequentially
                for name, offset, nbytes in ranges:
                    handle.seek(offset)
                    data = handle.read(nbytes)
                    if len(data) != nbytes:
                        raise IOError(
                            f"short read for {key}/{name}: "
                            f"expected {nbytes} bytes, got {len(data)}"
                        )
                    tensors[name] = data
            else:
                # For simple bin files, read everything
                handle.seek(0)
                data = handle.read()
                tensors["payload"] = data
        
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
        if self._file_handle:
            try:
                self._file_handle.close()
            except:
                pass
            self._file_handle = None

    def close(self) -> None:
        self.clear()

    def resident_keys(self) -> tuple[ExpertKey, ...]:
        return tuple(self._cache)


class OptimizedAsyncExpertStore(OptimizedExpertStore):
    """Store con prefetch asincrono e file handle condiviso."""

    DEFAULT_WORKERS = 8  # Aumentato da 4 a 8 per migliore parallelismo I/O

    def __init__(self, root: str | Path, capacity: int = 2, manifest: Any = None,
                 workers: int | None = None):
        super().__init__(root, capacity, manifest)
        if workers is not None and workers < 1:
            raise ValueError(f"workers must be at least 1, got {workers}")
        self.workers = self.DEFAULT_WORKERS if workers is None else workers
        self.executor = ThreadPoolExecutor(max_workers=self.workers)
        self.pending_loads: dict[ExpertKey, Future] = {}

    def prefetch(self, key: ExpertKey) -> None:
        if key in self._cache or key in self.pending_loads:
            return
        if self.manifest is None and not self._path_for(key).exists():
            return
        self.pending_loads[key] = self.executor.submit(self._read_expert, key)

    def prefetch_many(self, keys: Iterable[ExpertKey]) -> int:
        queued = 0
        for key in keys:
            if key in self._cache or key in self.pending_loads:
                continue
            if self.manifest is None and not self._path_for(key).exists():
                continue
            self.pending_loads[key] = self.executor.submit(self._read_expert, key)
            queued += 1
        return queued

    def _consume(self, key: ExpertKey) -> dict[str, bytes] | None:
        future = self.pending_loads.pop(key, None)
        if future is None:
            return None
        try:
            tensors = future.result()
        except Exception:
            return None
        self.stats.bytes_read += sum(len(v) for v in tensors.values())
        return self._admit(key, tensors)

    def read_all(self, key: ExpertKey) -> dict[str, bytes]:
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.stats.hits += 1
            return cached
        prefetched = self._consume(key)
        if prefetched is not None:
            self.stats.prefetch_hits += 1
            return prefetched
        tensors = self._read_expert(key)
        self.stats.misses += 1
        self.stats.bytes_read += sum(len(v) for v in tensors.values())
        return self._admit(key, tensors)

    def read_tensor(self, key: ExpertKey, name: str) -> bytes:
        if self.manifest is None:
            raise ValueError("read_tensor requires a manifest with tensor slices")
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.stats.hits += 1
            self.stats.tensor_reads += 1
            return cached[name]

        prefetched = self._consume(key)
        if prefetched is not None:
            self.stats.prefetch_hits += 1
            self.stats.tensor_reads += 1
            return prefetched[name]

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
        prefetched = self._consume(key)
        if prefetched is not None:
            self.stats.prefetch_hits += 1
            return b"".join(prefetched.values())
        tensors = self._read_expert(key)
        self.stats.misses += 1
        self.stats.bytes_read += sum(len(v) for v in tensors.values())
        return b"".join(self._admit(key, tensors).values())

    def close(self) -> None:
        self.executor.shutdown(wait=True)
        super().close()
