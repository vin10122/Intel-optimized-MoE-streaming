import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator

from .key import ExpertKey, StoreStats


class ExpertStore:
    """Bounded expert store with LRU cache and tensor slicing.

    The cache always holds one type per expert: ``dict[str, bytes]`` mapping
    tensor name to its raw (still quantized) bytes. Keeping a single type is
    what makes ``get()`` and ``read_tensor()`` interchangeable; mixing ``bytes``
    and ``dict`` in the same slot makes one of the two raise ``TypeError``
    depending on access order.
    """

    def __init__(self, root: str | Path, capacity: int = 1, manifest: Any = None) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.root = Path(root)
        self.capacity = capacity
        self.manifest = manifest
        self._cache: OrderedDict[ExpertKey, dict[str, bytes]] = OrderedDict()
        self.stats = StoreStats()

    # ------------------------------------------------------------------ paths

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

    # ------------------------------------------------------------------ loading

    def _read_expert(self, key: ExpertKey) -> dict[str, bytes]:
        """Read every tensor of one expert, and only those tensor ranges.

        Must never read the whole model file: with a single-file GGUF manifest
        the expert path is the 16 GiB checkpoint itself, so a naive
        ``path.read_bytes()`` here loads the entire model per expert.
        """
        path = self._path_for(key)
        tensors: dict[str, bytes] = {}
        with path.open("rb") as handle:
            for name, offset, nbytes in self._iter_slices(key):
                handle.seek(offset)
                data = handle.read(nbytes)
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

    # ------------------------------------------------------------------ public

    def read_tensor(self, key: ExpertKey, name: str) -> bytes:
        """Return the raw bytes of one tensor of one expert."""
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
        """Return every tensor of one expert concatenated, in manifest order.

        This is a convenience for callers that want the expert as one blob
        (packing, checksums, raw dumps). Inference code should use
        ``read_tensor`` so it can address the tensors by name.
        """
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
        """Return the dict of tensors of one expert."""
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

    def resident_keys(self) -> tuple[ExpertKey, ...]:
        return tuple(self._cache)


class RAMBoundedExpertStore(ExpertStore):
    """Store policy: enforce a memory cap on resident experts."""

    def __init__(self, root: str | Path, capacity: int = 2, manifest: Any = None) -> None:
        super().__init__(root=root, capacity=capacity, manifest=manifest)


class AsyncPrefetchMixin:
    """Bounded prefetch with cache metadata protected across expert workers.

    Future waits and file I/O stay outside the lock. Failed reads propagate;
    a prefetched result enters the LRU only when inference consumes it.
    """

    def _init_async(self, workers):
        if workers is not None and workers < 1:
            raise ValueError("workers must be at least 1")
        self.workers = self.DEFAULT_WORKERS if workers is None else workers
        self.executor = ThreadPoolExecutor(max_workers=self.workers)
        self.pending_loads = {}
        self.pending_limit = 8
        self._cache_lock = threading.RLock()

    def prefetch(self, key):
        return self.prefetch_many([key])

    def get(self, key):
        return b"".join(self.read_all(key).values())

    def discard_prefetch_except(self, keys):
        needed = set(keys)
        with self._cache_lock:
            for key, future in list(self.pending_loads.items()):
                if key not in needed and (future.done() or future.cancel()):
                    self.pending_loads.pop(key, None)

    def resident_keys(self):
        with self._cache_lock:
            return tuple(self._cache)

    def clear(self):
        with self._cache_lock:
            for future in self.pending_loads.values():
                future.cancel()
            self.pending_loads.clear()
            self._cache.clear()

    def prefetch_many(self, keys):
        queued = 0
        with self._cache_lock:
            for key in keys:
                if key in self._cache or key in self.pending_loads:
                    continue
                if len(self.pending_loads) >= self.pending_limit:
                    break
                if self.manifest is None and not self._path_for(key).exists():
                    continue
                self.pending_loads[key] = self.executor.submit(self._read_expert, key)
                queued += 1
        return queued

    def read_all(self, key):
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                self.stats.hits += 1
                return cached
            future = self.pending_loads.get(key)
        if future is None:
            tensors = self._read_expert(key)
        else:
            # Preserve read errors; silently retrying hides real failures.
            tensors = future.result()
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                self.stats.hits += 1
                return cached
            if future is not None and self.pending_loads.get(key) is future:
                self.pending_loads.pop(key)
                self.stats.prefetch_hits += 1
            else:
                self.stats.misses += 1
            self.stats.bytes_read += sum(map(len, tensors.values()))
            return self._admit(key, tensors)

    def read_tensor(self, key, name):
        if self.manifest is None:
            raise ValueError('read_tensor requires a manifest')
        tensors = self.read_all(key)
        with self._cache_lock:
            self.stats.tensor_reads += 1
        return tensors[name]


class AsyncExpertStore(AsyncPrefetchMixin, RAMBoundedExpertStore):
    """Asynchronous range reads; four I/O workers and eight pending experts."""

    DEFAULT_WORKERS = 4

    def __init__(self, root, capacity=2, manifest=None, workers=None):
        super().__init__(root, capacity, manifest)
        self._init_async(workers)

    def close(self):
        self.executor.shutdown(wait=True)
        self.clear()
