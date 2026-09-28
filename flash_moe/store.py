from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

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


class AsyncExpertStore(RAMBoundedExpertStore):
    """Asynchronous look-ahead pre-fetching expert store.

    A prefetch submits only the expert's own tensor ranges to the executor.
    The future holds the loaded tensors; the cache is mutated exclusively by
    the calling thread when it consumes the future, so no lock is needed and
    a prefetched-but-unused expert never pollutes the LRU.
    """

    def __init__(self, root: str | Path, capacity: int = 2, manifest: Any = None):
        super().__init__(root, capacity, manifest)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending_loads: dict[ExpertKey, Future] = {}

    def prefetch(self, key: ExpertKey) -> None:
        if key in self._cache or key in self.pending_loads:
            return
        if self.manifest is None and not self._path_for(key).exists():
            return
        self.pending_loads[key] = self.executor.submit(self._read_expert, key)

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
            self.stats.hits += 1
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
            self.stats.hits += 1
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
            self.stats.hits += 1
            return b"".join(prefetched.values())
        tensors = self._read_expert(key)
        self.stats.misses += 1
        self.stats.bytes_read += sum(len(v) for v in tensors.values())
        return b"".join(self._admit(key, tensors).values())

    def close(self) -> None:
        self.executor.shutdown(wait=True)
