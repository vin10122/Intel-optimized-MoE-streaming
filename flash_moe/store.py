from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any, Iterator
from .key import ExpertKey, StoreStats

class ExpertStore:
    """Bounded expert store with LRU cache and tensor slicing."""

    def __init__(self, root: str | Path, capacity: int = 1, manifest: Any = None) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.root = Path(root)
        self.capacity = capacity
        self.manifest = manifest
        self._cache: OrderedDict[ExpertKey, bytes] = OrderedDict()
        self.stats = StoreStats()

    def _path_for(self, key: ExpertKey) -> Path:
        if self.manifest is not None:
            return self.manifest.path_for(key)
        return self.root / f"layer_{key.layer:03d}" / f"expert_{key.expert:03d}.bin"

    def get(self, key: ExpertKey) -> bytes:
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.stats.hits += 1
            return cached

        path = self._path_for(key)
        payload = path.read_bytes()
        self.stats.misses += 1
        self.stats.bytes_read += len(payload)
        self._cache[key] = payload
        self._cache.move_to_end(key)

        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return payload

    def read_tensor(self, key: ExpertKey, name: str) -> bytes:
        """Read a tensor slice using in-memory cache if resident, or direct stream.
        
        When expert is not in cache, loads ALL TENSORS for that expert into cache
        as a dict {tensor_name: tensor_bytes} to enable LRU caching with capacity limit.
        """
        if self.manifest is None:
            raise ValueError("read_tensor requires a manifest with tensor slices")
        
        # Check if expert is fully loaded in cache (cache stores dict of tensors)
        if key in self._cache:
            self.stats.hits += 1
            self.stats.tensor_reads += 1
            # Cache contains dict {tensor_name: bytes}
            return self._cache[key][name]
        
        # Expert not in cache - load ALL TENSORS for this expert
        record = self.manifest.records[key]
        path = self._path_for(key)
        
        # Load all tensors for this expert into a dict
        expert_tensors = {}
        total_bytes = 0
        for tensor_name in record.tensors:
            tensor_slice = record.tensors[tensor_name]
            with path.open("rb") as handle:
                handle.seek(tensor_slice.offset)
                tensor_data = handle.read(tensor_slice.nbytes)
            if len(tensor_data) != tensor_slice.nbytes:
                raise IOError(f"short read for {key}/{tensor_name}: expected {tensor_slice.nbytes} bytes")
            expert_tensors[tensor_name] = tensor_data
            total_bytes += len(tensor_data)
        
        # Add to cache (cache stores dict {tensor_name: bytes})
        self._cache[key] = expert_tensors
        self._cache.move_to_end(key)
        
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        
        self.stats.misses += 1
        self.stats.bytes_read += total_bytes
        self.stats.tensor_reads += 1
        
        # Return the requested tensor
        return expert_tensors[name]

    def clear(self) -> None:
        self._cache.clear()

    def resident_keys(self) -> tuple[ExpertKey, ...]:
        return tuple(self._cache)


class RAMBoundedExpertStore(ExpertStore):
    """Store policy: enforce a memory cap on resident experts."""
    def __init__(self, root: str | Path, capacity: int = 2, manifest: Any = None) -> None:
        super().__init__(root=root, capacity=capacity, manifest=manifest)


class AsyncExpertStore(RAMBoundedExpertStore):
    """Asynchronous look-ahead pre-fetching expert store."""

    def __init__(self, root: str | Path, capacity: int = 2, manifest: Any = None):
        super().__init__(root, capacity, manifest)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending_loads: dict[ExpertKey, Future[bytes]] = {}

    def prefetch(self, key: ExpertKey) -> None:
        if key in self._cache or key in self.pending_loads:
            return

        def _load():
            path = self._path_for(key)
            return path.read_bytes()

        self.pending_loads[key] = self.executor.submit(_load)

    def get(self, key: ExpertKey) -> bytes:
        future = self.pending_loads.pop(key, None)
        if future is not None:
            payload = future.result()
            self.stats.misses += 1
            self.stats.bytes_read += len(payload)
            self._cache[key] = payload
            self._cache.move_to_end(key)
            while len(self._cache) > self.capacity:
                self._cache.popitem(last=False)
            return payload

        return super().get(key)

    def close(self) -> None:
        self.executor.shutdown(wait=True)
