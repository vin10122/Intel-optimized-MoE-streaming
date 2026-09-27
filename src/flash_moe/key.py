from dataclasses import dataclass

@dataclass(frozen=True)
class ExpertKey:
    layer: int
    expert: int

@dataclass
class StoreStats:
    hits: int = 0
    misses: int = 0
    bytes_read: int = 0
    tensor_reads: int = 0
