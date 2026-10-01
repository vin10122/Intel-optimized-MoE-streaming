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
    # Letture servite da un prefetch ancora in volo: il byte e' stato letto
    # dal disco lo stesso, ma l'attesa e' caduta su altro lavoro. Va tenuto
    # separato da hits, altrimenti un batch prefetchato azzera misses e la
    # cache sembra perfetta mentre il traffico da disco e' identico.
    prefetch_hits: int = 0

    @property
    def disk_loads(self) -> int:
        """Quante volte un expert e' stato portato in RAM dal disco."""
        return self.prefetch_hits + self.misses
