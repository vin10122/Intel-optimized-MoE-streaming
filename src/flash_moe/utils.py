import struct
from pathlib import Path
from .key import ExpertKey

def unpack_f32(payload: bytes) -> list[float]:
    if len(payload) % 4 != 0:
        raise ValueError("f32 payload size must be divisible by 4")
    return list(struct.unpack(f"<{len(payload) // 4}f", payload))

def pack_f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)

def unpack_f16(payload: bytes) -> list[float]:
    if len(payload) % 2 != 0:
        raise ValueError("f16 payload size must be divisible by 2")
    return list(struct.unpack(f"<{len(payload) // 2}e", payload))

def pack_f16(values: list[float]) -> bytes:
    """Impacchetta in float16. E' il formato con cui il backend legge gli
    expert non quantizzati: i pesi in RAM sono f16."""
    return struct.pack(f"<{len(values)}e", *values)

def pack_expert(
    root: str | Path,
    key: ExpertKey,
    tensors: dict[str, bytes],
) -> dict[str, object]:
    required = ("gate", "up", "down")
    missing = [name for name in required if name not in tensors]
    if missing:
        raise ValueError(f"missing expert tensors: {', '.join(missing)}")

    root_path = Path(root)
    relative_path = Path("experts") / f"layer_{key.layer:03d}" / f"expert_{key.expert:03d}.bin"
    output_path = root_path / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    offset = 0
    tensor_manifest: dict[str, dict[str, int]] = {}
    with output_path.open("wb") as output:
        for name in required:
            payload = tensors[name]
            tensor_manifest[name] = {"offset": offset, "nbytes": len(payload)}
            output.write(payload)
            offset += len(payload)

    return {
        "layer": key.layer,
        "expert": key.expert,
        "path": relative_path.as_posix(),
        "nbytes": offset,
        "tensors": tensor_manifest,
    }

def write_manifest(root: str | Path, records: list[dict[str, object]]) -> Path:
    import json
    manifest_path = Path(root) / "manifest.json"
    manifest_path.write_text(
        json.dumps({"version": 1, "experts": records}, indent=2),
        encoding="utf-8",
    )
    return manifest_path
