import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .key import ExpertKey

@dataclass(frozen=True)
class ExpertTensorSlice:
    name: str
    offset: int
    nbytes: int

@dataclass(frozen=True)
class ExpertRecord:
    key: ExpertKey
    relative_path: str
    nbytes: int
    tensors: dict[str, ExpertTensorSlice]

class ExpertManifest:
    def __init__(self, root: str | Path, records: dict[ExpertKey, ExpertRecord]) -> None:
        self.root = Path(root).resolve()
        self.records = records

    @classmethod
    def load(cls, manifest_path: str | Path) -> "ExpertManifest":
        manifest_file = Path(manifest_path).resolve()
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        records: dict[ExpertKey, ExpertRecord] = {}

        for item in payload.get("experts", []):
            key = ExpertKey(layer=int(item["layer"]), expert=int(item["expert"]))
            relative_path = str(item["path"])
            candidate = (manifest_file.parent / relative_path).resolve()
            if manifest_file.parent not in candidate.parents and candidate != manifest_file.parent:
                raise ValueError(f"expert path escapes model directory: {relative_path}")
            if key in records:
                raise ValueError(f"duplicate expert: {key}")

            tensors: dict[str, ExpertTensorSlice] = {}
            intervals: list[tuple[int, int]] = []
            for name, tensor in item.get("tensors", {}).items():
                offset = int(tensor["offset"])
                nbytes = int(tensor["nbytes"])
                if offset < 0 or nbytes < 0:
                    raise ValueError(f"invalid tensor range for {key}: {name}")
                end = offset + nbytes
                if any(offset < other_end and other_offset < end for other_offset, other_end in intervals):
                    raise ValueError(f"overlapping tensor ranges for {key}")
                intervals.append((offset, end))
                tensors[name] = ExpertTensorSlice(name, offset, nbytes)

            total_nbytes = int(item["nbytes"])
            if any(end > total_nbytes for _, end in intervals):
                raise ValueError(f"tensor range exceeds expert size for {key}")
            records[key] = ExpertRecord(key, relative_path, total_nbytes, tensors)

        return cls(manifest_file.parent, records)

    def path_for(self, key: ExpertKey) -> Path:
        try:
            record = self.records[key]
        except KeyError as error:
            raise KeyError(f"expert not present in manifest: {key}") from error
        return self.root / record.relative_path

    def tensor_slice(self, key: ExpertKey, name: str) -> ExpertTensorSlice:
        try:
            return self.records[key].tensors[name]
        except KeyError as error:
            raise KeyError(f"tensor not present for expert {key}: {name}") from error

    @classmethod
    def from_safetensors_dir(cls, model_dir: str | Path) -> "ExpertManifest":
        """Scans a model directory for safetensors / binary expert weights and generates a manifest."""
        model_path = Path(model_dir).resolve()
        records: list[dict[str, Any]] = []

        # Scans for packed expert binary files or safetensors metadata
        expert_files = sorted(model_path.rglob("expert_*.bin"))
        if not expert_files:
            expert_files = sorted(model_path.rglob("*.safetensors"))

        if not expert_files:
            raise FileNotFoundError(f"No expert .bin or .safetensors files found in {model_path}")

        for file_path in expert_files:
            relative = file_path.relative_to(model_path).as_posix()
            match = re.search(r"layer_(\d+).*expert_(\d+)", file_path.name)
            if match:
                layer = int(match.group(1))
                expert = int(match.group(2))
            else:
                # Default parsing fallback
                layer = 0
                expert = 0

            records.append({
                "layer": layer,
                "expert": expert,
                "path": relative,
                "nbytes": file_path.stat().st_size,
            })

        manifest_path = model_path / "manifest.json"
        manifest_path.write_text(json.dumps({"version": 1, "experts": records}, indent=2), encoding="utf-8")
        return cls.load(manifest_path)
