"""Il peso qkv e' letto con gli assi scambiati?

`load_layer_weights` prende la forma dal GGUF cosi' com'e':
    shape = tuple(int(v) for v in tensor.shape)
mentre `_get_tensor` la prende al contrario:
    shape = tuple(int(s) for s in reversed(tensor.shape))

Nel GGUF l'asse 0 e' quello rapido, quindi la memoria di un peso e'
(out, in) e per un (n_in, n_out) la forma da usare e' l'inversa di quella
del GGUF. Qui si provano entrambe le letture sugli stessi elementi che
llama.cpp ha stampato, e si vede quale coincide.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\valer\Desktop\moe-client-prot\Nuova cartella")
sys.path.insert(0, str(ROOT / "src"))

import gguf  # noqa: E402
from flash_moe.dequant import dequantize_to, name_of  # noqa: E402
from flash_moe.gated_deltanet import _silu  # noqa: E402
from flash_moe.norm import RMSNorm  # noqa: E402
from flash_moe.paths import model_path  # noqa: E402

PROMPT = "La capital of France is"
REF_QKV = [  # primi 3 e ultimi 3 di linear_attn_qkv_mixed-0, posizione 0
    -234.818970, None, None, None, None, None,
]

path = model_path()
reader = gguf.GGUFReader(path)
tmap = {t.name: t for t in reader.tensors}

t = tmap["blk.0.attn_qkv.weight"]
gguf_shape = tuple(int(v) for v in t.shape)
n = t.n_elements
print(f"attn_qkv: n={n}, forma nel GGUF {gguf_shape}, "
      f"inverso {gguf_shape[::-1]}, tipo {name_of(int(t.tensor_type))}")

flat = np.ascontiguousarray(t.data, dtype=np.float16).reshape(-1)
varianti = {
    "reshape(forma GGUF)  (2048, 8192)": flat.reshape(gguf_shape),
    "reshape(inversa)    (8192, 2048)": flat.reshape(gguf_shape[::-1]),
}

norm_t = tmap["blk.0.attn_norm.weight"]
norm_w = np.asarray(norm_t.data, dtype=np.float16)

# attn_norm: il suo output coincide con llama.cpp, quindi lo ricalcolo qui e
# uso quello come ingresso del confronto.
from flash_moe.tokenizer import GGUFTokenizer  # noqa: E402
from flash_moe.embedding import EmbeddingLookup  # noqa: E402

tok = GGUFTokenizer(reader)
ids = tok.encode(PROMPT, add_bos=False)
emb = EmbeddingLookup.from_gguf_tensor(tmap["token_embd.weight"])
x = emb.forward(np.asarray(ids, dtype=np.int64)).reshape(1, len(ids), -1)
normed = RMSNorm(norm_w, eps=1e-6).forward(x)[0]

LCPP = Path(r"C:\Users\valer\AppData\Local\Temp\kilo\lcpp_eval.txt")
NUM = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?")
TENSOR = re.compile(r"^\s*common_debug_cb_eval:\s+(\S+) = \(f32\)")


def ref_row(name: str, pos: int) -> list[float]:
    """I 6 valori stampati per la posizione data del tensore `name`."""
    lines = LCPP.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        if TENSOR.match(line) and TENSOR.match(line).group(1) == name:
            rows = []
            for j in range(i + 1, len(lines)):
                st = lines[j].strip()
                if TENSOR.match(lines[j]) or st.startswith("sum"):
                    break
                if st.startswith("["):
                    nums = [float(v) for v in NUM.findall(st)]
                    if nums:
                        rows.append(nums)
            return rows[pos]
    raise KeyError(name)


llcpp = ref_row("linear_attn_qkv_mixed-0", 0)
print(f"\nllama.cpp, qkv posizione 0, primi 3 valori: "
      f"{llcpp[0]:.6f} {llcpp[1]:.6f} {llcpp[2]:.6f}")

for label, w in varianti.items():
    qkv = normed[0] @ (w.T if w.shape[0] != 2048 else w).astype(np.float32)
    print(f"\n{label}")
    print(f"  nostri primi 3: {np.array2string(qkv[:3], precision=4)}")
    print(f"  llama.cpp     : {np.array2string(np.asarray(llcpp[:3]), precision=4)}")
    err = float(np.abs(qkv[:3] - np.asarray(llcpp[:3])).max())
    print(f"  errore assoluto sui primi tre: {err:.6f}")

