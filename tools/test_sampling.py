"""Il campionamento, verificato contro la specifica e non a occhio.

``sample`` decide il token successivo ed e' l'ultima cosa che un client
streaming consegna all'utente, ma non era mai stato esercitato. Il difetto
che qui conta non e' che scelga un token diverso dal greedy: e' che possa
scegliere un token fuori dal nucleo. Un token fuori dal nucleo non e' un
testo peggiore, e' testo che non poteva uscire.

Il confronto non puo' essere fatto con llama.cpp campionando: i due runtime
hanno PRNG diversi, quindi con lo stesso seed non darebbero lo stesso token
nemmeno se il campionamento fosse identico. Quindi qui si confronta con una
implementazione lenta e ovviamente corretta della stessa specifica, e in piu'
si controlla la proprieta' che conta davvero: il campionamento non esce mai
fuori dal nucleo, e la frequenza dei token somiglia alla probabilita' che
dice la specifica.

Le distribuzioni si controllano su un vocabolario ridotto ai 200 token piu'
probabili, perche' un argsort su 248320 elementi costa 20 ms e il test ne
farebbe decine di migliaia. Il nucleo di partenza e' pero' calcolato sul
vocabolario intero e poi confrontato con lo slice: se il nucleo esce dai 200
token, il test lo dice e salta quel caso invece di verificare uno slice che
non e' piu' quello di chi chiama ``sample``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# La console di Windows e' cp1252 e non sa scrivere i caratteri che questi
# test usano nelle etichette. Senza questo, il test muore su un accento invece
# di dire cosa ha trovato.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gguf  # noqa: E402
from run_q4k_f16_openvino import dequant_to_f16  # noqa: E402
from flash_moe import Qwen3MoEModel  # noqa: E402
from flash_moe.embedding import sample  # noqa: E402
from flash_moe.paths import model_path  # noqa: E402
from flash_moe.tokenizer import GGUFTokenizer  # noqa: E402

VOCAB_SLICE = 200
DRAWS = 4000
FULL_DRAWS = 400
REF_PATH = Path(__file__).resolve().parent / "golden_reference.json"

ok = bad = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global ok, bad
    if cond:
        ok += 1
        print(f"  PASS  {label}  ({detail})")
    else:
        bad += 1
        print(f"  FAIL  {label}  ({detail})")


def reference_nucleus(logits: np.ndarray, temperature: float,
                      top_k: int, top_p: float) -> tuple[list[int], np.ndarray]:
    """Specifica del campionamento, scritta lentamente e in chiaro.

    Ordinamento decrescente sui logit, si tiene il primo ``top_k``, poi su
    quelli si prende il primo set che raggiunge ``top_p``, e infine si
    ripartisce la probabilita' sul set rimasto. Restituisce gli id tenuti e
    le loro probabilita', cosi' il test puo' controllarle entrambe.
    """
    z = np.asarray(logits, dtype=np.float64)
    if temperature <= 0.0:
        return [int(np.argmax(z))], np.array([1.0])
    if temperature != 1.0:
        z = z / temperature
    order = [i for i in sorted(range(len(z)), key=lambda i: -z[i])]
    if top_k > 0:
        order = order[:top_k]
    if top_p > 0.0:
        total = float(sum(np.exp(z[i] - z[order[0]]) for i in order))
        acc = 0.0
        cut = []
        for i in order:
            cut.append(i)
            acc += float(np.exp(z[i] - z[order[0]])) / total
            if acc >= top_p:
                break
        order = cut
    top = z[order[0]]
    weights = np.array([float(np.exp(z[i] - top)) for i in order])
    return order, weights / weights.sum()


def main() -> int:
    path = model_path()
    reader = gguf.GGUFReader(path)
    tok = GGUFTokenizer(reader)
    model = Qwen3MoEModel(path, device="CPU", max_seq_len=256,
                          cache_capacity=64, dequant_fn=dequant_to_f16)
    model.init_caches()

    ids = tok.encode("La capital of France is", add_bos=False)
    x = model.embedding.forward(np.asarray(ids, dtype=np.int64))
    x = x.reshape(1, len(ids), model.hidden_size)
    x = model._forward_all(x)
    logits = np.asarray(model.lm_head.forward(model.output_norm.forward(x)))
    last = np.asarray(logits[0, -1], dtype=np.float64).reshape(-1)
    greedy = int(np.argmax(last))
    print(f"vocabolario {last.size}, token greedy {greedy} "
          f"{tok.decode([greedy])!r}\n")

    print("1. invarianti")
    check("temperature 0 dà il greedy", sample(last, temperature=0.0) == greedy)
    check("top_k 1 dà il greedy", sample(last, top_k=1) == greedy)
    check("temperature 0 con top_k e top_p dà il greedy",
          sample(last, temperature=0.0, top_k=20, top_p=0.9) == greedy)
    top_p_tiny = float(np.exp(last[greedy] - np.max(last))
                       / np.sum(np.exp(last - np.max(last))))
    check("top_p sotto la probabilità del primo tiene un solo token",
          all(sample(last, top_p=top_p_tiny * 0.5,
                     rng=np.random.default_rng(s)) == greedy
              for s in range(5)),
          f"top_p={top_p_tiny * 0.5:.4f} sotto il primo {top_p_tiny:.4f}")
    try:
        sample(np.where(np.arange(last.size) == greedy, np.inf, last))
        check("logit non finiti rifiutati", False, "nessuna eccezione")
    except ValueError:
        check("logit non finiti rifiutati", True, "ValueError")
    nan_logits = last.astype(np.float64).copy()
    nan_logits[0] = np.nan
    try:
        sample(nan_logits)
        check("logit NaN rifiutati", False, "nessuna eccezione")
    except ValueError:
        check("logit NaN rifiutati", True, "ValueError")

    print("\n2. determinismo")
    a = sample(last, temperature=0.9, top_k=20, top_p=0.95,
               rng=np.random.default_rng(7))
    b = sample(last, temperature=0.9, top_k=20, top_p=0.95,
               rng=np.random.default_rng(7))
    check("stesso seed, stesso token", a == b, f"{a} e {b}")

    draws = [sample(last, temperature=0.9, top_k=20, top_p=0.95,
                    rng=np.random.default_rng(s)) for s in range(24)]
    ids_ref, probs = reference_nucleus(last, 0.9, 20, 0.95)
    # "seed diversi danno token diversi" e' una domanda probabilistica: senza
    # guardare le probabilita' non si sa se il test puo' fallire per caso. Con
    # 24 estrazioni la probabilita' di un falso negativo e' p_max**23, quindi
    # la domanda si pone solo sotto una soglia che la rende trascurabile,
    # altrimenti si controlla solo la proprieta' che conta: essere nel nucleo.
    check("seed diversi campionano dentro il nucleo",
          set(draws) <= {int(i) for i in ids_ref},
          f"{len(set(draws))} token distinti su 24 estrazioni")
    if float(probs.max()) ** 23 < 1e-3:
        check("seed diversi danno token diversi", len(set(draws)) > 1,
              f"{len(set(draws))} distinti, p_max {probs.max():.3f}")
    else:
        print(f"  info  p_max {probs.max():.3f} troppo alto per pretendere "
              f"token distinti: il controllo sopra sul nucleo basta")

    print(f"\n3. il nucleo sui {VOCAB_SLICE} token più probabili")
    top_ids = np.argsort(last)[-VOCAB_SLICE:]
    sliced = last[top_ids]

    combos = [
        ("temperature 0.7, top_k 40, top_p 0.95", 0.7, 40, 0.95),
        ("temperature 1.0, top_k 20, top_p 0.90", 1.0, 20, 0.90),
        ("temperature 1.3, top_k 100, top_p 0.99", 1.3, 100, 0.99),
        ("temperature 0.5, top_k 200, top_p 0.0", 0.5, 200, 0.0),
        ("temperature 1.0, top_k 0, top_p 0.99", 1.0, 0, 0.99),
    ]
    for label, temp, tk, tp in combos:
        # Il riferimento si calcola SEMPRE sul vocabolario intero. Sarebbe
        # comodo farlo sullo slice, ma li' si assume che il nucleo stia nei
        # 200 token piu' probabili, e con top_k = 0 quella ipotesi non e'
        # garantita da niente: e' il caso in cui uno slice silenziosamente
        # diverso da quello vero darebbe un test verde falso. Qui si verifica
        # invece che il nucleo stia nello slice, e se non ci sta il confronto
        # viene saltato e detto.
        ids_full, probs_full = reference_nucleus(last, temp, tk, tp)
        allowed = {int(i) for i in ids_full}
        inside = allowed <= {int(i) for i in top_ids}
        if not inside:
            print(f"  info  {label}: nucleo di {len(allowed)} token, non sta "
                  f"tutti nei {VOCAB_SLICE} piu' probabili, confronto saltato")
            continue

        rng = np.random.default_rng(1234)
        drawn = np.array([sample(sliced, temperature=temp, top_k=tk, top_p=tp,
                                 rng=rng) for _ in range(DRAWS)])
        drawn_ids = {int(top_ids[i]) for i in drawn}

        outside = drawn_ids - allowed
        if outside:
            check(label, False,
                  f"{len(outside)} token fuori dal nucleo, es. "
                  f"{sorted(outside)[:3]}, nucleo di {len(allowed)}")
            continue

        want = probs_full * DRAWS
        # drawn contiene indici nello slice, quindi si riporta agli id globali
        # prima di contare: altrimenti si contano occorrenze di token che non
        # ci sono e lo scarto esce falso.
        got = np.bincount(np.asarray(top_ids)[drawn],
                          minlength=last.size)[np.asarray(ids_full)]
        # Tolleranza larga: 4000 estrazioni su un multinomiale, il rumore da
        # solo vale qualche percento del valore piu' probabile. Serve a
        # prendere un errore di normalizzazione, non a misurarlo.
        dev = float(np.abs(got - want).max() / DRAWS)
        check(label, dev < 0.02,
              f"nucleo {len(allowed)}, scarto massimo {dev * 100:.2f}%")

    # Con top_k = 0 il nucleo puo' essere molto piu' largo dello slice, quindi
    # questo caso si controlla sul vocabolario intero e con meno estrazioni:
    # un argsort da 248320 elementi costa ~25 ms, 400 estrazioni sono 10 s.
    # La frequenza non e' confrontabile qui (14423 candidati, 400 estrazioni:
    # ogni token esce zero o una volta), quindi la proprieta' che si controlla
    # e' la sola che conta: nessuna estrazione fuori dal nucleo.
    ids_full, _ = reference_nucleus(last, 1.0, 0, 0.99)
    allowed = {int(i) for i in ids_full}
    rng = np.random.default_rng(99)
    drawn = np.array([sample(last, temperature=1.0, top_k=0, top_p=0.99,
                             rng=rng) for _ in range(FULL_DRAWS)])
    outside = {int(t) for t in drawn} - allowed
    check("temperature 1.0, top_k 0, top_p 0.99 sul vocabolario intero",
          not outside,
          f"nucleo {len(allowed)}, {len(set(drawn.tolist()))} distinti su "
          f"{FULL_DRAWS}"
          + (f", fuori dal nucleo: {sorted(outside)[:3]}" if outside else ""))

    print("\n4. end-to-end su generate")
    # Collega il campionamento al resto: generate deve arrivare al token che
    # il golden test ha gia' fissato, altrimenti il percorso che l'utente
    # tocca e' diverso da quello verificato numericamente.
    out = model.generate(ids, max_new_tokens=1, temperature=0.0)
    check("generate greedy dà il token del riferimento",
          out[-1] == greedy, f"{out[-1]} e {greedy}")
    ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
    check("il token greedy è quello di llama.cpp",
          greedy == ref["final"]["argmax"],
          f"noi {greedy} llama.cpp {ref['final']['argmax']}")

    # Una generate da 6 token costa quanto 6 forward, quindi qui si fa il
    # minimo: una greedy da un token (che e' quella confrontata con
    # llama.cpp) e una coppia con seed per il determinismo.
    a = model.generate(ids, max_new_tokens=6, temperature=0.8, top_k=20,
                       top_p=0.95, seed=11)
    b = model.generate(ids, max_new_tokens=6, temperature=0.8, top_k=20,
                       top_p=0.95, seed=11)
    check("generate con lo stesso seed è riproducibile", a == b,
          f"{tok.decode(a[len(ids):])!r}")

    print(f"\n{'=' * 66}\n PASSATI: {ok}   FALLITI: {bad}\n{'=' * 66}")
    model.close()
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())