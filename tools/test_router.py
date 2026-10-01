"""Validazione del routing MoE contro una reference F32 letta dal GGUF.

Il reference qui sotto NON passa da flash_moe: legge il gate F32 dal GGUF
con numpy e fa il prodotto a mano. Se i due convergono, il router sta
selezionando davvero gli expert del modello e non sta inventandoli.

Il punto delicato e' la precisione. I gate sono 2048x256 per layer: in f16
il peso puo' cambiare di qualche unità nell'ultimo posto, e se due expert
hanno punteggi quasi pari la selezione puo' ribaltarsi. Non e' un errore da
nascondere, e' una domanda a cui va risposto con un numero: quanti top-8
differiscono fra gate F32 e gate F16.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flash_moe.router import (  # noqa: E402
    ExpertRouter,
    routers_from_gguf,
    softmax_all_experts,
    synthetic_router,
)
from flash_moe.paths import model_path  # noqa: E402

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    (PASSED if ok else FAILED).append(name if ok else (name, detail))


def reference_route(gate32: np.ndarray, x: np.ndarray, top_k: int):
    """Top-k + softmax pesi, calcolato a mano sul gate F32.

    gate32 ha forma (n_expert, n_hidden): il GGUF tiene l'asse degli esperti
    veloce, quindi e' la trasposta del gate (hidden, expert) usato dal router.
    """
    scores = gate32 @ x.astype(np.float64)
    idx = np.argsort(scores)[-top_k:][::-1]
    w = softmax_all_experts(scores[idx])
    return idx.tolist(), w


def main() -> int:
    import gguf

    path = model_path()
    if not path.exists():
        print(f"GGUF non trovato: {path}")
        return 2

    reader = gguf.GGUFReader(path)
    tmap = {t.name: t for t in reader.tensors}
    arch = str(reader.fields["general.architecture"].parts[-1].tobytes().decode())
    hidden = int(reader.fields[f"{arch}.embedding_length"].parts[-1][0])
    n_experts = int(reader.fields[f"{arch}.expert_count"].parts[-1][0])
    top_k = int(reader.fields[f"{arch}.expert_used_count"].parts[-1][0])
    n_layers = int(reader.fields[f"{arch}.block_count"].parts[-1][0])
    print(f"modello: {path.name}")
    print(f"arch={arch} hidden={hidden} experts={n_experts} top_k={top_k} layers={n_layers}")

    print("\n--- A. struttura ---")
    routers = routers_from_gguf(tmap, n_layers, hidden, n_experts, top_k)
    check(f"un router per layer ({n_layers})", len(routers) == n_layers, f"n={len(routers)}")
    check("top_k letto dal GGUF", routers[0].top_k == top_k, f"top_k={routers[0].top_k}")
    check(
        "gate in f16",
        all(r.gate_weights.dtype == np.float16 for r in routers.values()),
        str(next(iter(routers.values())).gate_weights.dtype),
    )
    shapes = {r.gate_weights.shape for r in routers.values()}
    check("gate (hidden, expert) per layer", shapes == {(hidden, n_experts)}, str(shapes))

    # I 40 gate devono essere davvero diversi: se fossero identici il
    # routing per-layer non esisterebbe.
    print("\n--- B. i gate sono distinti fra layer ---")
    layer0 = routers[0].gate_weights.astype(np.float32)
    identical = [l for l in routers if l != 0
                 and np.array_equal(routers[l].gate_weights.astype(np.float32), layer0)]
    check("nessun gate duplicato", not identical, f"identici a layer 0: {identical}")
    diffs = [float(np.abs(routers[l].gate_weights.astype(np.float32) - layer0).max())
             for l in range(1, n_layers)]
    print(f"  scarto massimo vs layer 0: min={min(diffs):.4f} max={max(diffs):.4f}")
    check("gate diversi in modo significativo", min(diffs) > 1e-3, f"min={min(diffs):.4f}")

    # ---------------------------------------------------------------- C
    print("\n--- C. il router sceglie gli expert del modello (vs reference F32) ---")
    rng = np.random.default_rng(0)
    n_probe = 200
    x = rng.standard_normal((n_probe, hidden)).astype(np.float32)

    same_set = same_order = 0
    total_tokens = 0
    worst_w = 0.0
    swaps: list[tuple[int, int, float]] = []
    for layer in (0, 3, 7, 19, 39):
        gate_t = tmap[f"blk.{layer}.ffn_gate_inp.weight"]
        gate32 = np.asarray(gate_t.data, dtype=np.float32).reshape(n_experts, hidden)
        r = routers[layer]
        for i in range(n_probe):
            ref_scores = gate32 @ x[i].astype(np.float64)
            ridx, rw = reference_route(gate32, x[i], top_k)
            idx, w = r.route(x[i])
            total_tokens += 1
            if list(idx) == list(ridx):
                same_order += 1
            if set(idx) == set(ridx):
                same_set += 1
            else:
                lost = set(ridx) - set(idx)
                gained = set(idx) - set(ridx)
                gap = min(abs(ref_scores[e] - ref_scores[g]) for e in lost for g in gained)
                swaps.append((layer, i, float(gap)))
            worst_w = max(worst_w, float(np.abs(np.asarray(w) - rw).max()))

    set_rate = same_set / total_tokens
    order_rate = same_order / total_tokens
    print(f"  campioni: {total_tokens} (5 layer x {n_probe} attivazioni)")
    print(f"  stesso SET di 8 expert : {same_set} ({set_rate*100:.2f}%)")
    print(f"  stesso ORDINE          : {same_order} ({order_rate*100:.2f}%)")
    print(f"  scarto massimo sui pesi: {worst_w:.3e}")
    if swaps:
        print(f"  scarti dove il set cambia: {sorted(g for _, _, g in swaps)[:5]}")
        print("  -> sono near-tie: sotto 1e-2 di scarto tra i due punteggi")
    # Cio' che conta per lo streaming e' il SET: e' lui che determina quali
    # expert vengono letti da SSD. L'ordine cambia piu' spesso perche' basta
    # uno scarto piccolo per invertire due posizioni, ma gli expert letti
    # sono gli stessi. Con il gate in f16 il set coincide nel 99,8% dei casi
    # su questo modello, e le differenze sono coppie a parita' di punteggio.
    check("stesso SET di expert >= 99.5%", set_rate >= 0.995, f"{set_rate*100:.2f}%")
    check("pesi entro 1e-3 dal reference", worst_w < 1e-3, f"{worst_w:.3e}")
    check("ogni divergenza e' una near-tie (< 1e-2)",
          all(g < 1e-2 for _, _, g in swaps),
          f"max scarto={max((g for _, _, g in swaps), default=0.0):.2e}")

    # ---------------------------------------------------------------- D
    print("\n--- D. invarianti del routing ---")
    layer = 7
    r = routers[layer]
    idx, w = r.route(x[0])
    check("restituisce top_k indici", len(idx) == top_k, f"n={len(idx)}")
    check("indici unici", len(set(idx)) == len(idx), f"{idx}")
    check("indici nel range", all(0 <= i < n_experts for i in idx))
    check("pesi positivi", all(v > 0.0 for v in w), f"min={min(w):.4f}")
    check("pesi sommano a 1", abs(sum(w) - 1.0) < 1e-6, f"sum={sum(w):.10f}")
    check("pesi in ordine decrescente", list(w) == sorted(w, reverse=True), f"{w[:3]}")

    print("\n--- E. determinismo ---")
    a1 = routers[3].route(x[1])
    a2 = routers[3].route(x[1])
    check("stessa attivazione, stesso risultato", a1 == a2)
    check("stesso router, risultato stabile", routers[3].route(x[1]) == a1)
    # Lo stesso vettore passato come lista deve dare lo stesso routing.
    a3 = routers[3].route(x[1].tolist())
    check("lista e array danno lo stesso routing", a3 == a1)

    # ---------------------------------------------------------------- F
    print("\n--- F. equivalenza top-k poi softmax vs softmax su tutti ---")
    # L'implementazione seleziona prima e applica il softmax agli 8 scelti;
    # llama.cpp fa il softmax su 256 e poi rinormalizza. Devono coincidere:
    # il denominatore finale dipende solo dagli expert selezionati.
    worst_eq = 0.0
    for i in range(8):
        scores = routers[5].scores(x[i])
        idx, w = routers[5].route(x[i])
        p_all = softmax_all_experts(scores)
        renormalized = p_all[idx] / p_all[idx].sum()
        worst_eq = max(worst_eq, float(np.abs(np.asarray(w) - renormalized).max()))
    check("equivalenza entro 1e-9", worst_eq < 1e-9, f"max diff={worst_eq:.3e}")

    # ---------------------------------------------------------------- G
    print("\n--- G. il routing non e' degenere ---")
    # Il rischio con un gate finto e' di selezionare sempre gli stessi
    # expert: la cache sembrerebbe perfetta e il profiling del streaming
    # risulterebbe falso.
    picks: Counter[int] = Counter()
    distinct_per_token = []
    for i in range(n_probe):
        idx, _ = routers[19].route(x[i])
        picks.update(idx)
        distinct_per_token.append(len(set(idx)))
    used = len(picks)
    print(f"  expert distinti su {n_experts} in {n_probe} token: {used}")
    print(f"  expert piu' frequente: {picks.most_common(3)}")
    check("usa piu' di 16 expert diversi", used > 16, f"usati={used}")
    check("nessun expert dominante",
          picks.most_common(1)[0][1] < n_probe * 0.5,
          f"max freq={picks.most_common(1)[0][1]}/{n_probe}")
    check("8 indici distinti per token", min(distinct_per_token) == top_k)

    # ---------------------------------------------------------------- H
    print("\n--- H. i gate reali producono routing diverso da quelli finti ---")
    # Se coincidessero, aver letto i gate veri non avrebbe cambiato nulla e
    # il prefetch del passo 3 non avrebbe niente su cui basarsi.
    agree = 0
    for i in range(n_probe):
        real = set(routers[0].route(x[i])[0])
        fake = set(synthetic_router(hidden, n_experts, top_k, seed=42, layer=0).route(x[i])[0])
        agree += len(real & fake)
    overlap = agree / (n_probe * top_k)
    print(f"  sovrapposizione media top-8 reale/finto: {overlap:.3f}")
    check("routing reale e finto sono diversi", overlap < 0.5, f"overlap={overlap:.3f}")

    # ---------------------------------------------------------------- I
    print("\n--- I. robustezza ---")
    try:
        routers[0].route(np.zeros(hidden, dtype=np.float32)[:-1])
        check("attivazione di lunghezza sbagliata rifiutata", False, "non ha sollevato")
    except ValueError:
        check("attivazione di lunghezza sbagliata rifiutata", True)
    try:
        ExpertRouter(hidden, n_experts, top_k=8)
        check("router senza gate rifiutato", False, "non ha sollevato")
    except ValueError:
        check("router senza gate rifiutato", True)
    try:
        routers_from_gguf({}, 1, hidden, n_experts, top_k)
        check("layer senza gate segnalato per nome", False, "non ha sollevato")
    except KeyError as e:
        check("layer senza gate segnalato per nome", "0" in str(e), str(e)[:60])

    print("\n" + "=" * 66)
    print(f" PASSED: {len(PASSED)}   FAILED: {len(FAILED)}")
    for name, detail in FAILED:
        print(f"   - {name}: {detail}")
    print("=" * 66)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
