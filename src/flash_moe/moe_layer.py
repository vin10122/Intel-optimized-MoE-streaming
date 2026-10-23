"""Il calcolo di un layer MoE, in un posto solo.

Perche' questo modulo esiste
---------------------------
Il layer MoE era calcolato in due punti diversi: ``MoEEngine.process_token``
(engine.py) e ``TransformerBlock._moe_forward`` (transformer.py). Erano due
implementazioni dello stesso calcolo, e quando lo shared expert e' stato
aggiunto e' finito solo nella prima: ``TransformerBlock`` continuava a
produrre output sbagliati, senza errori.

Due copie dello stesso calcolo divergono, ed e' un difetto silenzioso: la
seconda continua a girare e a restituire numeri plausibili. Qui il calcolo
e' definito una volta sola e i due percorsi lo chiamano, cosi' non possono
divergere di nuovo.

Il percorso e' quello che il modello richiede:

    1. i top-k expert selezionati dal router del layer, ciascuno pesato dal
       proprio peso di routing
    2. il contributo dello shared expert, che NON e' tra gli k selezionati e
       che ogni token attraversa: si somma al totale, gia' scalato dal suo
       gate sigmoid, senza peso di routing

Nessuno dei due passi puo' mancare senza cambiare l'output di ogni token.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from .key import ExpertKey
from .router import ExpertRouter


def run_moe_layer(
    layer: int,
    activation: list[float] | np.ndarray,
    router: ExpertRouter,
    backend: Any,
    hidden_size: int,
    intermediate_size: int,
    shared_expert: Any = None,
) -> np.ndarray:
    """Output del layer MoE per un token: somma ponderata + shared expert.

    Restituisce un array f32 di ``hidden_size`` elementi. Solleva ``KeyError``
    se manca il router del layer: senza i gate non si puo' determinare quali
    expert eseguire, e indovinarli produrrebbe output plausibile e falso.
    """
    # 1. Top-k e pesi di routing del layer.
    expert_ids, weights = router.route(activation)
    keys = [ExpertKey(layer, expert_id) for expert_id in expert_ids]

    # 2. Accoda tutta la decisione di routing in un colpo: con piu' worker le
    # letture dei top-k girano in parallelo. Non e' un prefetch a un layer di
    # distanza, perche' quello non puo' esistere: E_{L+1} dipende dall'output
    # di E_L.
    store = getattr(backend, "store", None)
    prefetch = getattr(backend, "prefetch_weights", None)
    if prefetch is not None:
        prefetch(keys)
    elif store is not None and hasattr(store, "prefetch_many"):
        store.prefetch_many(keys)

    # 3. Esecuzione degli expert selezionati e somma pesata.
    combined = np.zeros(hidden_size, dtype=np.float32)
    if len(keys) == 1:
        key, weight = keys[0], weights[0]
        out = backend.forward(key, activation, hidden_size, intermediate_size)
        combined += weight * np.asarray(out, dtype=np.float32)
    else:
        executor = getattr(backend, "_moe_executor", None)
        own_executor = executor is None
        if own_executor:
            executor = ThreadPoolExecutor(max_workers=len(keys))
        try:
            futures = [executor.submit(
                backend.forward, key, activation, hidden_size, intermediate_size
            ) for key in keys]
            try:
                for future, weight in zip(futures, weights):
                    combined += weight * np.asarray(future.result(), dtype=np.float32)
            finally:
                # A failed expert must not leave other workers mutating caches
                # during the next user turn.
                for future in futures:
                    try:
                        future.result()
                    except Exception:
                        pass
        finally:
            if own_executor:
                executor.shutdown(wait=True)

    # 4. Shared expert: non e' fra i top-k e ogni token lo attraversa. Si
    # somma al totale senza peso di routing, perche' il suo peso e' gia' il
    # gate sigmoid interno.
    if shared_expert is not None:
        shared_out = shared_expert.forward(layer, activation)
        if shared_out is not None:
            combined = combined + shared_out

    return combined


def run_moe_sequence(
    layer: int,
    activations: np.ndarray,
    router: ExpertRouter,
    backend: Any,
    hidden_size: int,
    intermediate_size: int,
    shared_expert: Any = None,
) -> np.ndarray:
    """Run one MoE layer over a sequence, batching tokens by routed expert."""
    if activations.ndim != 2 or activations.shape[1] != hidden_size:
        raise ValueError(
            f"MoE activations must have shape (tokens, {hidden_size}), "
            f"got {activations.shape}"
        )
    forward_batch = getattr(backend, "forward_batch", None)
    if forward_batch is None:
        return np.stack([
            run_moe_layer(
                layer, activation, router, backend, hidden_size,
                intermediate_size, shared_expert,
            )
            for activation in activations
        ])

    token_count = activations.shape[0]
    routed: list[list[tuple[ExpertKey, float]]] = [
        [] for _ in range(token_count)
    ]
    grouped: dict[ExpertKey, list[tuple[int, int, float]]] = {}
    for token_index, activation in enumerate(activations):
        expert_ids, weights = router.route(activation)
        for slot, (expert_id, weight) in enumerate(zip(expert_ids, weights)):
            key = ExpertKey(layer, expert_id)
            routed[token_index].append((key, weight))
            grouped.setdefault(key, []).append((token_index, slot, weight))

    store = getattr(backend, "store", None)
    prefetch_many = getattr(backend, "prefetch_weights",
                            getattr(store, "prefetch_many", None))
    keys = list(grouped)
    batch_outputs = np.empty(
        (token_count, router.top_k, hidden_size), dtype=np.float32
    )
    lookahead = 8
    if prefetch_many is not None:
        prefetch_many(keys[:lookahead])
    for index, key in enumerate(keys):
        routes = grouped[key]
        token_indices = [token_index for token_index, _, _ in routes]
        if len(routes) == 1:
            outputs = np.asarray([backend.forward(
                key, activations[token_indices[0]], hidden_size,
                intermediate_size,
            )])
        else:
            outputs = forward_batch(
                key, activations[token_indices], hidden_size,
                intermediate_size,
            )
        for output_index, (token_index, slot, _) in enumerate(routes):
            batch_outputs[token_index, slot] = outputs[output_index]
        if prefetch_many is not None:
            prefetch_many(keys[index + 1:index + 1 + lookahead])

    result = np.zeros((token_count, hidden_size), dtype=np.float32)
    for token_index, routes in enumerate(routed):
        for slot, (_, weight) in enumerate(routes):
            result[token_index] += (
                weight * batch_outputs[token_index, slot]
            )
        if shared_expert is not None:
            shared_out = shared_expert.forward(
                layer, activations[token_index]
            )
            if shared_out is not None:
                result[token_index] += shared_out
    return result
