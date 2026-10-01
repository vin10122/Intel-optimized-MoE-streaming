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
    if store is not None and hasattr(store, "prefetch_many"):
        store.prefetch_many(keys)

    # 3. Esecuzione degli expert selezionati e somma pesata.
    combined = np.zeros(hidden_size, dtype=np.float32)
    for key, weight in zip(keys, weights):
        # No store.get() here: concatenerebbe l'intero expert per buttarlo
        # via. backend.forward() -> store.read_tensor() prende il path del
        # prefetch e tira solo il tensor che serve.
        out = backend.forward(key, activation, hidden_size, intermediate_size)
        combined += weight * np.asarray(out, dtype=np.float32)

    # 4. Shared expert: non e' fra i top-k e ogni token lo attraversa. Si
    # somma al totale senza peso di routing, perche' il suo peso e' gia' il
    # gate sigmoid interno.
    if shared_expert is not None:
        shared_out = shared_expert.forward(layer, activation)
        if shared_out is not None:
            combined = combined + shared_out

    return combined