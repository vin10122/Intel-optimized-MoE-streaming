# Phase 0 - Diagnostica: cosa c'è davvero nel modello e dove vanno i ms

Data: 2026-09-27. Script: `bench_phases.py`, `tools/probe_gguf.py`,
`tools/check_ov_precisions.py`, `tools/check_ov_tensor_feeds.py`.
Output grezzo: `bench_phases_out.txt`.

Host: Windows 11, 8 CPU logici, Lunar Lake. OpenVINO 2026.4.0, numpy 2.5.3,
Python 3.14.7. Device OpenVINO disponibili: **CPU, GPU, NPU**.

---

## 1. Il modello non è quello che il codice assume

`tools/probe_gguf.py` sull'unico GGUF presente.

| Metadato | Valore |
|---|---|
| `general.architecture` | `qwen35moe` |
| `general.name` | `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive` |
| `block_count` | 40 |
| `embedding_length` | 2048 |
| `expert_count` | 256 |
| `expert_used_count` | 8 |
| dimensioni hardcoded nel codice | 2048 / 512 / 256 / 40 / top-8 |

I valori hardcoded sono corretti, ma per caso: vanno comunque letti dai metadata.

### 1a. E' un modello ibrido lineare-attention + MoE, non un transformer puro

30 dei 40 layer contengono tensori `ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`,
`ssm_dt`, `ssm_norm`, `ssm_out` (griglia Mamba / gated delta net). Solo 10 layer
(L3, L7, L11, L15, L19, L23, L27, L31, L35, L39) hanno `attn_q` / `attn_k` /
`attn_v` con `attn_q_norm` / `attn_k_norm`.

**Conseguenza:** un KV cache da solo non basta, serve anche lo stato ricorrente
SSM. Questo non riguarda il microbenchmark, ma riguarda l'obiettivo "client di
 testo".

### 1b. Esistono shared expert, e il codice li ignora

`ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp` (Q6_K) presenti in **tutti**
i 40 layer, sempre attivi, più il router `ffn_gate_inp_shexp`. Ogni layer MoE è
`shared_expert + 8 esperti routati`. Lo script attuale esegue solo i routati:
risultato numericamente sbagliato anche se fosse veloce.

### 1c. La quantizzazione NON è "Q4_K_M": per 30 layer su 40 e' Q3_K

Distribuzione byte effettiva (totale 16,23 GiB, 4,02 bit/param):

| Tipo | n tensor | Dimensione | % | bpw |
|---|---|---|---|---|
| **Q3_K** | 90 | 9,67 GiB | **59,6%** | 3,44 |
| Q4_K | 162 | 3,46 GiB | 21,3% | 4,50 |
| Q6_K | 156 | 2,51 GiB | 15,4% | 6,56 |
| Q8_0 | 6 | 306 MiB | 1,8% | 8,50 |
| Q5_K | 16 | 176 MiB | 1,1% | 5,50 |
| F32 | 301 | 84,7 MiB | 0,5% | 32,0 |
| F16 | 2 | 48 MiB | 0,3% | 16,0 |

Matrice layer -> tipo dei soli tensori MoE:

| Ruolo | Tipo | Layer |
|---|---|---|
| `ffn_gate_exps` / `ffn_up_exps` | Q4_K | L0-4, L35-39 (10) |
| `ffn_gate_exps` / `ffn_up_exps` | **Q3_K** | **L5-34 (30)** |
| `ffn_down_exps` | Q8_0 | L0 (1) |
| `ffn_down_exps` | Q6_K | L1-4, L35-39 (9) |
| `ffn_down_exps` | **Q3_K** | **L5-34 (30)** |
| `ffn_*_shexp` | Q6_K | L0-39 (40) |

Il tipo cambia **per layer**: un dequant "Q4_K" coprirebbe 10 layer su 40. Il
formato dominante è **Q3_K** (110 byte / 256 elementi), che è il più difficile
della famiglia K.

Verificato anche: `t.data_offset` in questa versione di `gguf` **è** l'offset
del singolo tensor (733 valori distinti, corretti), e lo slicing
`n_bytes // 256` produce slice allineate al blocco di quantizzazione per tutti i
tensor `*_exps` (`block_aligned=True` ovunque). La parte store del manifest è
sana.

---

## 2. bf16 non funziona: usa f16

`tools/check_ov_precisions.py` e `tools/check_ov_tensor_feeds.py`.

Passare pesi bf16 come `numpy` `uint16` (bit pattern) a una porta `ov.Type.bf16`
**non funziona**: i bit pattern vengono reinterpretati come interi e convertiti.

| Device | f32 | f16 via uint16 | bf16 via uint16 |
|---|---|---|---|
| CPU | err 1,8e-06 | err 6,0e+19 | **9 NaN** |
| GPU | err 4,5e-04 | **2048 NaN** | **2048 NaN** |
| NPU | err 1,0e-03 | err 6,9e+05 | compilazione fallita |

Il codice attuale fa esattamente questo: `dst = np.empty((rows, cols),
dtype=np.uint16)` + `ops.parameter(..., ov.Type.bf16)`. Produce NaN.

Il modo corretto e' **`numpy` `float16` nativo** su porta `ov.Type.f16`:

| Device | f16 nativo, errore relativo |
|---|---|
| CPU | 3,0e-04 |
| GPU | 3,8e-04 |
| NPU | 8,2e-04 |

Per bf16 servirebbe `ml_dtypes.bfloat16` (non installato). Dato che f16 funziona,
non serve: f16 ha anche 3 bit di mantissa in più di bf16, e per i pesi
dequantizzati non c'è rischio di overflow. **Decisione: f16.**

---

## 3. Il fallback Python attuale produce pesi tutti zero

`run_q8_0_bf16_openvino.py:188` e `src/run_q4k_bf16_openvino.py:157`:

```python
dequant = gguf.quants.dequantize(...).astype(np.float32)
return dequant.astype(np.uint16)     # troncamento a intero, NON reinterpretazione
```

Misurato: i pesi reali di un expert valgono `0 .. 0,0855`. Il cast produce
`0 .. 0`. **L'intera matrice di pesi è zero.** Ogni output degli script senza
DLL è quindi zero, e non solo impreciso.

Correzione (già verificata): errore relativo f16 vs fp32 = 3,46e-04.

---

## 4. Dove vanno i ms per expert (Q4_K, 1M elementi, GPU)

| Fase | ms/expert | Proiezione 320 esperti/token |
|---|---|---|
| Lettura slice dal file | 0,083 | 0,027 s |
| Dequant reference `gguf.quants` -> fp32 | **13,5** | 4,3 s |
| Cast fp32 -> uint16 (fallback attuale, rotto) | 1,40 | 0,45 s |
| Cast fp32 -> float16 (numpy) | 4,28 | 1,37 s |
| **DLL C++ esistente** (ordine di grandezza) | **0,32-0,44** | ~0,11 s |
| OpenVINO f16, pesi agli input, request riusato | 1,91 | 0,61 s |
| **OpenVINO f16, pesi = Constant** | **0,251** | **0,080 s** |

Proiezione per token (320 esperti = 8 x 40):

| Strategia OpenVINO | ms/expert | s/token | token/s |
|---|---|---|---|
| f32, pesi agli input | 4,25 | 1,360 | 0,74 |
| f32, pesi Constant | 0,486 | 0,156 | 6,43 |
| f16, pesi agli input | 1,91 | 0,611 | 1,64 |
| **f16, pesi Constant** | **0,251** | **0,080** | **12,47** |

Confronto device sul percorso migliore (f16 + Constant): GPU 0,356 ms, CPU
0,496 ms, NPU 0,561 ms. La GPU vince ma il margine è modesto: **l'overhead
OpenVINO non è trascurabile rispetto al compute**.

Stima end-to-end con lettura + dequant C++ + OpenVINO Constant:
**0,216 s/token = 4,6 token/s** per il solo FFN MoE, contro gli attuali
~5,3 s/token (0,19 token/s) e contro i ~0,19 ms/expert dichiarati dal commit
che sono semplicemente sbagliati.

Potenziale: **~24x**.

---

## 5. Cause dei risultati attuali

1. **Pesi passati come input a ogni `infer()`** → 1,91 ms/expert invece di
   0,251. È la singola causa più grande. Il fix è `Constant` / `share_weights`,
   non `create_infer_request()` (che costa solo 0,018 ms, irrilevante).
2. **Dequant di fallback rotto** → pesi zero.
3. **DLL C++ Q8_0** → numericamente falso (layout interleaved `d,qs[32],d,qs[32]`
   dichiarato come `scales[8],qs[256]`, e offset `-128` inesistente: Q8_0 è
   `y = qs[j] * d`, senza offset).
4. **bf16 via uint16** → NaN su GPU.
5. **`AsyncExpertStore.prefetch`** (`flash_moe/store.py:116`) fa
   `path.read_bytes()` sul GGUF da 16,24 GiB invece di leggere la slice.
6. **Collisione tipi in `_cache`**: `get()` scrive `bytes` (riga 126),
   `read_tensor()` scrive `dict` (riga 76) sulla stessa chiave. `MoEEngine`
   chiama `get()` prima di `backend.forward()` (`engine.py:41`) →
   `TypeError`.
7. **Router a pesi random** (`router.py:11`) → statistiche cache non
   rappresentative.
8. **Shared expert mancante** (§1b) e **stato SSM mancante** (§1a).

---

## 6. Ordine di lavoro rivisto alla luce dei dati

| # | Azione | Gain atteso |
|---|---|---|
| 1 | OpenVINO: pesi come `Constant` / `share_weights`, tipo **f16** | 7,6x |
| 2 | Correggere il fallback (o eliminarlo: fallire è meglio che tornare zeri) | correttezza |
| 3 | Correggere lo store: slice nel prefetch, un solo tipo in `_cache` | correttezza |
| 4 | Aggiungere shared expert al path di esecuzione | correttezza |
| 5 | Dequant C++ per **Q3_K** (non Q4_K) + Q4_K + Q6_K, con gate numerico | ~40x sul dequant |
| 6 | Batch degli 8 esperti per layer (2 GEMM invece di 8) | da valutare dopo 1 |
| 7 | Stato SSM + attention se l'obiettivo è un client di testo | necessario per l'obiettivo |

Il dequant non è più il collo di bottiglia *relativo* dopo il punto 1, ma resta
necessario: senza un kernel C++ per Q3_K il costo per expert è 13,5 ms
(dequant) contro 0,25 ms (OpenVINO), cioè il dequant diventa il 98%.

### Gate numerico obbligatorio per i kernel C++

Nessun kernel entra nel build senza assert su tensor reale:

```python
ref = gguf.quants.dequantize(raw, type)                  # reference llama.cpp
assert np.abs(out.astype(np.float32) - ref).max() < 1e-3
```

Il kernel Q8_0 attuale non ha mai avuto questo test, ed è per questo che è
"più veloce e sbagliato".
