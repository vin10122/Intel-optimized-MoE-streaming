# FLASH-MOE

Inferenza MoE ibrida lineare-attention/MoE su **Intel Arc 130V** (Xe2 LPG, integrata) via OpenVINO.

Il modello di riferimento è un GGUF `qwen35moe` da 16,24 GiB, più grande della
RAM di sistema: l'intero approccio è **streaming degli expert** — in RAM stanno
i pesi non-expert e gli 8 expert attivi del layer corrente, il resto viene
letto dal GGUF su richiesta.

## Stato

| Componente | Stato |
|---|---|
| Dequantizzazione K-quant in C++ (Q8_0, Q3_K, Q4_K, Q5_K, Q6_K) | **validato**, gate 162/162 |
| Precisione OpenVINO: `float16` nativo su porte a 16 bit | **validato** |
| Slicing degli expert dal GGUF | letto e offset verificati |
| **Inferenza end-to-end** (OpenVINO vs NumPy) | **non ancora validata** |
| `AsyncExpertStore` / `build_manifest_from_gguf` | mai esercitati |

Il gate numerico copre i kernel e la precisione, **non** l'inference. Nessun
token è stato ancora generato e validato end-to-end.

## Modello

```
Hermes3.6-35B-A3B-Uncensored-Genesis-Final-APEX-Compact.gguf
architettura  qwen35moe      40 layer      hidden 2048
expert        256 (8 attivi/token)         intermediate 512
```

Il percorso del modello vive in un solo posto: `src/flash_moe/paths.py`.
Per puntare a un altro GGUF senza toccare il codice:

```powershell
$env:FLASH_MOE_GGUF = "D:\modelli\altro.gguf"
```

## Struttura

```
cpp/                          kernel C++ K-quant
  dequant_k.cpp                 dequant_k, dequant_k_supported, dequant_k_block_size
  dequant_k_tables.h            generato: mappature ql/qh misurate
src/flash_moe/                 unica copia del pacchetto
  paths.py                      modello e DLL, con override via env
  dequant.py                    dequantizzazione GGUF (f32 / f16)
  store.py  router.py  engine.py  backend_ov.py
src/run_q4k_f16_openvino.py   runner expert singolo, SwiGLU in float16
tools/
  test_dequant_k.py             il gate numerico (162/162)
  derive_k_tables.py            misura le mappature di interlacciamento
  test_phase1.py                dimostra che la vecchia DLL era errata
scripts/compile_dequant_k.bat  build MSVC della DLL
docs/PHASE0_FINDINGS.md         rilievi di Fase 0
```

## Costruire la DLL

```powershell
scripts\compile_dequant_k.bat
```

Produce `src/flash_moe/fast_dequant_k.dll` (AVX2 + F16C). La DLL non è nel
repository: si ricostruisce.

## Verificare

```powershell
python tools\test_dequant_k.py
```

Confronta ogni kernel contro `gguf.quants` — la stessa reference di llama.cpp —
bit per bit sul primo blocco e con tolleranza `1e-6` sull'intero, su blocchi
sintetici e su 12 tensori reali del GGUF. Attesi **162 passati, 0 falliti**.

## Tre errori che è costato imparare

Sono qui perché sono tutti non ovvi e tutti hanno prodotto numeri plausibili
invece che errori evidenti.

**1. `astype(np.uint16)` sui pesi dequantizzati produce una matrice di zeri.**
I pesi di un expert valgono circa `0..0,0855`, quindi troncarli a intero dà
tutto zero. Non è un'approssimazione grossolana: sono pesi nulli. Il cast
corretto è reinterpretare i bit in `float16`.

**2. `bf16` non funziona su questa piattaforma.** Passare `numpy.uint16` su
porte a 16 bit produce `NaN`. Il metodo che regge è `numpy.float16` nativo su
`ov.Type.f16` (vedi `tools/check_ov_precisions.py`).

**3. Lo stesso ruolo expert non ha sempre lo stesso tipo quantizzato.**
`ffn_gate_exps` e `ffn_up_exps` sono `Q4_K`, ma `ffn_down_exps` è `Q8_0`.
Dequantizzarli tutti con lo stesso tipo non dà errore, dà pesi sbagliati. I
tipi vanno letti dal GGUF.

## Formato Q8_0 corretto

La documentazione precedente riportava `8 scale float16 + 256 int8 = 272 B` e
`y = (qs - 128) * scale`. **Entrambe le cose sono sbagliate** ed erano il
formato della DLL rotta. Il layout reale:

```
34 byte per blocco di 32 elementi
  offset 0   : f16 d
  offset 2   : int8 qs[32]
  y[i] = qs[i] * d
```

Non c'è `dmin` e non c'è l'offset `-128`. Lo stesso vale per Q3_K, che non ha
`dmin` e ha `d` a offset 108.

## Prestazioni attese

Per token si leggono ~1,66 miliardi di parametri attivi (992M expert + 671M
attenzione) in `float16`, cioè ~3,33 GB. Con ~140 GB/s di banda il pavimento è
~24 ms/token. A cache expert fredda il collo di bottiglia diventa l'NVMe:
~734 MB/token, ~105 ms con Gen4. Entrambi i regimi sono usabili, quindi la
cache non è bloccante per partire.

Nessuna misura end-to-end è stata ancora presa: sono stime based sui numeri
sopra, non risultati.

## Note

Le DLL in `.gitignore` e il GGUF non sono versionati. `run_q8_0_bf16_openvino.py`
è stato rimosso: puntava a un modello diverso e alla DLL errata.
