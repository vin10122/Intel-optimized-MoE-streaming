"""Interfaccia a riga di comando, beta minimale.

Esiste perche' il runner di benchmark non poteva fare il mestiere di un
client: alimentava il modello con ``np.random.randn`` e misurava quanto ci
metteva, quindi non c'era modo di vedere una risposta vera. Qui il prompt lo
scrivi tu, viene avvolto nel formato chat del GGUF e i token compaiono a uno
a uno, che e' il punto di tutto il progetto.

    python chat.py                        # conversazione
    python chat.py "Che cos'e' un MoE?"     # domanda singola in chat format
    python chat.py --cache 1 "..."        # con un solo expert in RAM

Su ``--cache``. Il default tiene 700 expert in RAM, e con un prompt di una
frase non si arriva a riempirli: la cache non viene mai interrogata, e il
disco sembra non servire. Con ``--cache 1`` ogni expert arriva dal GGUF e
ogni token rilegge tutto: e' piu' lento ed e' il modo di vedere davvero da
dove viene il peso. Il resto del runtime e' identico, e l'output e' lo stesso,
perche' la cache non cambia i numeri: cambia solo quante volte si rilegge.

Su ``--device``. Il default e' CPU. GPU e NPU sono selezionabili, ma le
misure end-to-end dipendono da memoria disponibile, cache e dispositivo.

Su ``--ssm-device``. Sposta le proiezioni grandi del DeltaNet su OpenVINO
CPU/GPU/NPU in f16; ricorrenza e stato restano in f32 su CPU. Senza opzione
il percorso resta NumPy/CPU.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# La console di Windows e' cp1252 e non sa scrivere i caratteri che il
# modello produce: senza questo la risposta muore a meta' frase, con
# UnicodeEncodeError, e sembra un crash del runtime.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gguf  # noqa: E402
from run_q4k_f16_openvino import dequant_to_f16  # noqa: E402
from flash_moe import Qwen3MoEModel, StreamingDetokenizer  # noqa: E402
from flash_moe.paths import model_path  # noqa: E402
from flash_moe.tokenizer import GGUFTokenizer  # noqa: E402


def encode_user_turn(tok: GGUFTokenizer, prompt: str) -> list[int]:
    """Codifica una domanda col formato chat del template GGUF."""
    formatted = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    return tok.encode(formatted, add_bos=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chat.py",
        description="Client a riga di comando per flash-moe.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("prompt", nargs="*",
                        help="domanda da porre. Vuota: conversazione.")
    parser.add_argument("--device", default="CPU",
                        choices=("CPU", "GPU", "NPU"),
                        help="dispositivo OpenVINO. Default CPU.")
    parser.add_argument("--ssm-device", choices=("CPU", "GPU", "NPU"),
                        help="dispositivo OpenVINO per le proiezioni DeltaNet. "
                             "Default: NumPy su CPU.")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="token massimi da generare. Default 64.")
    parser.add_argument("--temp", type=float, default=0.7,
                        help="temperatura del campionamento. 0 = greedy. "
                             "Default 0.7.")
    parser.add_argument("--top-k", type=int, default=20,
                        help="candidati tenuti prima di top_p. Default 20.")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="massa cumulativa del nucleo. Default 0.95.")
    parser.add_argument("--seed", type=int, default=42,
                        help="seme del campionamento. Default 42.")
    parser.add_argument("--cache", type=int, default=700,
                        help="expert tenuti in RAM. 1 per vedere ogni peso "
                             "arrivare dal disco. Default 700.")
    parser.add_argument("--max-seq", type=int, default=512,
                        help="lunghezza massima del contesto. Default 512.")
    parser.add_argument("--preload", action="store_true",
                        help="Pre-carica tutti gli expert in cache prima della generazione")
    parser.add_argument("--mmap", action="store_true",
                        help="Usa memory-mapped I/O per caricamento più veloce")
    return parser


def footer(model: Qwen3MoEModel, eos: bool) -> str:
    """Statistiche di I/O e cache per il turno."""
    stats = model.store.stats
    bits = []
    if eos:
        bits.append("fine perite")
    bits.append(f"{stats.bytes_read / 2**20:.0f} MiB letti dal disco")
    bits.append(f"{stats.misses} cache miss, {stats.prefetch_hits} prefetch")
    bits.append(f"{len(model.store.resident_keys())} expert in RAM")
    return " | " + " | ".join(bits)


def ask(model: Qwen3MoEModel, tok: GGUFTokenizer, prompt: str,
        args: argparse.Namespace) -> None:
    """Un turno: il prompt, poi i token appena sono pronti.

    Il testo passa da ``StreamingDetokenizer`` e non da ``decode`` sul blocco,
    perche' qui i token arrivano uno alla volta: un carattere UTF-8 puo' stare
    su piu' token e va completato prima di stampare, altrimenti in coda
    all'output spunta un segnaposto al posto di un carattere.
    """
    ids = encode_user_turn(tok, prompt)
    stream = StreamingDetokenizer(tok)
    n = 0
    eos = False
    sys.stdout.write("  ")
    for produced, token in model.stream(
        ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    ):
        n = produced
        eos = token == model.eos_token_id
        sys.stdout.write(stream.push(token))
        sys.stdout.flush()
    # flush(): se l'ultimo token porta l'inizio di un carattere multi-byte, il
    # resto e' gia' arrivato ma non e' stampabile finche' non si chiude lo
    # stream. Se il testo era troncato a meta' carattere, qui viene un
    # errore: e' meglio di mezzo carattere finto.
    sys.stdout.write(stream.flush())
    sys.stdout.flush()
    timings = model.last_stream_timings
    generation_seconds = timings["generation_seconds"]
    tok_s = (
        timings["generated_tokens"] / generation_seconds
        if generation_seconds > 0
        else float("nan")
    )
    print(
        f"\nprompt: {timings['prompt_tokens']} token, "
        f"prefill {timings['prefill_seconds']:.1f} s"
    )
    print(
        f"generazione: {n} token in {generation_seconds:.1f} s "
        f"({tok_s:.3f} tok/s; prefill escluso)"
    )
    print(f"{footer(model, eos)}\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.temp <= 0.0:
        # greedy: senza top_k e top_p che scatterebbero il risultato
        args.top_k, args.top_p = 0, 0.0

    path = model_path()
    if not path.exists():
        print(f"modello non trovato: {path}", flush=True)
        print("Imposta FLASH_MOE_GGUF per puntare altrove.", flush=True)
        return 1

    print(f"carico {path.name} su {args.device} "
          f"(cache {args.cache} expert; DeltaNet "
          f"{args.ssm_device or 'NumPy/CPU'}), regge un attimo...", flush=True)
    started = time.perf_counter()
    model = Qwen3MoEModel(path, device=args.device, max_seq_len=args.max_seq,
                          cache_capacity=args.cache, dequant_fn=dequant_to_f16,
                          ssm_device=args.ssm_device,
                          use_mmap=args.mmap)
    tok = GGUFTokenizer(gguf.GGUFReader(path))
    
    # Pre-caricamento expert se richiesto
    if args.preload:
        print(f"Pre-carico {args.cache} expert in cache...", flush=True)
        preload_start = time.perf_counter()
        # Preload ALL experts needed for all layers (40 layers * 8 experts)
        from flash_moe.key import ExpertKey
        num_experts = model.num_experts
        experts_per_layer = model.top_k  # Usually 8
        
        # Preload all layer expert combinations
        total_experts_to_load = min(args.cache, 40 * experts_per_layer)
        experts_loaded = 0
        
        for layer in range(40):
            if experts_loaded >= total_experts_to_load:
                break
            for expert_idx in range(experts_per_layer):
                if experts_loaded >= total_experts_to_load:
                    break
                key = ExpertKey(layer, expert_idx % num_experts)
                model.store.prefetch(key)
                experts_loaded += 1
        
        # Wait for prefetch to complete
        import time as time_module
        while len(model.store.pending_loads) > 0:
            time_module.sleep(0.1)
        preload_time = time.perf_counter() - preload_start
        print(f"Pre-caricamento completato in {preload_time:.1f}s. ", flush=True)
    
    print(f"modello pronto (inizializzazione e preparazione): "
          f"{time.perf_counter() - started:.1f} s. "
          "/ esce, vuoto esce.\n", flush=True)

    one_shot = " ".join(args.prompt).strip()
    if one_shot:
        print(f"> {one_shot}", flush=True)
        try:
            ask(model, tok, one_shot, args)
        except KeyboardInterrupt:
            print("\ninterrotto")
        model.close()
        return 0

    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt in ("/esce", "/exit", "/quit"):
            break
        if not prompt:
            continue
        try:
            ask(model, tok, prompt, args)
        except KeyboardInterrupt:
            # Il Ctrl-C durante un turno ferma la generazione e lascia il
            # prompt: la cache e lo stato del modello vengono azzerati dal
            # prefillo del turno dopo, quindi si puo' continuare.
            print("\ninterrotto")
        except Exception as error:  # noqa: BLE001
            print(f"\nerrore: {type(error).__name__}: {error}")

    model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())