"""Interfaccia a riga di comando, beta minimale.

Esiste perche' il runner di benchmark non poteva fare il mestiere di un
client: alimentava il modello con ``np.random.randn`` e misurava quanto ci
metteva, quindi non c'era modo di vedere una risposta vera. Qui il prompt lo
scrivi tu, viene avvolto nel formato chat del GGUF e i token compaiono a uno
a uno, che e' il punto di tutto il progetto.

    python chat.py                        # conversazione
    python chat.py "Che cos'e' un MoE?"     # domanda singola in chat format
    python chat.py --cache 1 --weight-cache 1 "..."  # cache minima

Su ``--cache`` e ``--weight-cache``. La prima tiene in RAM i byte quantizzati
letti dal GGUF; la seconda i pesi f16 già convertiti. I default sono 320 raw
e 1 convertito. Il decode nativo legge i byte quantizzati direttamente e
non conserva pesi expert f16. Le cache cambiano il lavoro di lettura e
conversione, non i valori dei pesi.

Su ``--device``. Il default e' CPU. GPU e NPU sono selezionabili, ma le
misure end-to-end dipendono da memoria disponibile, cache e dispositivo.

Su ``--ssm-device``. Sposta le proiezioni grandi del DeltaNet su OpenVINO
CPU/GPU/NPU in f16; ricorrenza e stato restano in f32 su CPU. Default GPU.

Su ``--attention-device``. Sposta le proiezioni Q/K/V e output della full
attention su OpenVINO FP16. Default GPU.
"""
from __future__ import annotations

import argparse
import os
import io
import sys
import time
from contextlib import redirect_stdout
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

# Set BLAS policy before importing GGUF/NumPy. OpenVINO's explicit thread
# settings are separate. The override is local to this Python process.
_blas_threads = int(os.environ.get("FLASH_MOE_BLAS_THREADS", "1"))
if _blas_threads < 1:
    raise ValueError("FLASH_MOE_BLAS_THREADS must be positive")
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_variable] = str(_blas_threads)

import gguf  # noqa: E402
from run_q4k_f16_openvino import dequant_to_f16  # noqa: E402
from flash_moe import Qwen3MoEModel, StreamingDetokenizer  # noqa: E402
from flash_moe.paths import model_path  # noqa: E402
from flash_moe.progress import StartupProgress  # noqa: E402
from flash_moe.tokenizer import GGUFTokenizer  # noqa: E402


class LiveProfiler:
    """Misura le fasi principali e le sottofasi del backend durante lo stream."""

    def __init__(self, model: Qwen3MoEModel) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.prefill_totals: dict[str, float] = {}
        self.prefill_counts: dict[str, int] = {}
        self.enabled = False
        self.prefill_captured = False

        def wrap(obj, method_name: str, label: str,
                 classify=None) -> None:
            original = getattr(obj, method_name)

            def timed(*args, **kwargs):
                if not self.enabled:
                    return original(*args, **kwargs)
                if classify is not None:
                    classify(*args, **kwargs)
                started = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    self.totals[label] = self.totals.get(label, 0.0) + (
                        time.perf_counter() - started
                    )
                    self.counts[label] = self.counts.get(label, 0) + 1

            setattr(obj, method_name, timed)

        for block in model.blocks:
            if block.is_ssm:
                if block.ssm_runner is not None:
                    wrap(block, "ssm_runner", "DeltaNet")
            else:
                wrap(block, "_attention_forward", "attenzione full")
            wrap(block, "_moe_forward", "MoE totale")

        wrap(model.embedding, "forward", "embedding")
        wrap(model.output_norm, "forward", "output norm")
        wrap(model.lm_head, "forward", "lm_head")
        def classify_weight_cache(key, *_args, **_kwargs) -> None:
            label = ("weight-cache hit" if key in model.backend._weight_cache
                     else "weight-cache miss")
            self.counts[label] = self.counts.get(label, 0) + 1

        wrap(model.backend, "_weights", "pesi expert (sottofase MoE)",
             classify=classify_weight_cache)
        wrap(model.backend, "_decode", "dequant/view pesi (sottofase pesi)")
        wrap(model.store, "read_tensor", "lettura raw (sottofase pesi)")
        wrap(model.backend, "_infer", "expert OpenVINO (sottofase MoE)")
        wrap(model.backend, "_native_forward", "expert nativo (sottofase MoE)")
        if model.shared_expert is not None:
            wrap(model.shared_expert, "forward", "shared expert (sottofase MoE)")
            wrap(model.shared_expert, "forward_batch", "shared expert (sottofase MoE)")

        original_forward_all = model._forward_all

        def timed_forward_all(x):
            result = original_forward_all(x)
            if self.enabled and x.shape[1] > 1 and not self.prefill_captured:
                self.prefill_totals = self.totals.copy()
                self.prefill_counts = self.counts.copy()
                self.prefill_captured = True
                self.reset()
            return result

        model._forward_all = timed_forward_all

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()

    def report(self, elapsed: float, totals: dict[str, float] | None = None,
               counts: dict[str, int] | None = None) -> None:
        totals = self.totals if totals is None else totals
        counts = self.counts if counts is None else counts
        for label, seconds in sorted(totals.items(),
                                     key=lambda item: -item[1]):
            percent = seconds / elapsed * 100 if elapsed > 0 else 0.0
            print(f"    {label:38s} {seconds * 1000:8.1f} ms "
                  f"({percent:5.1f}%, {counts.get(label, 0)} chiamate)")
        phase_total = sum(
            totals.get(label, 0.0)
            for label in ("DeltaNet", "attenzione full", "MoE totale",
                          "embedding", "output norm", "lm_head")
        )
        other = max(0.0, elapsed - phase_total)
        print(f"    {'altro (norme, residui, sampling)':38s} "
              f"{other * 1000:8.1f} ms "
              f"({other / elapsed * 100 if elapsed > 0 else 0.0:5.1f}%)")
        if "weight-cache hit" in counts or "weight-cache miss" in counts:
            print("    Cache pesi convertiti: "
                  f"{counts.get('weight-cache hit', 0)} hit, "
                  f"{counts.get('weight-cache miss', 0)} miss")
        print("    Nota: lettura/dequant sono comprese nel percorso pesi; "
              "pesi, inferenza e shared sono comprese in MoE totale. "
              "Le durate dei worker paralleli possono sovrapporsi.")


def encode_user_turn(tok: GGUFTokenizer, prompt: str) -> list[int]:
    """Codifica una domanda con l'istruzione di risposta concisa."""
    formatted = (
        "<|im_start|>system\n"
        "Rispondi nella lingua dell'utente. Sii diretto e conciso: dai la "
        "risposta essenziale senza premesse o ripetizioni. Per domande "
        "semplici bastano una o poche frasi; aggiungi dettagli solo se "
        "servono a rispondere correttamente. Concludi appena hai risposto."
        "<|im_end|>\n"
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
                        help="domanda da porre. Vuota: conversazione. "
                             "Usa 'default' come primo argomento per la "
                             "configurazione CPU/GPU consigliata.")
    parser.add_argument("--prompt-file", type=Path,
                        help="Legge il prompt UTF-8 completo, incluse le righe multiple.")
    parser.add_argument("--device", default="CPU",
                        choices=("CPU", "GPU", "NPU"),
                        help="dispositivo OpenVINO. Default CPU.")
    parser.add_argument("--ssm-device", default="GPU", choices=("CPU", "GPU", "NPU"),
                        help="dispositivo OpenVINO per le proiezioni DeltaNet. "
                             "Default: GPU (con XMX).")
    parser.add_argument("--attention-device", default="GPU", choices=("CPU", "GPU", "NPU"),
                        help="dispositivo OpenVINO per le proiezioni full "
                             "attention FP16. Default: GPU (con XMX).")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="token massimi da generare. Default 128.")
    parser.add_argument("--temp", type=float, default=0.0,
                        help="temperatura del campionamento. 0 = greedy. "
                             "Default 0 (greedy).")
    parser.add_argument("--top-k", type=int, default=20,
                        help="candidati tenuti prima di top_p. Default 20.")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="massa cumulativa del nucleo. Default 0.95.")
    parser.add_argument("--seed", type=int, default=42,
                        help="seme del campionamento. Default 42.")
    parser.add_argument("--cache", type=int, default=320,
                        help="expert grezzi quantizzati tenuti in RAM. "
                             "Default 320.")
    parser.add_argument("--weight-cache", type=int, default=1,
                        help="pesi expert convertiti tenuti in RAM; "
                             "default 1. Imposta 1 insieme a --cache 1 "
                             "per riconvertire ogni volta.")
    parser.add_argument("--native-decode", choices=("auto", "native", "openvino"),
                        default="auto", help="Kernel expert CPU: auto usa la DLL "
                        "se disponibile; native la richiede; openvino la disabilita.")
    parser.add_argument("--expert-workers", type=int, default=4,
                        help="Worker expert nel decode. Default 4.")
    parser.add_argument("--prefill-threads", type=int, default=4,
                        help="Thread OpenVINO CPU per gruppi di token. Default 4.")
    parser.add_argument("--max-seq", type=int, default=512,
                        help="lunghezza massima del contesto. Default 512.")
    parser.add_argument("--preload", action="store_true",
                        help="Pre-carica tutti gli expert in cache prima della generazione")
    parser.add_argument("--mmap", action="store_true",
                        help="Usa memory-mapped I/O per caricamento più veloce")
    parser.add_argument("--profile-live", action="store_true",
                        help="Mostra latenza per token e tempi delle fasi "
                             "durante la generazione")
    parser.add_argument("--profile-startup", action="store_true",
                        help="Mostra la durata delle fasi di caricamento")
    return parser


def footer(model: Qwen3MoEModel, eos: bool, token_limit_reached: bool,
           prompt_tokens: int, prefill_seconds: float) -> str:
    """Statistiche di I/O e cache per il turno."""
    stats = model.store.stats
    bits = []
    if eos:
        bits.append("EOS")
    elif token_limit_reached:
        bits.append("limite token raggiunto")
    prefill_rate = (
        prompt_tokens / prefill_seconds if prefill_seconds > 0 else float("nan")
    )
    bits.append(
        f"prefill {prompt_tokens} token in {prefill_seconds:.1f} s "
        f"({prefill_rate:.2f} tok/s)"
    )
    bits.append(f"{stats.bytes_read / 2**20:.0f} MiB di letture logiche cumulative")
    bits.append(f"{stats.misses} cache miss, {stats.prefetch_hits} prefetch")
    bits.append(f"{len(model.store.resident_keys())} expert in RAM")
    return " | " + " | ".join(bits)


def ask(model: Qwen3MoEModel, tok: GGUFTokenizer, prompt: str,
        args: argparse.Namespace, profiler: LiveProfiler | None = None) -> None:
    """Un turno: il prompt, poi i token appena sono pronti.

    Il testo passa da ``StreamingDetokenizer`` e non da ``decode`` sul blocco,
    perche' qui i token arrivano uno alla volta: un carattere UTF-8 puo' stare
    su piu' token e va completato prima di stampare, altrimenti in coda
    all'output spunta un segnaposto al posto di un carattere.
    """
    ids = encode_user_turn(tok, prompt)
    if len(ids) + args.max_tokens - 1 > args.max_seq:
        raise ValueError(
            f"Prompt ({len(ids)} token) + output richiesto superano "
            f"--max-seq {args.max_seq}; aumenta il contesto o riduci l'output"
        )
    stream = StreamingDetokenizer(tok)
    n = 0
    eos = False
    previous_generation_seconds = 0.0
    if profiler is not None:
        profiler.reset()
        profiler.prefill_totals.clear()
        profiler.prefill_counts.clear()
        profiler.prefill_captured = False
        profiler.enabled = True
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
        if profiler is not None:
            timings = model.last_stream_timings
            generation_seconds = float(timings["generation_seconds"])
            step_seconds = generation_seconds - previous_generation_seconds
            previous_generation_seconds = generation_seconds
            rate = 1.0 / step_seconds if step_seconds > 0 else float("inf")
            if produced == 1:
                print(f"\n  [token 1] sampling dopo prefill: {step_seconds:.3f} s")
            else:
                print(f"\n  [token {produced}] {step_seconds:.3f} s "
                      f"({rate:.3f} tok/s)")
            if produced == 1:
                print(f"    prefill: {timings['prefill_seconds']:.3f} s "
                      "(misurato a parte)")
                if profiler.prefill_captured:
                    print("    fasi del prefill:")
                    profiler.report(
                        float(timings["prefill_seconds"]),
                        profiler.prefill_totals,
                        profiler.prefill_counts,
                    )
                # Il primo forward contiene il prompt; le fasi per token
                # iniziano dal decode successivo.
                profiler.reset()
            else:
                profiler.report(step_seconds)
                profiler.reset()
    # flush(): se l'ultimo token porta l'inizio di un carattere multi-byte, il
    # resto e' gia' arrivato ma non e' stampabile finche' non si chiude lo
    # stream. Se il testo era troncato a meta' carattere, qui viene un
    # errore: e' meglio di mezzo carattere finto.
    sys.stdout.write(stream.flush())
    sys.stdout.flush()
    timings = model.last_stream_timings
    if profiler is not None:
        profiler.enabled = False
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
    speed = f"{tok_s:.3f} tok/s" if n > 1 else "velocita decode non misurabile"
    print(
        f"generazione: {n} token in {generation_seconds:.1f} s "
        f"({speed}; prefill escluso)"
    )
    prompt_tokens = int(timings["prompt_tokens"])
    prefill_seconds = float(timings["prefill_seconds"])
    print(f"{footer(model, eos, n >= args.max_tokens and not eos,
                 prompt_tokens, prefill_seconds)}\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prompt and args.prompt[0].casefold() == "default":
        args.prompt = args.prompt[1:]
        # Parser defaults are the measured preset. Explicit flags win.
    if args.prompt_file is not None:
        if args.prompt:
            raise ValueError("Usa --prompt-file oppure il prompt posizionale")
        text = args.prompt_file.read_text(encoding="utf-8-sig").strip()
        if not text:
            raise ValueError("Il file del prompt e' vuoto")
        args.prompt = [text]
    if args.cache < 1:
        raise ValueError("--cache deve essere positivo")
    if args.weight_cache is not None and args.weight_cache < 1:
        raise ValueError("--weight-cache deve essere positivo")
    if min(args.expert_workers, args.prefill_threads) < 1:
        raise ValueError("worker e thread devono essere positivi")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens deve essere positivo")
    if args.temp <= 0.0:
        # greedy: senza top_k e top_p che scatterebbero il risultato
        args.top_k, args.top_p = 0, 0.0

    path = model_path()
    if not path.exists():
        print(f"modello non trovato: {path}", flush=True)
        print("Imposta FLASH_MOE_GGUF per puntare altrove.", flush=True)
        return 1

    console = sys.stdout
    startup_progress = StartupProgress(console)

    print("[vin10122] AI client optimized for integrated Intel hardware",
          flush=True)
    print("https://github.com/vin10122/MoE-ssd-streaming-for-openVINO",
          flush=True)
    print(f"Cache: {args.cache} raw experts, "
          f"{args.weight_cache} converted weights | DeltaNet: "
          f"{args.ssm_device or 'NumPy/CPU'} | Full attention: "
          f"{args.attention_device or 'NumPy/CPU'}", flush=True)
    startup_progress(0, f"Loading {path.name} on {args.device}")
    started = time.perf_counter()
    startup_logs = io.StringIO()
    with startup_progress, redirect_stdout(startup_logs):
        model = Qwen3MoEModel(
            path, device=args.device, max_seq_len=args.max_seq,
            cache_capacity=args.cache, dequant_fn=dequant_to_f16,
            weight_cache_capacity=args.weight_cache,
            native_decode=args.native_decode,
            expert_workers=args.expert_workers,
            prefill_threads=args.prefill_threads,
            ssm_device=args.ssm_device,
            attention_device=args.attention_device,
            use_mmap=args.mmap,
            profile_startup=args.profile_startup,
            startup_progress=startup_progress,
        )
        startup_stage = time.perf_counter()
        startup_progress(99, "Loading tokenizer")
        tok = GGUFTokenizer(model.reader)
        if args.profile_startup:
            print(f"[Startup] lettura tokenizer GGUF: "
                  f"{time.perf_counter() - startup_stage:.2f} s", flush=True)
        profiler = LiveProfiler(model) if args.profile_live else None

        # Pre-caricamento expert se richiesto
        if args.preload:
            startup_progress(99, f"Preloading up to {args.cache} experts")
            preload_start = time.perf_counter()
            from flash_moe.key import ExpertKey
            num_experts = model.num_experts
            experts_per_layer = model.top_k

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

            # Prefetch admission is intentionally bounded; consume each
            # selected expert rather than waiting for an unconsumed queue.
            remaining = min(args.cache, model.block_count * model.top_k)
            for layer in range(model.block_count):
                for expert_idx in range(model.top_k):
                    if remaining <= 0:
                        break
                    model.store.read_all(ExpertKey(layer, expert_idx % num_experts))
                    remaining -= 1
            preload_time = time.perf_counter() - preload_start
            print(f"Expert preload completed in {preload_time:.1f}s.",
                  flush=True)

    startup_progress(100, "Model ready")
    if startup_progress.animated:
        console.write("\n")
    console.write(startup_logs.getvalue())
    console.flush()
    print(f"Model initialization and preparation: "
          f"{time.perf_counter() - started:.1f} s. "
          "Type /exit to quit; submit an empty prompt to exit.\n", flush=True)

    one_shot = " ".join(args.prompt).strip()
    if one_shot:
        print(f"> {one_shot}", flush=True)
        try:
            ask(model, tok, one_shot, args, profiler)
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
            ask(model, tok, prompt, args, profiler)
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
