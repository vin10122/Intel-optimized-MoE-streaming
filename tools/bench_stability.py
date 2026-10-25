"""Temporary long-input benchmark with decode rates per 64 steps."""
from pathlib import Path
import argparse
import math
import statistics
import sys


def summarize(times, width=64):
    if not times:
        print('[STABILITY] No decode steps: first token belongs to prefill.')
        return
    for start in range(0, len(times), width):
        part = times[start:start+width]
        label = 'full window' if len(part) == width else 'partial window'
        print(f'[STABILITY] decode steps {start+1}-{start+len(part)}: '
              f'{len(part)/sum(part):.3f} tok/s ({label})')
    ordered = sorted(times)
    p95 = ordered[max(0, math.ceil(.95*len(ordered))-1)]
    print(f'[STABILITY] total={len(times)} decode steps; '
          f'rate={len(times)/sum(times):.3f} tok/s; '
          f'median={statistics.median(times)*1000:.1f} ms; p95={p95*1000:.1f} ms')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt-file', type=Path, required=True)
    options, rest = parser.parse_known_args()
    prompt = options.prompt_file.read_text(encoding='utf-8-sig').strip()
    if not prompt:
        parser.error('prompt file is empty')
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root/'src'), str(root)]
    import chat
    from flash_moe.model import Qwen3MoEModel
    original_stream = Qwen3MoEModel.stream
    original_ask = chat.ask
    samples = []

    def stream(model, *args, **kwargs):
        previous = 0.
        for produced, token in original_stream(model, *args, **kwargs):
            total = float(model.last_stream_timings['generation_seconds'])
            if produced > 1:
                samples.append(total-previous)
            previous = total
            yield produced, token

    def ask(model, tok, text, args, profiler=None):
        count = len(chat.encode_user_turn(tok, text))
        if count + args.max_tokens - 1 > args.max_seq:
            raise ValueError(f'Input={count} tokens; input + output budget '
                             f'exceeds --max-seq {args.max_seq}')
        samples.clear()
        try:
            return original_ask(model, tok, text, args, profiler)
        finally:
            summarize(samples)

    Qwen3MoEModel.stream = stream
    chat.ask = ask
    raise SystemExit(chat.main([prompt] + (rest[1:] if rest and rest[0].casefold() == 'default' else rest)))


if __name__ == '__main__':
    main()
