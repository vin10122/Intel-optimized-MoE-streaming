# INTEL-OPTIMIZED-MOE

**An experimental project for running MoE language models locally on Intel
hardware with OpenVINO.**

This project explores splitting inference work between the CPU and integrated
graphics, while loading only some of the model weights from disk at a time.
The goal is to explore what may be possible on a regular PC without a
high-memory discrete graphics card.

> **Status: work in progress.** The program has generated answers to real
> prompts, but speed, answer quality, and stability still need more testing.
> Expect long wait times, changes, and possible issues.

## Inspiration

This project was inspired by Apple's **Flash-MoE** app, as seen in
[this Instagram Reel](https://www.instagram.com/reel/DdqjqtAxAk6/). It is an
independent software project and is not affiliated with, endorsed by, or
sponsored by Apple.

## At a glance

- Reads models in GGUF format and lets you chat with them from PowerShell.
- Uses OpenVINO to run calculations on the CPU or a supported Intel device.
- The model uses a **Mixture of Experts (MoE)** architecture: for each step,
  it activates only some groups of model weights, called *experts*, rather
  than all of them.
- The command-line interface reports prompt processing time and generation
  speed separately.

## Status and results

The first model tested was **Hermes 3.6 35B A3B**, a Qwen3.6-MoE-based model
in GGUF format. The model file is not included in this repository; you need
to obtain it separately.

The current configuration combines **native quantized CPU decode with four
expert workers**, OpenVINO expert prefill with four CPU threads, batched
DeltaNet projections on the GPU, a sliding eight-expert prefetch window,
router weights promoted to FP32 once per layer, and native FP32 DeltaNet
recurrence during decode. The recurrence and its
state remain sequential FP32 operations. Raw expert cache: **320**; converted
weight cache: **1**. Native batched expert prefill was tested and excluded
because the complete prompt ran slower.

Manual runs used an **Intel Core Ultra 5 226V, integrated Arc 130V graphics,
15.7 GiB usable RAM, and Python 3.14 on Windows**:

| Power / request | Prompt tokens | Prefill | Generated tokens | Generation time | Generation speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Battery, first request | 88 | 57.0 s | 95 | 101.7 s | 0.934 tok/s |
| AC adapter, first request | 88 | 43.1 s | 95 | 82.8 s | 1.148 tok/s |
| AC adapter, repeated request in the same process | 88 | 22.8 s | 95 | 55.8 s | 1.702 tok/s |
| AC adapter, different question in the same process | 81 | 22.7 s | 94 | 56.1 s | 1.677 tok/s |

**Connecting the AC adapter improved the observed performance.** Comparing
the two first requests, prefill fell by about 24% and generation speed rose
by about 23%. These are observations from manual runs, not a controlled
power benchmark. Compilation, OS file cache, power settings, temperature,
and prompt routing also affect the results. The faster repeated request
benefits from warm-up as well; do not attribute its entire gain to AC power.
For comparisons, keep the power source and performance mode consistent and
report first-request and warmed-request timings separately.

The earlier measurements above came from the first temporary native-decode
configuration; the newer Q3_K/DeltaNet results are recorded below. Full-model measurements of this final
integration still need to be repeated on the target Windows PC. A native
decode comparison matched 16 greedy token IDs; four forced-input logits
checks had maximum relative error below 0.0006. The same 95-token answer
was observed with the optimized DeltaNet prefill. This is a numerical smoke
check, not proof of general answer-quality parity. The first generated token
comes from prefill logits: a one-token run does **not** measure decode speed.

### Q3_K and DeltaNet update: all new measurements on AC power

The latest tests below were **all run with the AC adapter connected**, on
that same Intel PC, using four expert workers, raw cache 320 and converted
cache 1. They used temporary launchers with the kernels now integrated here:

| Prompt / run | Prompt tokens | Prefill | Generated tokens | Generation time | CLI rate | Decode-step rate, approximately |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DAS explanation, first request with checks | 101 | 32.4 s | 64 | 31.9 s | 2.004 tok/s | 1.97 tok/s |
| Nuclear plant explanation, same session | 98 | 26.6 s | 64 | 31.8 s | 2.010 tok/s | 1.98 tok/s |
| Longer nuclear plant answer, new session | 117 | 32.2 s | 351 | 186.9 s | 1.878 tok/s | 1.87 tok/s |

The approximate decode-step rate uses `(generated_tokens - 1) / generation_time`,
because the first token comes from prefill. The timer also includes sampling
and, on the first request, temporary numerical checks/microbenchmarks. These
are manual observations with varying cache and thermal conditions; **stable
2+ tok/s has not been established**. The long test used a 512-token output
limit and stopped at EOS after 351 tokens. Only the first paragraph of the
proposed multiline prompt was submitted: it was a long answer, not a long
input-context test. The current interactive CLI submits one line at a time.

The Q3_K kernel extracts signed values before widening to AVX2 32-bit lanes
and uses constant shifts, retaining the original FMA accumulation order and
FP16 rounding. All tested Q3 expert samples from layers 5–34 matched the old
native kernel bit for bit. A same-expert Windows microbenchmark measured
1.502 to 0.750 ms (2.00x); another measured 1.390 to 0.607 ms (2.29x).
These single-expert ratios are **not** whole-model speedups.

The DeltaNet kernel processes all value heads in one native call, leaving
convolution, normalization, projection devices and token recurrence order
unchanged. Target-PC checks compared local outputs, recurrent state and
convolution state with the previous path, with observed maximum relative
errors below 5e-7. This does not prove general answer-quality parity.
The native recurrence is enabled only after prefill; OpenVINO/NumPy prefill
is retained. `--native-decode openvino` also retains the NumPy recurrence.

A larger raw cache of 640 was rejected: in one 32-token diagnostic run,
generation increased from 18.1 to 24.0 seconds. Cache misses, logical read
bytes and physical storage traffic are different quantities; zero demand
misses does not imply no prefetch waits or memory pressure.

Automated tests check individual components such as attention,
dequantization, and the DeltaNet path. They are not a full validation of
answer quality or reliability across different PCs.

## Requirements

- 64-bit Windows 10 or 11.
- Python 3.14 and the packages listed in `requirements.txt`. The tests used
  Python 3.14.7; other versions may work, but have not been checked with the
  same setup.
- A compatible GGUF model file. The model used in testing is described
  below.
- Up-to-date Intel drivers if you want to use integrated graphics.

From PowerShell, in the project directory, install the dependencies with:

```powershell
py -3.14 -m pip install -r requirements.txt
```

If the `py` command is not available, use `python` instead.

## Set up the model

If the GGUF file is in the project directory, the program will try to find it
automatically. Otherwise, set its path before starting the chat:

```powershell
$env:FLASH_MOE_GGUF = "D:\models\model.gguf"
```

This project **does not distribute the model**. Obtain it from an authorized
source and check the terms provided by its author. The model's license and
the code's license are separate.

### Source and attribution for the tested model

The closest public match to the filename used in testing is
[`LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-GGUF`](https://huggingface.co/LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-GGUF).
Its model card describes it as based on
[`HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive`](https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive),
with a Hermes contribution attributed to
[`DJLougen`](https://huggingface.co/DJLougen/hermes-qwen3.5-35b-a3b-GGUF).
The LuffyTheFox and HauhauCS model cards declare Apache-2.0 for their
respective repositories.

The file used in testing is named
`Hermes3.6-35B-A3B-Uncensored-Genesis-Final-APEX-Compact.gguf`, while the
corresponding variant listed in the LuffyTheFox repository omits `Final`
from its name. The match is likely, but this alone does not confirm that
both files are the same revision or contain the same weights. The model is
excluded from this project repository; before using or redistributing it,
verify the source and terms for the exact file you obtained.

## Start a chat

```powershell
py -3.14 chat.py default
```

Enter a prompt after the `>` symbol. To exit, type `/esce` or press
`Ctrl+C`.

The preset runs the main model on the CPU and uses the Intel GPU for DeltaNet
projections and full attention. If the GPU or its driver is unavailable, use
`--ssm-device CPU --attention-device CPU` instead, including after `default`.

The `default` preset selects CPU for the main device, GPU for DeltaNet and
full-attention projections, cache sizes of 320 and 1, deterministic
sampling, and a 128-token maximum. It uses native expert decode automatically
when the CPU and compiled DLL support it; otherwise it reports an OpenVINO
fallback. Explicit command-line flags override the preset. Append a prompt after `default` for a
one-shot run.

The assistant is instructed to keep its answers concise. `--max-tokens` is
a **maximum limit**, not a required answer length: generation stops earlier
if the model emits its end-of-response token. If it reaches the limit, the
answer may be incomplete; in that case, raise the limit further.

After each answer, the program reports generated tokens per second, prompt
processing time, and some cache statistics. Initial model loading may take
a long time. Logical read bytes and cache counters in the footer are
cumulative for the process; they do not measure physical SSD traffic. Each
question currently starts a fresh model context, retaining reusable caches.

### Useful options

- `--device CPU`, `GPU`, or `NPU`: selects the main OpenVINO device.
  Availability depends on your hardware and drivers.
- `--max-tokens 512`: sets the maximum number of tokens to generate per
  answer.
- `--cache 320`: limits how many quantized expert weight groups to keep in
  memory.
- `--weight-cache 1`: limits how many converted expert weight groups to
  keep in memory. The recommended value is 1 for the measured native-decode
  configuration; larger values can consume substantial RAM.
- `--temp 0`: uses deterministic token selection; higher values introduce
  more variation.
- `--max-seq 512`: sets the maximum supported context length. For a larger
  context, for example use `--max-seq 2048 --max-tokens 512`. This increases
  context capacity; it does not force the model to generate 512 tokens.

## Build the native decode kernel

Install Visual Studio Build Tools with the x64 C++ tools, then run from
PowerShell in the project root:

```powershell
scripts\compile_dequant_k.bat
scripts\compile_native_expert.bat
py -3.14 chat.py default --native-decode native
```

After updating the C++ sources, close running chats and rebuild with
`scripts\compile_native_expert.bat`; a DLL built before this update does not
contain the new Q3_K code or DeltaNet entry point. Recompilation is not
needed merely because you reopened PowerShell. Startup reports whether the
native DeltaNet entry point is present; older DLLs retain NumPy recurrence.

The first DLL accelerates weight conversion during prefill. The second
(`src/flash_moe/moe_quant_dot.dll`) reads quantized expert weights directly
during single-token decode, avoiding full FP16 expert materialization. It
requires AVX2, FMA and F16C and supports Q8_0, Q3_K, Q4_K, Q5_K and Q6_K.
The same native DLL also contains the FP32 DeltaNet recurrence kernel.
The build does not replace the weight-conversion DLL. Binaries and model
files are excluded from Git; each Windows installation builds its own DLLs.

`--native-decode native` requires the kernel and reports an error if it
cannot be used. The default `auto` falls back to OpenVINO if unavailable;
`--native-decode openvino` explicitly selects the reference expert path.
The prefill uses OpenVINO in all three modes. `--expert-workers 4` and
`--prefill-threads 4` expose the measured thread settings for other PCs.

## Optional C++ weight-conversion acceleration

On Windows, you can compile the C++ module used to convert some weight
formats:

```powershell
scripts\compile_dequant_k.bat
```

This requires Visual Studio Build Tools with the x64 C++ compiler. The
generated file is not included in the repository. If the DLL is unavailable,
the program uses a Python-based fallback, which may be slower but does not
require the DLL.

## Run some checks

Tests are Python scripts. For example, run these from the project root:

```powershell
py -3.14 tools\test_optimized_inference.py
py -3.14 tools\test_attention.py
py -3.14 tools\test_gated_deltanet_real.py
```

Some tests require the configured GGUF model and may take a long time. The
second test uses the configured model; set it up as described above first.
These checks cover individual components and do not replace a full chat
test.

Scripts that compare results with `llama.cpp` dumps look for
`lcpp_eval.txt` in the project directory by default. To use a dump elsewhere,
set `FLASH_MOE_LCPP_EVAL` to its path before running the script.

## Current limitations

- Testing has mainly been done on one Intel PC; performance and compatibility
  may differ on other hardware.
- Generation is slow compared with online services and systems with a
  discrete GPU.
- Performance depends heavily on model-file access and available memory.
- The concise-answer instruction can help limit answer length, but the
  model's behavior varies; the token limit can still cut off a long answer.
- This is not yet ready for production use.

## Code license

The code in this repository is distributed under the
[Apache-2.0 license](./LICENSE). This license applies to the project code,
not to external model weights or data.
