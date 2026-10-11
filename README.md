# FLASH-MOE

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

An initial manual test on an **Intel Core Ultra 5 226V with integrated Intel
Arc 130V graphics** produced understandable answers. Across the prompts
tested, generation speed ranged from about **0.2 to 0.8 tokens per second**.
These are indicative results, not a controlled benchmark: performance varies
with the prompt, answer length, and cache state.

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
py -3.14 chat.py --device GPU --ssm-device GPU --attention-device GPU --cache 320 --weight-cache 384 --max-tokens 256 --temp 0
```

Enter a prompt after the `>` symbol. To exit, type `/esce` or press
`Ctrl+C`.

This command selects the Intel GPU for some internal operations as well. If
the driver or device is unavailable, try omitting `--ssm-device GPU` and
`--attention-device GPU`, or use `--device CPU`.

The assistant is instructed to keep its answers concise. `--max-tokens` is
a **maximum limit**, not a required answer length: generation stops earlier
if the model emits its end-of-response token. If it reaches the limit, the
answer may be incomplete; in that case, raise the limit, for example with
`--max-tokens 384`.

After each answer, the program reports generated tokens per second, prompt
processing time, and some cache statistics. Initial model loading may take
a long time.

### Useful options

- `--device CPU`, `GPU`, or `NPU`: selects the main OpenVINO device.
  Availability depends on your hardware and drivers.
- `--max-tokens 256`: sets the maximum number of tokens to generate per
  answer.
- `--cache 320`: limits how many quantized expert weight groups to keep in
  memory.
- `--weight-cache 384`: limits how many converted expert weight groups to
  keep in memory. Lowering this value may help if the PC is short on RAM.
- `--temp 0`: uses deterministic token selection; higher values introduce
  more variation.
- `--max-seq 512`: sets the maximum supported context length.

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
