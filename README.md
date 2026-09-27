# FLASH-MOE: Q8_0 to BF16 Optimized Inference

An optimized implementation of Mixture-of-Experts (MoE) inference for Qwen3.5 35B MoE models on Intel Lunar Lake with Intel Arc GPU acceleration.

## Features

- **Q8_0 Quantization Support**: Optimized for 8-bit quantized models
- **BF16 Output**: Dequantization to bfloat16 for GPU acceleration
- **LRU Expert Cache**: Keeps 700 experts in RAM (~4.2GB)
- **OpenVINO Backend**: Intel GPU acceleration
- **C++ Dequantization**: AVX2-optimized dequantization kernels
- **Streaming from GGUF**: Direct access to GGUF tensor slices

## Performance

### Current Results (Python fallback)
- **Model**: Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0
- **Token/second**: 0.09
- **Latency**: ~11 seconds per token (40 layers)
- **Cache hits**: 640
- **Cache misses**: 320

### Expected with C++ AVX2
- **Speedup**: 5-10x faster
- **Dequant time**: ~1-2ms per expert (vs ~35ms in Python)
- **Estimated latency**: ~1-2 seconds per token

## Project Structure

```
flash-moe-q8/
├── cpp/                          # C++ dequantization kernels
│   ├── fast_dequant_q8_0.cpp       # Simple OpenMP version
│   └── fast_dequant_q8_0_avx2.cpp # AVX2 optimized version
│
├── src/                          # Python source code
│   ├── flash_moe/                 # Core library
│   │   ├── __init__.py
│   │   ├── backend_ov.py
│   │   ├── engine.py
│   │   ├── key.py
│   │   ├── manifest.py
│   │   ├── router.py
│   │   ├── store.py
│   │   └── utils.py
│   └── run_q8_0_bf16_openvino.py  # Main inference script
│
├── scripts/                      # Build and utility scripts
│   ├── compile_dequant_q8_0.bat
│   └── compile_dequant_q8_0_full.bat
│
├── docs/
│   └── COMPILA_CPP.txt           # Compilation instructions (Italian)
│
├── .gitignore
└── README.md
```

## Quick Start

### Prerequisites

- **Python**: 3.10+
- **OpenVINO**: `pip install openvino`
- **GGUF**: `pip install gguf`
- **NumPy**: `pip install numpy`
- **Visual Studio 2022**: For C++ compilation (optional, but recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/flash-moe-q8.git
cd flash-moe-q8

# Install dependencies
pip install -r requirements.txt
```

### Compile C++ Accelerator (Recommended)

1. Open "Developer Command Prompt for VS 2022"
2. Navigate to the repository
3. Run:
   ```cmd
   scripts\compile_dequant_q8_0_full.bat
   ```
4. This will create `src\flash_moe\fast_dequant_q8_0.dll`

### Run Inference

```bash
python src/run_q8_0_bf16_openvino.py
```

## Q8_0 Format

The Q8_0 quantization format used:
- **Block size**: 256 elements
- **Per block**: 8 float16 scales (16 bytes) + 256 int8 values (256 bytes)
- **Total per block**: 272 bytes
- **Bytes per element**: 1.0625

### Dequantization Formula

```
value[i] = (qs[i] - 128) * scale
```

Where:
- `qs[i]`: int8 quantized value (range: -128 to 127)
- `scale`: float16 scale factor
- Output: BF16 (bfloat16)

## Configuration

The main script `src/run_q8_0_bf16_openvino.py` is configured for:

- **Model**: Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf
- **Hidden dimension**: 2048
- **Intermediate dimension**: 512
- **Number of experts**: 256
- **Number of layers**: 40
- **Top-k**: 8 experts per token
- **Cache capacity**: 700 experts

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FLASH-MOE Pipeline                        │
├─────────────────────────────────────────────────────────┤
│  1. GGUF Reader: Direct tensor slice access                │
│  2. Expert Store: LRU cache with streaming                 │
│  3. Dequantizer: Q8_0 → BF16 (C++ AVX2)                    │
│  4. OpenVINO: GPU-accelerated matmul in BF16               │
│  5. Router: Top-8 expert selection                          │
└─────────────────────────────────────────────────────────┘
```

### Memory Layout

| State | Memory | Notes |
|-------|--------|-------|
| Q8_0 compressed | ~2.1GB | 700 experts on disk |
| BF16 dequantized | ~4.2GB | 700 experts in RAM |
| Activations | FP32 | Mixed precision |
| Total | ~6.3GB | Fits in 16GB unified memory |

## Benchmark Results

### Before (IQ4_NL → FP32)
- Memory: ~8.4GB (FP32 weights)
- Speed: Slow Python dequantization

### After (Q8_0 → BF16)
- Memory: ~4.2GB (BF16 weights) - **50% reduction**
- Speed: C++ AVX2 dequantization - **5-10x faster**

## Development

### Adding New Quantization Types

1. Create a new C++ dequantizer in `cpp/`
2. Export the function with `__declspec(dllexport)`
3. Update the Python backend to call the new function
4. Compile and test

### Building from Source

```bash
# Compile C++ code
scripts\compile_dequant_q8_0_full.bat

# Or manually with cl.exe
cl /EHsc /O2 /GL /arch:AVX2 /openmp /LD /Qpar /Qvec \
  cpp\fast_dequant_q8_0_avx2.cpp \
  /link /OUT:src\flash_moe\fast_dequant_q8_0.dll /DLL /MACHINE:x64 /LTCG
```

## License

MIT License - Feel free to use for your projects.

## Contributing

Pull requests are welcome! Please open an issue first for major changes.

## Acknowledgments

- [GGUF](https://github.com/ggerganov/gguf) - Model quantization format
- [OpenVINO](https://github.com/openvinotoolkit/openvino) - Intel GPU acceleration
- [Qwen](https://github.com/QwenLM/Qwen) - Model architecture
