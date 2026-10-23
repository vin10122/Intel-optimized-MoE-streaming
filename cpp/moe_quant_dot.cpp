// Fused quantized expert for single-token decode.
// Reuse the validated block decoders; build separately from weight conversion.
#include "dequant_k.cpp"
#include <cmath>
#include <vector>
#include <limits>

static inline float round16(float x) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtps_ph(_mm_set_ss(x), 0)));
}

static inline bool decode_block(int type, const uint8_t* src, float* tmp) {
    switch (type) {
        case GGML_TYPE_Q8_0: dequant_q8_0(src, tmp, 1); return true;
        case GGML_TYPE_Q3_K: dequant_q3_k(src, tmp, 1); return true;
        case GGML_TYPE_Q4_K: dequant_q4_k(src, tmp, 1); return true;
        case GGML_TYPE_Q5_K: dequant_q5_k(src, tmp, 1); return true;
        case GGML_TYPE_Q6_K: dequant_q6_k(src, tmp, 1); return true;
        default: return false;
    }
}

// Q3_K layout follows gguf.quants.Q3_K.dequantize_blocks: eight groups
// of 32, each split into two scale groups of 16. Decode directly in AVX2.
static inline __m256 q3_dot_block(const uint8_t* blk, const float* x,
                                  __m256 sum, bool round_weights) {
    const uint8_t* s = blk + 96;
    const float d = f16_to_f32(uint16_t(blk[108] | (blk[109] << 8)));
    float scales[16];
    for (int j = 0; j < 16; ++j) {
        const int low = (j < 8) ? (s[j] & 15) : (s[j-8] >> 4);
        const int high = (s[8+(j&3)] >> (2*(j>>2))) & 3;
        scales[j] = d * float((low | (high << 4)) - 32);
    }
    const __m256i ones = _mm256_set1_epi32(1);
    const __m256i threes = _mm256_set1_epi32(3);
    for (int group = 0; group < 8; ++group) {
        const uint8_t* low = blk + 32 + (group / 4) * 32;
        const __m128i ls = _mm_cvtsi32_si128(2*(group%4));
        const __m128i hs = _mm_cvtsi32_si128(group);
        for (int j = 0; j < 32; j += 8) {
            __m256i ql = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(low+j)));
            __m256i qh = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(blk+j)));
            ql = _mm256_and_si256(_mm256_srl_epi32(ql,ls),threes);
            qh = _mm256_xor_si256(_mm256_and_si256(_mm256_srl_epi32(qh,hs),ones),ones);
            const __m256i qi = _mm256_sub_epi32(ql,_mm256_slli_epi32(qh,2));
            __m256 w = _mm256_mul_ps(_mm256_cvtepi32_ps(qi),_mm256_set1_ps(scales[group*2+j/16]));
            if (round_weights) w = _mm256_cvtph_ps(_mm256_cvtps_ph(w,0));
            sum = _mm256_fmadd_ps(w,_mm256_loadu_ps(x+group*32+j),sum);
        }
    }
    return sum;
}

// Materialize at most one quantization block, kept on the stack/in cache.
// round_weights reproduces the existing quantized -> F16 weight values.
static float quant_dot(int type, const uint8_t* row, const float* x,
                       int columns, bool round_weights) {
    const int bs = block_size_of(type), pb = bytes_per_block(type);
    alignas(32) float tmp[256];
    __m256 sum = _mm256_setzero_ps();
    for (int offset = 0; offset < columns; offset += bs, row += pb) {
        if (type == GGML_TYPE_Q3_K) {
            sum = q3_dot_block(row, x + offset, sum, round_weights);
            continue;
        }
        decode_block(type, row, tmp);
        for (int j = 0; j < bs; j += 8) {
            __m256 w = _mm256_load_ps(tmp + j);
            if (round_weights) w = _mm256_cvtph_ps(_mm256_cvtps_ph(w, 0));
            sum = _mm256_fmadd_ps(w, _mm256_loadu_ps(x + offset + j), sum);
        }
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    float result = 0;
    for (float lane : lanes) result += lane;
    return result;
}

extern "C" __declspec(dllexport) int moe_quant_expert(
    int gate_type, const uint8_t* gate, int64_t gate_bytes,
    int up_type, const uint8_t* up, int64_t up_bytes,
    int down_type, const uint8_t* down, int64_t down_bytes,
    const float* input, int hidden, int intermediate, float* output,
    int fp16_graph) {
    const int types[3] = {gate_type, up_type, down_type};
    const int cols[3] = {hidden, hidden, intermediate};
    const int rows[3] = {intermediate, intermediate, hidden};
    const int64_t sizes[3] = {gate_bytes, up_bytes, down_bytes};
    if (!gate || !up || !down || !input || !output || hidden <= 0 ||
        intermediate <= 0 || hidden > 65536 || intermediate > 65536 || (fp16_graph != 0 && fp16_graph != 1)) return -1;
    for (int i = 0; i < 3; ++i) {
        const int bs = block_size_of(types[i]);
        if (!bs || cols[i] % bs) return -2;
        const int64_t expected = int64_t(rows[i]) * (cols[i] / bs) * bytes_per_block(types[i]);
        if (sizes[i] != expected) return -3;
    }
    try {
        std::vector<float> x(hidden), activation(intermediate);
        for (int i = 0; i < hidden; ++i)
            x[i] = fp16_graph ? round16(input[i]) : input[i];
        const int64_t gr = int64_t(hidden / block_size_of(gate_type)) * bytes_per_block(gate_type);
        const int64_t ur = int64_t(hidden / block_size_of(up_type)) * bytes_per_block(up_type);
        const int64_t dr = int64_t(intermediate / block_size_of(down_type)) * bytes_per_block(down_type);
        for (int row = 0; row < intermediate; ++row) {
            float g = quant_dot(gate_type, gate + row * gr, x.data(), hidden, fp16_graph);
            float u = quant_dot(up_type, up + row * ur, x.data(), hidden, fp16_graph);
            if (fp16_graph) { g = round16(g); u = round16(u); }
            float sigmoid = 1.0f / (1.0f + std::exp(-g));
            if (fp16_graph) sigmoid = round16(sigmoid);
            float silu = g * sigmoid;
            if (fp16_graph) silu = round16(silu);
            float a = silu * u;
            activation[row] = fp16_graph ? round16(a) : a;
        }
        for (int row = 0; row < hidden; ++row) {
            float y = quant_dot(down_type, down + row * dr, activation.data(), intermediate, fp16_graph);
            output[row] = fp16_graph ? round16(y) : y;
            if (!std::isfinite(output[row])) return -4;
        }
    } catch (...) { return -5; }
    return 0;
}
