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
template<int Group, int J>
static inline __m256 q3_part(const uint8_t* blk, const float* x,
                             const float* scales, __m256 sum, bool rounding) {
    const __m128i low = _mm_loadl_epi64((const __m128i*)(blk + 32 + (Group / 4)*32 + J));
    const __m128i high = _mm_loadl_epi64((const __m128i*)(blk + J));
    const __m128i l = _mm_and_si128(_mm_srli_epi16(low, 2*(Group%4)), _mm_set1_epi8(3));
    const __m128i h = _mm_xor_si128(
        _mm_and_si128(_mm_srli_epi16(high, Group), _mm_set1_epi8(1)), _mm_set1_epi8(1));
    const __m128i signed_bytes = _mm_sub_epi8(l, _mm_slli_epi16(h, 2));
    const __m256i integers = _mm256_cvtepi8_epi32(signed_bytes);
    __m256 w = _mm256_mul_ps(_mm256_cvtepi32_ps(integers), _mm256_set1_ps(scales[Group*2+J/16]));
    if (rounding) w = _mm256_cvtph_ps(_mm256_cvtps_ph(w, 0));
    return _mm256_fmadd_ps(w, _mm256_loadu_ps(x + Group*32 + J), sum);
}

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
    sum = q3_part<0, 0>(blk, x, scales, sum, round_weights);
    sum = q3_part<0, 8>(blk, x, scales, sum, round_weights);
    sum = q3_part<0, 16>(blk, x, scales, sum, round_weights);
    sum = q3_part<0, 24>(blk, x, scales, sum, round_weights);
    sum = q3_part<1, 0>(blk, x, scales, sum, round_weights);
    sum = q3_part<1, 8>(blk, x, scales, sum, round_weights);
    sum = q3_part<1, 16>(blk, x, scales, sum, round_weights);
    sum = q3_part<1, 24>(blk, x, scales, sum, round_weights);
    sum = q3_part<2, 0>(blk, x, scales, sum, round_weights);
    sum = q3_part<2, 8>(blk, x, scales, sum, round_weights);
    sum = q3_part<2, 16>(blk, x, scales, sum, round_weights);
    sum = q3_part<2, 24>(blk, x, scales, sum, round_weights);
    sum = q3_part<3, 0>(blk, x, scales, sum, round_weights);
    sum = q3_part<3, 8>(blk, x, scales, sum, round_weights);
    sum = q3_part<3, 16>(blk, x, scales, sum, round_weights);
    sum = q3_part<3, 24>(blk, x, scales, sum, round_weights);
    sum = q3_part<4, 0>(blk, x, scales, sum, round_weights);
    sum = q3_part<4, 8>(blk, x, scales, sum, round_weights);
    sum = q3_part<4, 16>(blk, x, scales, sum, round_weights);
    sum = q3_part<4, 24>(blk, x, scales, sum, round_weights);
    sum = q3_part<5, 0>(blk, x, scales, sum, round_weights);
    sum = q3_part<5, 8>(blk, x, scales, sum, round_weights);
    sum = q3_part<5, 16>(blk, x, scales, sum, round_weights);
    sum = q3_part<5, 24>(blk, x, scales, sum, round_weights);
    sum = q3_part<6, 0>(blk, x, scales, sum, round_weights);
    sum = q3_part<6, 8>(blk, x, scales, sum, round_weights);
    sum = q3_part<6, 16>(blk, x, scales, sum, round_weights);
    sum = q3_part<6, 24>(blk, x, scales, sum, round_weights);
    sum = q3_part<7, 0>(blk, x, scales, sum, round_weights);
    sum = q3_part<7, 8>(blk, x, scales, sum, round_weights);
    sum = q3_part<7, 16>(blk, x, scales, sum, round_weights);
    sum = q3_part<7, 24>(blk, x, scales, sum, round_weights);
    return sum;
}

// Direct Q6_K dot product: preserve FP16 rounding and FMA order.

template<int G, int J>
static inline __m256 q6_part(const uint8_t* b, const float* x,
                             const float* scales, __m256 sum, bool rounding) {
    const __m128i lo = _mm_loadl_epi64((const __m128i*)(b + (G/4)*64 + (G%2)*32 + J));
    const __m128i hi = _mm_loadl_epi64((const __m128i*)(b + 128 + (G/4)*32 + J));
    const __m128i l = _mm_and_si128(_mm_srli_epi16(lo, (G%4 >= 2) ? 4 : 0), _mm_set1_epi8(15));
    const __m128i h = _mm_and_si128(_mm_srli_epi16(hi, 2*(G%4)), _mm_set1_epi8(3));
    const __m128i q = _mm_sub_epi8(_mm_or_si128(l, _mm_slli_epi16(h, 4)), _mm_set1_epi8(32));
    __m256 w = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q)),
                            _mm256_set1_ps(scales[G*2+J/16]));
    if (rounding) w = _mm256_cvtph_ps(_mm256_cvtps_ph(w, 0));
    return _mm256_fmadd_ps(w, _mm256_loadu_ps(x + G*32 + J), sum);
}
static inline __m256 q6_dot_block(const uint8_t* b, const float* x,
                                  __m256 sum, bool rounding) {
    const float d = f16_to_f32(uint16_t(b[208] | (b[209] << 8)));
    const int8_t* s = (const int8_t*)(b + 192);
    float scales[16];
    for (int j = 0; j < 16; ++j) scales[j] = d * float(s[j]);
    sum = q6_part<0, 0>(b, x, scales, sum, rounding);
    sum = q6_part<0, 8>(b, x, scales, sum, rounding);
    sum = q6_part<0, 16>(b, x, scales, sum, rounding);
    sum = q6_part<0, 24>(b, x, scales, sum, rounding);
    sum = q6_part<1, 0>(b, x, scales, sum, rounding);
    sum = q6_part<1, 8>(b, x, scales, sum, rounding);
    sum = q6_part<1, 16>(b, x, scales, sum, rounding);
    sum = q6_part<1, 24>(b, x, scales, sum, rounding);
    sum = q6_part<2, 0>(b, x, scales, sum, rounding);
    sum = q6_part<2, 8>(b, x, scales, sum, rounding);
    sum = q6_part<2, 16>(b, x, scales, sum, rounding);
    sum = q6_part<2, 24>(b, x, scales, sum, rounding);
    sum = q6_part<3, 0>(b, x, scales, sum, rounding);
    sum = q6_part<3, 8>(b, x, scales, sum, rounding);
    sum = q6_part<3, 16>(b, x, scales, sum, rounding);
    sum = q6_part<3, 24>(b, x, scales, sum, rounding);
    sum = q6_part<4, 0>(b, x, scales, sum, rounding);
    sum = q6_part<4, 8>(b, x, scales, sum, rounding);
    sum = q6_part<4, 16>(b, x, scales, sum, rounding);
    sum = q6_part<4, 24>(b, x, scales, sum, rounding);
    sum = q6_part<5, 0>(b, x, scales, sum, rounding);
    sum = q6_part<5, 8>(b, x, scales, sum, rounding);
    sum = q6_part<5, 16>(b, x, scales, sum, rounding);
    sum = q6_part<5, 24>(b, x, scales, sum, rounding);
    sum = q6_part<6, 0>(b, x, scales, sum, rounding);
    sum = q6_part<6, 8>(b, x, scales, sum, rounding);
    sum = q6_part<6, 16>(b, x, scales, sum, rounding);
    sum = q6_part<6, 24>(b, x, scales, sum, rounding);
    sum = q6_part<7, 0>(b, x, scales, sum, rounding);
    sum = q6_part<7, 8>(b, x, scales, sum, rounding);
    sum = q6_part<7, 16>(b, x, scales, sum, rounding);
    sum = q6_part<7, 24>(b, x, scales, sum, rounding);
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
        if (type == GGML_TYPE_Q6_K) {
            sum = q6_dot_block(row, x + offset, sum, round_weights);
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

// Shared native library; independent DeltaNet entry point.
#include "moe_delta.cpp"
