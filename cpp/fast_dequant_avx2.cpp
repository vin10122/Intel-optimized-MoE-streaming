/*
 * fast_dequant_avx2.cpp  —  IQ4_NL AVX2 dequantizer
 *
 * Block layout (18 bytes, #pragma pack(1)):
 *   uint16_t d      — fp16 scale
 *   uint8_t  qs[16] — 32 nibbles packed as lo|hi per byte
 *
 * Per-block output: 32 float32
 *   out[0..15]  = scale * table[lo_nibble[i]]   i=0..15
 *   out[16..31] = scale * table[hi_nibble[i]]   i=0..15
 *
 * AVX2 plan per block:
 *   1. fp16→fp32 the scale using F16C (_mm_cvtph_ps)
 *   2. _mm_loadu_si128 → 16 bytes of qs
 *   3. mask & shift to get lo8, hi8 (each 16×uint8 in __m128i)
 *   4. Four rounds of:
 *        _mm256_cvtepu8_epi32(4-byte slice)  → __m256i of 8 int32 indices
 *        _mm256_i32gather_ps(table, idx, 4) → __m256 of 8 floats
 *        _mm256_mul_ps(..., vscale)           → scale
 *        _mm256_storeu_ps(out+offset, ...)    → store
 *      covering lo[0..7], lo[8..15], hi[0..7], hi[8..15]
 */

#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <omp.h>

/* ---- IQ4_NL non-linear lookup table (from GGUF spec / llama.cpp) ---- */
alignas(32) static const float kvalues_iq4_nl[16] = {
    -1.0f,        -0.6961914f,  -0.515625f,   -0.375f,
    -0.25390625f, -0.1484375f,  -0.0546875f,   0.02734375f,
     0.1015625f,   0.1796875f,   0.265625f,    0.359375f,
     0.46875f,     0.59765625f,  0.7578125f,   1.0f
};

/* ---- Block struct (18 bytes) ---- */
#pragma pack(push, 1)
struct block_iq4_nl {
    uint16_t d;
    uint8_t  qs[16];
};
#pragma pack(pop)

/* ---- Decode fp16 scale using F16C (always available with /arch:AVX2) ---- */
static inline float decode_scale(uint16_t h) {
    __m128i v = _mm_cvtsi32_si128((int)(uint32_t)h);
    return _mm_cvtss_f32(_mm_cvtph_ps(v));
}

/*
 * Gather 8 floats from kvalues_iq4_nl using 8 byte-sized (uint8) indices
 * stored in the lowest 8 bytes of a __m128i.
 * We need to zero-extend uint8 → int32 for _mm256_i32gather_ps.
 * _mm_cvtepu8_epi32 only uses the lowest 4 bytes of the __m128i source.
 * So we call it twice: once on src, once on src shifted right by 4 bytes.
 */
static inline void gather_and_store_16(
    const __m128i src8,   /* 16 x uint8 nibble indices */
    const __m256  vscale,
    float*        out     /* write 16 floats */
) {
    /* lo 8 indices → out[0..7] */
    __m256i idx0 = _mm256_cvtepu8_epi32(src8);
    _mm256_storeu_ps(out,     _mm256_mul_ps(_mm256_i32gather_ps(kvalues_iq4_nl, idx0, 4), vscale));
    /* hi 8 indices → out[8..15] */
    __m256i idx1 = _mm256_cvtepu8_epi32(_mm_srli_si128(src8, 8));
    _mm256_storeu_ps(out + 8, _mm256_mul_ps(_mm256_i32gather_ps(kvalues_iq4_nl, idx1, 4), vscale));
}

/* ---- Dequantize one IQ4_NL block → 32 floats ---- */
static inline void dequant_block(const block_iq4_nl* blk, float* __restrict out) {
    const __m256 vscale = _mm256_set1_ps(decode_scale(blk->d));

    /* Load 16 packed bytes */
    const __m128i qraw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk->qs));

    /* Extract lo nibbles: q & 0x0F */
    const __m128i lo_mask = _mm_set1_epi8(0x0F);
    const __m128i lo8 = _mm_and_si128(qraw, lo_mask);

    /* Extract hi nibbles: (q >> 4) & 0x0F */
    const __m128i hi8 = _mm_and_si128(_mm_srli_epi16(qraw, 4), lo_mask);

    /* lo nibbles → out[0..15], hi nibbles → out[16..31] */
    gather_and_store_16(lo8, vscale, out);
    gather_and_store_16(hi8, vscale, out + 16);
}

/* ---- Exported DLL function ---- */
extern "C" __declspec(dllexport) void dequantize_iq4_nl_avx2(
    const uint8_t* src_bytes,
    float*         dst,
    int            rows,
    int            cols    /* must be multiple of 32 */
) {
    const int blocks_per_row = cols / 32;
    const block_iq4_nl* blocks = reinterpret_cast<const block_iq4_nl*>(src_bytes);

    #pragma omp parallel for schedule(static) num_threads(4)
    for (int r = 0; r < rows; ++r) {
        float*              row_dst = dst    + (size_t)r * cols;
        const block_iq4_nl* row_blk = blocks + (size_t)r * blocks_per_row;
        for (int b = 0; b < blocks_per_row; ++b)
            dequant_block(row_blk + b, row_dst + b * 32);
    }
}
