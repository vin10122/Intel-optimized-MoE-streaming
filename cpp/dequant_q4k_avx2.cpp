/*
 * dequant_q4k_avx2.cpp - Q4_K dequantizer with AVX2
 *
 * Q4_K block layout (144 bytes):
 *   uint16_t d       - fp16 scale
 *   uint16_t dmin    - fp16 min scale
 *   uint8_t  scales[12] - 6-bit scales packed (2 scales per byte)
 *   uint8_t  qs[128]   - 4-bit quantized values (2 per byte)
 *
 * Dequantization formula: y = d * sc * q - dmin * m
 * where sc = scale[i/32], m = min[i/32], q = quantized value
 *
 * Processing: 8 groups of 32 elements each
 * Group i: uses scale[i], min[i], and 32 bytes of qs
 */

#include <cstdint>
#include <immintrin.h>
#include "dequant_k_tables.h"

#define QK_K 256

static inline float f16_to_f32_avx2(uint16_t h) {
    // Same as f16_to_f32 in dequant_k.cpp, optimized for AVX2 path
    uint32_t w = (h & 0x7FFF) << 13;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    if (exp == 0x1F) {
        exp = 0xFF;
    } else if (exp != 0) {
        exp += 127 - 15;
    } else if (w != 0) {
        exp = 127 - 15 + 1;
        while ((w & 0x00800000) == 0) {
            w <<= 1;
            exp--;
        }
        w &= 0x007FFFFF;
    }
    uint32_t u = sign | (exp << 23) | w;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

/* Unpack 6-bit scales and mins from 12 bytes */
static inline void unpack_scales_avx2(const uint8_t* s, float* sc, float* mn, float d, float dmin) {
    // Unpack using the same logic as unpack_scale_min_k4 in dequant_k.cpp
    uint8_t sc8[8], mn8[8];
    
    for (int j = 0; j < 4; ++j) {
        sc8[j] = (uint8_t)(s[j] & 63);
        mn8[j] = (uint8_t)(s[j + 4] & 63);
    }
    for (int j = 4; j < 8; ++j) {
        sc8[j] = (uint8_t)((s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4));
        mn8[j] = (uint8_t)((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
    }
    
    // Convert to float and apply scaling
    for (int j = 0; j < 8; ++j) {
        sc[j] = d * (float)(int8_t)(sc8[j] - 32);  // sc8 is signed offset
        mn[j] = dmin * (float)(int8_t)(mn8[j] - 32);
    }
}

/* Process one Q4_K block (144 bytes -> 256 floats) */
static void dequant_q4_k_avx2_block(const uint8_t* blk, float* dst) {
    const float d = f16_to_f32_avx2((uint16_t)(blk[0] | (blk[1] << 8)));
    const float dmin = f16_to_f32_avx2((uint16_t)(blk[2] | (blk[3] << 8)));
    const uint8_t* qs = blk + 16;
    
    float sc[8], mn[8];
    unpack_scales_avx2(blk + 4, sc, mn, d, dmin);
    
    // Process 8 groups of 32 elements
    for (int g = 0; g < 4; ++g) {
        const float scale0 = sc[g * 2 + 0];
        const float scale1 = sc[g * 2 + 1];
        const float min0 = mn[g * 2 + 0];
        const float min1 = mn[g * 2 + 1];
        
        const uint8_t* group_qs = qs + g * 32;
        float* group_out = dst + g * 64;
        
        // Process 32 elements (16 bytes of qs)
        for (int l = 0; l < 32; ++l) {
            uint8_t q = group_qs[l];
            uint8_t q_lo = q & 0x0F;
            uint8_t q_hi = (q >> 4) & 0x0F;
            
            // For Q4_K: q values are signed (0-15 -> -8 to 7)
            // But the formula uses them as-is and subtracts min
            group_out[l] = scale0 * (float)q_lo - min0;
            group_out[l + 32] = scale1 * (float)q_hi - min1;
        }
    }
}

/* AVX2-optimized version processing 8 floats at a time */
static void dequant_q4_k_avx2_block_optimized(const uint8_t* blk, float* dst) {
    const float d = f16_to_f32_avx2((uint16_t)(blk[0] | (blk[1] << 8)));
    const float dmin = f16_to_f32_avx2((uint16_t)(blk[2] | (blk[3] << 8)));
    const uint8_t* qs = blk + 16;
    
    float sc[8], mn[8];
    unpack_scales_avx2(blk + 4, sc, mn, d, dmin);
    
    for (int g = 0; g < 4; ++g) {
        const __m256 scale0 = _mm256_set1_ps(sc[g * 2 + 0]);
        const __m256 scale1 = _mm256_set1_ps(sc[g * 2 + 1]);
        const __m256 min0 = _mm256_set1_ps(mn[g * 2 + 0]);
        const __m256 min1 = _mm256_set1_ps(mn[g * 2 + 1]);
        
        const uint8_t* group_qs = qs + g * 32;
        float* group_out = dst + g * 64;
        
        // Process 8 elements at a time
        for (int l = 0; l < 32; l += 8) {
            // Load 8 bytes of qs
            __m128i q_bytes = _mm_loadl_epi64((const __m128i*)(group_qs + l));
            
            // Extract low nibbles (4 bits each)
            __m128i lo = _mm_and_si128(q_bytes, _mm_set1_epi8(0x0F));
            // Extract high nibbles
            __m128i hi = _mm_and_si128(_mm_srli_epi16(q_bytes, 4), _mm_set1_epi8(0x0F));
            
            // Convert uint8 to int32 (for signed interpretation)
            // Note: Q4_K uses signed values, but the raw nibbles are 0-15
            // The formula is: y = d*sc*q - dmin*m, where q is treated as signed
            
            // For simplicity, we'll do scalar conversion here
            // (Full AVX2 vectorization of signed nibbles is complex)
            for (int i = 0; i < 8; ++i) {
                float q_lo = (float)(int8_t)(_mm_cvtsi128_si32(_mm_srli_si128(lo, i * 8)) & 0x0F) - 8.0f;
                float q_hi = (float)(int8_t)(_mm_cvtsi128_si32(_mm_srli_si128(hi, i * 8)) & 0x0F) - 8.0f;
                
                group_out[l + i] = _mm_cvtss_f32(_mm256_mul_ps(scale0, _mm256_set1_ps(q_lo))) - 
                                   _mm_cvtss_f32(min0);
                group_out[l + i + 32] = _mm_cvtss_f32(_mm256_mul_ps(scale1, _mm256_set1_ps(q_hi))) - 
                                        _mm_cvtss_f32(min1);
            }
        }
    }
}

/* Main dispatch function */
extern "C" __declspec(dllexport) void dequant_q4_k_avx2(
    const uint8_t* src,
    float* dst,
    int64_t nblocks
) {
    for (int64_t b = 0; b < nblocks; ++b) {
        dequant_q4_k_avx2_block(src + b * 144, dst + b * QK_K);
    }
}
