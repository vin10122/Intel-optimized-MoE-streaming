/*
 * fast_dequant_q8_0_avx2.cpp — Q8_0 to BF16 dequantizer with AVX2 optimization
 * 
 * Q8_0 Block structure (272 bytes for 256 elements):
 *   - 16 bytes: 8 float16 scales (1 per 32 elements)
 *   - 256 bytes: 256 int8 values
 * 
 * Total per block: 272 bytes
 * Elements per block: 256
 * Bytes per element: 272/256 = 1.0625
 * 
 * Dequantization formula for each 32-element sub-block:
 *   value[i] = (qs[i] - 128) * scale
 *   where qs[i] is int8, scale is float16
 * 
 * Output: BF16 (uint16_t)
 */

#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <omp.h>

// Structure for Q8_0 block (272 bytes)
#pragma pack(push, 1)
struct block_q8_0 {
    uint16_t scales[8];  // 8 float16 scales (16 bytes)
    int8_t qs[256];      // 256 int8 values (256 bytes)
};
#pragma pack(pop)

/*
 * Convert 8 float32 values to 8 bfloat16 using AVX2
 * BF16: sign bit + 7 exponent bits + 7 mantissa bits (truncated from FP32)
 * We process 8 floats at a time (from the low 256 bits of the __m256)
 */
static inline void fp32_to_bf16_8(const __m256& v_fp32, uint16_t* out_bf16) {
    // Extract the low 128 bits (first 4 floats)
    __m128i v_int_lo = _mm256_castps256_si256(v_fp32);
    // Shift right by 16 to get bfloat16 (discard lower 16 bits of mantissa)
    __m128i hi_128 = _mm_srli_epi32(v_int_lo, 16);
    
    // Pack to 128 bits
    __m128i packed = _mm_packus_epi32(hi_128, _mm_setzero_si128());
    
    // Store 8 uint16_t values
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_bf16), packed);
}

/*
 * Dequantize one Q8_0 block (256 elements) to BF16 using AVX2
 */
static inline void dequant_block_q8_0_to_bf16_avx2(
    const block_q8_0* blk,
    uint16_t* __restrict out
) {
    // Process 8 sub-blocks of 32 elements each
    for (int s = 0; s < 8; ++s) {
        // Load scale as float32
        uint16_t scale_fp16 = blk->scales[s];
        __m128i scale_vec = _mm_cvtsi32_si128((int)(uint32_t)scale_fp16);
        __m256 v_scale = _mm256_set1_ps(_mm_cvtss_f32(_mm_cvtph_ps(scale_vec)));
        
        // Load 32 int8 values
        const int8_t* qs = blk->qs + s * 32;
        uint16_t* out_sub = out + s * 32;
        
        // Process 32 elements in chunks of 16
        for (int i = 0; i < 32; i += 16) {
            // Load 16 int8 values
            __m128i v_qs = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs + i));
            
            // Unpack int8 to int16 for computation
            __m128i v_lo = _mm_cvtepi8_epi16(v_qs);
            __m128i v_hi = _mm_cvtepi8_epi16(_mm_srli_si128(v_qs, 8));
            
            // Convert to float32
            __m256 v_f_lo = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_lo));
            __m256 v_f_hi = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_hi));
            
            // Subtract 128
            __m256 v_sub_lo = _mm256_sub_ps(v_f_lo, _mm256_set1_ps(128.0f));
            __m256 v_sub_hi = _mm256_sub_ps(v_f_hi, _mm256_set1_ps(128.0f));
            
            // Multiply by scale
            v_sub_lo = _mm256_mul_ps(v_sub_lo, v_scale);
            v_sub_hi = _mm256_mul_ps(v_sub_hi, v_scale);
            
            // Convert to BF16
            fp32_to_bf16_8(v_sub_lo, out_sub + i);
            fp32_to_bf16_8(v_sub_hi, out_sub + i + 8);
        }
    }
}

/*
 * Dequantize Q8_0 to BF16 using AVX2 and OpenMP
 */
extern "C" __declspec(dllexport) void dequantize_q8_0_to_bf16_avx2(
    const uint8_t* src_bytes,
    uint16_t* dst,
    int rows,
    int cols
) {
    const int elements_per_block = 256;
    const int blocks_per_row = cols / elements_per_block;
    const block_q8_0* blocks = reinterpret_cast<const block_q8_0*>(src_bytes);

    #pragma omp parallel for schedule(static) num_threads(4)
    for (int r = 0; r < rows; ++r) {
        uint16_t* row_dst = dst + (size_t)r * cols;
        const block_q8_0* row_blk = blocks + (size_t)r * blocks_per_row;
        
        for (int b = 0; b < blocks_per_row; ++b) {
            dequant_block_q8_0_to_bf16_avx2(row_blk + b, row_dst + b * elements_per_block);
        }
    }
}

/*
 * Simpler fallback version without AVX2 optimizations
 * (for compatibility)
 */
extern "C" __declspec(dllexport) void dequantize_q8_0_to_bf16(
    const uint8_t* src_bytes,
    uint16_t* dst,
    int rows,
    int cols
) {
    const int elements_per_block = 256;
    const int blocks_per_row = cols / elements_per_block;
    const block_q8_0* blocks = reinterpret_cast<const block_q8_0*>(src_bytes);

    #pragma omp parallel for schedule(static) num_threads(4)
    for (int r = 0; r < rows; ++r) {
        uint16_t* row_dst = dst + (size_t)r * cols;
        const block_q8_0* row_blk = blocks + (size_t)r * blocks_per_row;
        
        for (int b = 0; b < blocks_per_row; ++b) {
            const block_q8_0* blk = row_blk + b;
            uint16_t* out = row_dst + b * elements_per_block;
            
            for (int s = 0; s < 8; ++s) {
                // Convert fp16 scale to fp32
                uint16_t scale_fp16 = blk->scales[s];
                __m128i scale_vec = _mm_cvtsi32_si128((int)(uint32_t)scale_fp16);
                float scale = _mm_cvtss_f32(_mm_cvtph_ps(scale_vec));
                
                const int8_t* qs = blk->qs + s * 32;
                
                for (int i = 0; i < 32; ++i) {
                    // Dequantize: (qs[i] - 128) * scale
                    float val = (float)(qs[i] - 128) * scale;
                    
                    // Convert FP32 to BF16
                    uint32_t u;
                    std::memcpy(&u, &val, sizeof(u));
                    uint16_t bf16 = (uint16_t)((u & 0xFF800000) >> 16) | 
                                   (uint16_t)((u & 0x007FFFFF) >> 16);
                    out[s * 32 + i] = bf16;
                }
            }
        }
    }
}
