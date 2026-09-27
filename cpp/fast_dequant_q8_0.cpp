/*
 * fast_dequant_q8_0.cpp — Q8_0 to BF16 dequantizer
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
 *   q[i] = (qs[i] - 128) * scale
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
 * Convert float32 to bfloat16
 * BF16: sign bit + 7 exponent bits + 7 mantissa bits (truncated from FP32)
 */
static inline uint16_t fp32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    // Extract sign and exponent
    uint32_t sign_exp = u & 0xFF800000;
    // Shift right by 16 to get bfloat16 (discard lower 16 bits of mantissa)
    uint16_t bf16 = (uint16_t)(sign_exp >> 16) | (uint16_t)((u & 0x007FFFFF) >> 16);
    return bf16;
}

/*
 * Dequantize one Q8_0 block (256 elements) to BF16
 * Each 32-element sub-block uses one scale
 */
static inline void dequant_block_q8_0_to_bf16(
    const block_q8_0* blk,
    uint16_t* __restrict out
) {
    // Convert float16 scales to float32 for computation
    // We'll use F16C instructions
    
    for (int s = 0; s < 8; ++s) {
        // Get the scale for this 32-element sub-block
        uint16_t scale_fp16 = blk->scales[s];
        
        // Convert fp16 to fp32
        __m128i scale_vec = _mm_cvtsi32_si128((int)(uint32_t)scale_fp16);
        float scale = _mm_cvtss_f32(_mm_cvtph_ps(scale_vec));
        
        // Process 32 elements
        const int8_t* qs = blk->qs + s * 32;
        uint16_t* out_sub = out + s * 32;
        
        for (int i = 0; i < 32; ++i) {
            // Dequantize: (qs[i] - 128) * scale
            // qs[i] is in range [-128, 127]
            // Subtract 128 to get range [-128, 127] -> [-256, 127] but actually
            // Q8_0 uses: q = (value / scale) + 128
            // So: value = (q - 128) * scale
            float val = (float)(qs[i] - 128) * scale;
            out_sub[i] = fp32_to_bf16(val);
        }
    }
}

/*
 * Dequantize Q8_0 to BF16
 * src_bytes: pointer to Q8_0 data (must be aligned to block boundaries)
 * dst: output BF16 (uint16_t array) with shape [rows, cols]
 * rows: number of rows
 * cols: number of columns (must be multiple of 256)
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
            dequant_block_q8_0_to_bf16(row_blk + b, row_dst + b * elements_per_block);
        }
    }
}

/*
 * Alternative version: Q8_0 to FP16
 * For systems that don't support BF16 well
 */
extern "C" __declspec(dllexport) void dequantize_q8_0_to_fp16(
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
                uint16_t scale_fp16 = blk->scales[s];
                __m128i scale_vec = _mm_cvtsi32_si128((int)(uint32_t)scale_fp16);
                float scale = _mm_cvtss_f32(_mm_cvtph_ps(scale_vec));
                
                const int8_t* qs = blk->qs + s * 32;
                
                for (int i = 0; i < 32; ++i) {
                    // Convert (qs[i] - 128) * scale to FP16
                    float val = (float)(qs[i] - 128) * scale;
                    
                    // Convert FP32 to FP16
                    // This is a simple implementation; for better performance use _mm_cvtps_ph
                    uint32_t u;
                    std::memcpy(&u, &val, sizeof(u));
                    uint32_t sign = (u & 0x80000000) >> 16;
                    uint32_t exp = (u & 0x7F800000) >> 13;
                    uint32_t mantissa = (u & 0x007FFFFF) >> 13;
                    
                    // Handle subnormals, normals, infinities, NaN
                    uint16_t fp16;
                    if (exp == 0x380) { // 0x7F800000 >> 13 = 0x380 (256 in decimal)
                        // Infinity or NaN
                        fp16 = (uint16_t)(sign | 0x7C00 | (mantissa & 0x3FF));
                    } else if (exp == 0) {
                        // Subnormal or zero
                        fp16 = (uint16_t)(sign | mantissa);
                    } else {
                        exp += 0xF000; // Adjust exponent bias (127 - 15 = 112)
                        fp16 = (uint16_t)(sign | (exp & 0x7C00) | (mantissa & 0x3FF));
                    }
                    out[s * 32 + i] = fp16;
                }
            }
        }
    }
}
