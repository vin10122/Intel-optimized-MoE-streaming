#include <cstdint>
#include <cstring>

static const float kvalues_iq4_nl[16] = {
    -1.0f, -0.6961914f, -0.515625f, -0.375f,
    -0.25390625f, -0.1484375f, -0.0546875f, 0.02734375f,
    0.1015625f, 0.1796875f, 0.265625f, 0.359375f,
    0.46875f, 0.59765625f, 0.7578125f, 1.0f
};

static inline float fp16_to_fp32(uint16_t h) {
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

#pragma pack(push, 1)
struct block_iq4_nl {
    uint16_t d;     // fp16 scale
    uint8_t qs[16]; // 32 nibbles
};
#pragma pack(pop)

extern "C" __declspec(dllexport) void dequantize_iq4_nl_cpp(
    const uint8_t* src_bytes,
    float* dst,
    int rows,
    int cols
) {
    const block_iq4_nl* blocks = reinterpret_cast<const block_iq4_nl*>(src_bytes);
    int blocks_per_row = cols / 32;

    for (int r = 0; r < rows; ++r) {
        float* row_dst = dst + r * cols;
        for (int b = 0; b < blocks_per_row; ++b) {
            const block_iq4_nl& blk = blocks[r * blocks_per_row + b];
            float scale = fp16_to_fp32(blk.d);
            float* out = row_dst + b * 32;

            for (int i = 0; i < 16; ++i) {
                uint8_t q = blk.qs[i];
                uint8_t q0 = q & 0x0F;
                uint8_t q1 = (q >> 4) & 0x0F;
                out[i] = scale * kvalues_iq4_nl[q0];
                out[i + 16] = scale * kvalues_iq4_nl[q1];
            }
        }
    }
}
