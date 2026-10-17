/*
 * dequant_k.cpp - dequantizzatori ggml K-quant -> f32 / f16.
 *
 * Copre i formati presenti nel GGUF Qwen3.6-35B-A3B:
 *   Q8_0 (id 8)    34 B / 32 elementi      blk.0.ffn_down_exps
 *   Q3_K (id 11)  110 B / 256 elementi    30 layer su 40  <- formato dominante
 *   Q4_K (id 12)  144 B / 256 elementi    10 layer
 *   Q5_K (id 13)  176 B / 256 elementi
 *   Q6_K (id 14)  210 B / 256 elementi    shared expert
 *
 * I layout e' l'algebra degli indici sono ricavati dalla reference
 * gguf.quants (la stessa che llama.cpp replica in ggml-quants.c), non dalla
 * memoria. tools/test_dequant_k.py confronta ogni kernel contro quella reference
 * su dati reali: se un indice e' sbagliato, il test fallisce.
 *
 * NOTA Q8_0: il blocco e'  f16 d + int8 qs[32]  e la formula e'  y = qs * d.
 * NON esiste l'offset -128. La versione precedente di questo progetto usava
 * sia il layout {scales[8], qs[256]} (invece dell'interleaved reale
 * d,qs[32],d,qs[32],...) sia -128: produceva NaN.
 *
 * Punto di ingresso unico, con l'id del tipo ggml:
 *   int dequant_k(int type, const uint8_t* src, void* dst, int64_t n, int f16)
 * Ritorna 0 su successo, -1 su argomenti non validi.
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <ppl.h>

/* Mappature misurate da tools/derive_k_tables.py, non derivate a mano. */
#include "dequant_k_tables.h"

#define QK_K 256

#define GGML_TYPE_Q8_0 8
#define GGML_TYPE_Q3_K 11
#define GGML_TYPE_Q4_K 12
#define GGML_TYPE_Q5_K 13
#define GGML_TYPE_Q6_K 14

/* Conversione f16 <-> f32 in software, senza dipendere dagli intrinsic F16C:
   il toolset 2022 BuildTools non dichiara _cvtsh_ss/_cvtss_sh senza
   l'header apposito, e una conversione software e' comunque vettorizzabile.
   Resta il fallback per CPU senza F16C: il path misurato su Qwen3-MoE e' 35x
   piu' lento della conversione vettorizzata, quindi conta solo come sicurezza. */

static inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;                       /* +/- zero */
        } else {
            /* subnormale f16 -> f32 normalizzato.
             * man * 2^-24, e shift per portare il bit 10 in testa: valore
             * = (1 + frac) * 2^(-14 - e), quindi esponente biased = 113 - e.
             * Con 112 - e si perde un fattore 2: 0x200 -> 2^-16 invece di
             * 2^-15. */
            uint32_t e = 0;
            uint32_t m = man;
            while ((m & 0x400u) == 0) { m <<= 1; ++e; }
            m &= 0x3FFu;
            bits = sign | ((113u - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (man << 13);   /* inf / nan */
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

static inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;

    if (((x >> 23) & 0xFFu) == 0xFFu) {           /* inf / nan */
        return (uint16_t)(sign | 0x7C00u | (man ? 0x200u | (man >> 13) : 0u));
    }
    if (exp >= 0x1F) {                            /* overflow -> inf */
        return (uint16_t)(sign | 0x7C00u);
    }
    if (exp <= 0) {                               /* subnormale / zero */
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t m = man >> shift;
        /* round to nearest even */
        const uint32_t rem = man & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (m & 1u))) ++m;
        return (uint16_t)(sign | m);
    }
    /* round to nearest even sulla mantissa */
    uint32_t m = man >> 13;
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (m & 1u))) {
        ++m;
        if (m == 0x400u) { m = 0; ++exp; if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u); }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | m);
}

#ifdef __AVX2__
#include <intrin.h>

static bool have_f16c() {
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 1);
    return (regs[2] & (1 << 29)) != 0;   /* ECX bit 29 = F16C */
}

/* 8 f32 -> 8 f16 per istruzione. Stessa semantica di arrotondamento della
   versione software: round to nearest even, subnormali inclusi. */
static void cvt_f32_to_f16(const float* src, uint16_t* dst, int64_t n) {
    int64_t i = 0;
    if (have_f16c()) {
        for (; i + 8 <= n; i += 8) {
            const __m256 v = _mm256_loadu_ps(src + i);
            /* 0 = round to nearest even, senza eccezioni: stessa aritmetica
               della versione software. */
            const __m128i h = _mm256_cvtps_ph(v, 0);
            _mm_storeu_si128((__m128i*)(dst + i), h);
        }
    }
    for (; i < n; ++i) dst[i] = f32_to_f16(src[i]);
}
#else
static void cvt_f32_to_f16(const float* src, uint16_t* dst, int64_t n) {
    for (int64_t i = 0; i < n; ++i) dst[i] = f32_to_f16(src[i]);
}
#endif

/* ------------------------------------------------------------------ Q8_0 --- */
/* 34 B:  f16 d,  int8 qs[32].   y[i] = qs[i] * d */

static void dequant_q8_0(const uint8_t* src, float* dst, int64_t nblocks) {
    for (int64_t b = 0; b < nblocks; ++b) {
        const uint8_t* blk = src + b * 34;
        const float d = f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        const int8_t* qs = (const int8_t*)(blk + 2);
        float* out = dst + b * 32;
        for (int j = 0; j < 32; ++j) out[j] = (float)qs[j] * d;
    }
}

/* ------------------------------------- scale/min 6 bit, condivisi Q4_K/Q5_K --- */
/*
 * s = 12 byte.  Per j in 0..7:
 *   low  = (j < 4) ? (s[j]      & 63) : (s[j-4] >> 6)
 *   high = (j < 4) ? (s[j+4]    & 63) : ((s[j] >> 6) << 4)
 *   d[j] = (uint8_t)(s[j] & 63)               se j < 4
 *         (s[j+4] & 0x0F) | ((s[j-4] >> 6) << 4)   se j >= 4
 *   m[j] = (s[j+4] & 63)                      se j < 4
 *         (s[j+4] >> 4)  | ((s[j]   >> 6) << 4)  se j >= 4
 */
static inline void unpack_scale_min_k4(const uint8_t* s, uint8_t* d, uint8_t* m) {
    for (int j = 0; j < 4; ++j) {
        d[j] = (uint8_t)(s[j] & 63);
        m[j] = (uint8_t)(s[j + 4] & 63);
    }
    for (int j = 4; j < 8; ++j) {
        d[j] = (uint8_t)((s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4));
        m[j] = (uint8_t)((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
    }
}

/* ------------------------------------------------------------------ Q4_K --- */
/* 144 B: f16 d, f16 dmin, scales[12], qs[128].
 * Ogni gruppo di 64 byte di qs produce 32 valori dal nibble basso e poi 32 dal
 * nibble alto.  y = d*sc*q - dmin*m,  8 sub-blocchi da 32. */

static void dequant_q4_k(const uint8_t* src, float* dst, int64_t nblocks) {
    for (int64_t b = 0; b < nblocks; ++b) {
        const uint8_t* blk = src + b * 144;
        const float d = f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        const float dmin = f16_to_f32((uint16_t)(blk[2] | (blk[3] << 8)));
        const uint8_t* qs = blk + 16;

        uint8_t sc[8], mn[8];
        unpack_scale_min_k4(blk + 4, sc, mn);

        float* out = dst + b * QK_K;
        int is = 0;
        for (int g = 0; g < 4; ++g) {
            const float d1 = d * (float)sc[is + 0];
            const float m1 = dmin * (float)mn[is + 0];
            const float d2 = d * (float)sc[is + 1];
            const float m2 = dmin * (float)mn[is + 1];
            for (int l = 0; l < 32; ++l) out[l] = d1 * (float)(qs[l] & 0x0F) - m1;
            for (int l = 0; l < 32; ++l) out[32 + l] = d2 * (float)(qs[l] >> 4) - m2;
            out += 64;
            qs += 32;
            is += 2;
        }
    }
}

/* ------------------------------------------------------------------ Q5_K --- */
/* 176 B: f16 d @0, f16 dmin @2, scales[12] @4, qh[32] @16, qs[128] @48.
 *
 *   y[i] = d * sc[i/32] * q[i] - dmin * mn[i/32]
 *
 * Il termine -dmin*mn e' una costante del sub-blocco, non un secondo campo
 * quantizzato. Le scale usano lo stesso impaccamento di Q4_K
 * (unpack_scale_min_k4): per j >= 4 i due bit alti arrivano da s[j-4] >> 6 e
 * non vanno persi, altrimenti metà dei sub-blocchi torna sbagliato.
 *
 * q e' un intero a 5 bit: ql a offset 0 (4 bit, qs @48) e qh a offset 4
 * (1 bit, qh @16). L'interlacciamento dei due byte e' misurato, non derivato.
 */
static void dequant_q5_k(const uint8_t* src, float* dst, int64_t nblocks) {
    for (int64_t b = 0; b < nblocks; ++b) {
        const uint8_t* blk = src + b * 176;
        const float d = f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        const float dmin = f16_to_f32((uint16_t)(blk[2] | (blk[3] << 8)));

        uint8_t sc[8], mn[8];
        unpack_scale_min_k4(blk + 4, sc, mn);

        float* out = dst + b * QK_K;
        for (int i = 0; i < QK_K; ++i) {
            const int j = i >> 5;
            const int q = (int)((blk[Q5_K_B1[i]] >> Q5_K_S1[i]) & ((1u << Q5_K_W1[i]) - 1u))
                        << Q5_K_O1[i]
                        | (int)((blk[Q5_K_B2[i]] >> Q5_K_S2[i]) & ((1u << Q5_K_W2[i]) - 1u))
                        << Q5_K_O2[i];
            out[i] = d * (float)sc[j] * (float)q - dmin * (float)mn[j];
        }
    }
}

/* ------------------------------------------------------------------ Q6_K --- */
/* 210 B: ql[128], qh[64], int8 scales[16], f16 d.
 *
 * Il mapping ql/qh -> output e' un interlacciamento non ovvio, quindi e'
 * misurato invece che derivato: tools/derive_k_tables.py accende un bit alla
 * volta sulla reference e registra, per ogni output, il byte e lo shift dei due
 * campi. Le tabelle sono in dequant_k_tables.h.
 *
 *   B1/S1 -> campo a 4 bit, in ql[0..127]
 *   B2/S2 -> campo a 2 bit, in qh[128..191]
 *   q = (low4 | (high2 << 4)) - 32,   y = d * scales[i/16] * q
 */
static void dequant_q6_k(const uint8_t* src, float* dst, int64_t nblocks) {
    for (int64_t b = 0; b < nblocks; ++b) {
        const uint8_t* blk = src + b * 210;
        const int8_t* sc = (const int8_t*)(blk + 192);
        const float d = f16_to_f32((uint16_t)(blk[208] | (blk[209] << 8)));

        float* out = dst + b * QK_K;
        for (int i = 0; i < QK_K; ++i) {
            const int low4 = (blk[Q6_K_B1[i]] >> Q6_K_S1[i]) & 0x0F;
            const int high2 = (blk[Q6_K_B2[i]] >> Q6_K_S2[i]) & 0x03;
            const int q = (low4 | (high2 << 4)) - 32;
            out[i] = d * (float)sc[i >> 4] * (float)q;
        }
    }
}

/* ------------------------------------------------------------------ Q3_K --- */
/* 110 B: hmask[32], ql[64], scales[12] (6 bit ciascuno), f16 d.
 *
 *   B1/S1 -> campo a 1 bit, in hmask[0..31]
 *   B2/S2 -> campo a 2 bit, in ql[32..95]
 *   q = qlv - ((bit ^ 1) << 2)     <- l'offset e' 0 quando il bit e' 1
 *   y[i] = d * scale[i/16] * q
 */
static void dequant_q3_k(const uint8_t* src, float* dst, int64_t nblocks) {
    for (int64_t b = 0; b < nblocks; ++b) {
        const uint8_t* blk = src + b * 110;
        const uint8_t* s = blk + 96;
        const float d = f16_to_f32((uint16_t)(blk[108] | (blk[109] << 8)));

        int8_t scale[16];
        for (int j = 0; j < 16; ++j) {
            const int low4 = (j < 8) ? (s[j] & 0x0F) : ((s[j - 8] >> 4) & 0x0F);
            const int high2 = (s[8 + (j & 3)] >> (2 * (j >> 2))) & 0x03;
            scale[j] = (int8_t)((low4 | (high2 << 4)) - 32);
        }

        float* out = dst + b * QK_K;
        for (int i = 0; i < QK_K; ++i) {
            const int qlv = (blk[Q3_K_B2[i]] >> Q3_K_S2[i]) & 0x03;
            const int qhb = ((blk[Q3_K_B1[i]] >> Q3_K_S1[i]) & 1) ^ 1;
            out[i] = d * (float)scale[i >> 4] * (float)(qlv - (qhb << 2));
        }
    }
}

/* -------------------------------------------------------------- dispatch --- */

static int block_size_of(int type) {
    switch (type) {
        case GGML_TYPE_Q8_0: return 32;
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K: return QK_K;
        default: return 0;
    }
}

static int bytes_per_block(int type) {
    switch (type) {
        case GGML_TYPE_Q8_0: return 34;
        case GGML_TYPE_Q3_K: return 110;
        case GGML_TYPE_Q4_K: return 144;
        case GGML_TYPE_Q5_K: return 176;
        case GGML_TYPE_Q6_K: return 210;
        default: return 0;
    }
}

static bool dequant_f16_chunk(int type, const uint8_t* src, uint16_t* dst,
                              int64_t done, int64_t n_now, int bs, int pb) {
    constexpr int64_t CHUNK = 8192;  /* 32 KiB: buffer temporaneo per task */
    alignas(32) float tmp[CHUNK];
    const int64_t b_now = n_now / bs;
    const uint8_t* chunk_src = src + done / bs * pb;
    switch (type) {
        case GGML_TYPE_Q8_0: dequant_q8_0(chunk_src, tmp, b_now); break;
        case GGML_TYPE_Q3_K: dequant_q3_k(chunk_src, tmp, b_now); break;
        case GGML_TYPE_Q4_K: dequant_q4_k(chunk_src, tmp, b_now); break;
        case GGML_TYPE_Q5_K: dequant_q5_k(chunk_src, tmp, b_now); break;
        case GGML_TYPE_Q6_K: dequant_q6_k(chunk_src, tmp, b_now); break;
        default: return false;
    }
    cvt_f32_to_f16(tmp, dst + done, n_now);
    return true;
}

extern "C" __declspec(dllexport) int dequant_k_supported(int type) {
    return block_size_of(type) != 0;
}

extern "C" __declspec(dllexport) int dequant_k_block_size(int type) {
    return block_size_of(type);
}

extern "C" __declspec(dllexport) int dequant_k(int type, const uint8_t* src,
                                              void* dst, int64_t n, int want_f16) {
    const int bs = block_size_of(type);
    const int pb = bytes_per_block(type);
    if (bs == 0 || src == nullptr || dst == nullptr) return -1;
    if (n <= 0 || n % bs != 0) return -1;
    const int64_t nblocks = n / bs;

    if (!want_f16) {
        float* o = (float*)dst;
        switch (type) {
            case GGML_TYPE_Q8_0: dequant_q8_0(src, o, nblocks); break;
            case GGML_TYPE_Q3_K: dequant_q3_k(src, o, nblocks); break;
            case GGML_TYPE_Q4_K: dequant_q4_k(src, o, nblocks); break;
            case GGML_TYPE_Q5_K: dequant_q5_k(src, o, nblocks); break;
            case GGML_TYPE_Q6_K: dequant_q6_k(src, o, nblocks); break;
        }
        return 0;
    }

    /* f16: si dequantizza a chunk in un buffer piccolo che resta in L1 e si
       converte subito, invece di materializzare l'intero tensore in f32.
       Con 3 MiB di picco su 1M elementi il buffer f32 finiva fuori cache e il
       ciclo di scrittura+conversione dominate il tempo: era il 94% del costo
       dell'expert. Lo stesso risultato in bit: la conversione e' indipendente
       per blocco. */
    constexpr int64_t CHUNK = 8192;              /* elementi per passata */
    uint16_t* o16 = (uint16_t*)dst;
    const int64_t n_chunks = (n + CHUNK - 1) / CHUNK;
    if (n_chunks < 4) {
        for (int64_t chunk = 0; chunk < n_chunks; ++chunk) {
            const int64_t done = chunk * CHUNK;
            const int64_t n_now = (n - done < CHUNK) ? (n - done) : CHUNK;
            if (!dequant_f16_chunk(type, src, o16, done, n_now, bs, pb))
                return -1;
        }
    } else {
        Concurrency::parallel_for<int64_t>(0, n_chunks, [&](int64_t chunk) {
            const int64_t done = chunk * CHUNK;
            const int64_t n_now = (n - done < CHUNK) ? (n - done) : CHUNK;
            dequant_f16_chunk(type, src, o16, done, n_now, bs, pb);
        });
    }
    return 0;
}
