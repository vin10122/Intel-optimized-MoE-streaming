#include <immintrin.h>
#include <cmath>
#include <cstdint>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif

static float sum8(__m256 value) {
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, value);
    float sum = 0;
    for (float x : lanes) sum += x;
    return sum;
}

API int moe_delta_heads(float* state, const float* q, const float* k,
    const float* v, const float* beta, const float* gate, float* output,
    int heads, int key_dim, int value_dim) {
    if (!state || !q || !k || !v || !beta || !gate || !output ||
        heads < 1 || heads > 256 || key_dim < 1 || key_dim > 1024 ||
        value_dim < 1 || value_dim > 1024) return -1;
    for (int h = 0; h < heads; ++h) {
        const double decay = std::exp(double(gate[h]));
        const __m256d decay4 = _mm256_set1_pd(decay);
        const float* kh = k + h * key_dim;
        const float* qh = q + h * key_dim;
        for (int r = 0; r < value_dim; ++r) {
            float* row = state + (int64_t(h) * value_dim + r) * key_dim;
            __m256 kv8 = _mm256_setzero_ps();
            int j = 0;
            for (; j + 8 <= key_dim; j += 8) {
                const __m256 old = _mm256_loadu_ps(row + j);
                const __m128 lo = _mm256_cvtpd_ps(_mm256_mul_pd(
                    _mm256_cvtps_pd(_mm256_castps256_ps128(old)), decay4));
                const __m128 hi = _mm256_cvtpd_ps(_mm256_mul_pd(
                    _mm256_cvtps_pd(_mm256_extractf128_ps(old, 1)), decay4));
                const __m256 decayed = _mm256_insertf128_ps(
                    _mm256_castps128_ps256(lo), hi, 1);
                _mm256_storeu_ps(row + j, decayed);
                kv8 = _mm256_add_ps(kv8, _mm256_mul_ps(decayed, _mm256_loadu_ps(kh + j)));
            }
            float kv = sum8(kv8);
            for (; j < key_dim; ++j) {
                row[j] = float(double(row[j]) * decay);
                kv += row[j] * kh[j];
            }
            const float delta = beta[h] * (v[h * value_dim + r] - kv);
            const __m256 d8 = _mm256_set1_ps(delta);
            __m256 out8 = _mm256_setzero_ps();
            j = 0;
            for (; j + 8 <= key_dim; j += 8) {
                const __m256 updated = _mm256_add_ps(_mm256_loadu_ps(row + j),
                    _mm256_mul_ps(d8, _mm256_loadu_ps(kh + j)));
                _mm256_storeu_ps(row + j, updated);
                out8 = _mm256_add_ps(out8, _mm256_mul_ps(updated, _mm256_loadu_ps(qh + j)));
            }
            float result = sum8(out8);
            for (; j < key_dim; ++j) {
                row[j] += delta * kh[j];
                result += row[j] * qh[j];
            }
            if (!std::isfinite(result)) return -2;
            output[h * value_dim + r] = result;
        }
    }
    return 0;
}
