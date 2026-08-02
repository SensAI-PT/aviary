#ifdef __AVX512F__
#include <immintrin.h>
#endif

#define coli_fp8_matmul_batch_ref coli_fp8_matmul_batch_baseline_v6
/* ---- begin inlined native_quant_batch_interleaved_v4.c ---- */
/* Keep the established FP8 path and replace only FP4 batch execution. */
#define coli_fp4_matmul_batch_ref coli_fp4_matmul_batch_baseline_v4
#include "native_quant_batch.c"
#undef coli_fp4_matmul_batch_ref

int coli_fp4_matmul_batch_ref(float *outputs, const ColiTensorView *weight,
                              const float *inputs, int batch) {
    return coli_fp4_matmul_batch_baseline_v4(outputs, weight, inputs, batch);
}
/* ---- end inlined native_quant_batch_interleaved_v4.c ---- */

#undef coli_fp8_matmul_batch_ref

int coli_fp8_matmul_batch_ref(float *outputs, const ColiTensorView *weight,
                              const float *inputs, int batch) {
#ifndef __AVX512F__
    return coli_fp8_matmul_batch_baseline_v6(outputs, weight, inputs, batch);
#else
    if (!outputs || !weight || !inputs || batch < 1 || batch > 64 ||
        weight->format != COLI_TENSOR_FP8_E4M3_BLOCK ||
        weight->scale_format != COLI_SCALE_UE8M0 ||
        !weight->data || !weight->scales || weight->rows < 1 ||
        weight->columns < 1 || weight->columns % 128 ||
        weight->block_rows != 128 || weight->block_columns != 128) return -1;
    size_t rows = (size_t)weight->rows, columns = (size_t)weight->columns;
    size_t scale_rows = (rows + 127) / 128;
    size_t scale_columns = columns / 128;
    if (weight->data_bytes != rows * columns ||
        weight->scale_bytes != scale_rows * scale_columns) return -1;

    float *activations = malloc((size_t)batch * columns * sizeof(*activations));
    uint8_t *activation_scales = malloc((size_t)batch * scale_columns);
    if (!activations || !activation_scales) {
        free(activation_scales); free(activations); return -1;
    }
    for (int item = 0; item < batch; item++)
        if (coli_fp8_activation_qdq_ref(
                activations + (size_t)item * columns,
                activation_scales + (size_t)item * scale_columns,
                inputs + (size_t)item * columns, columns, 128)) {
            free(activation_scales); free(activations); return -1;
        }

    float fp8[256], e8[256];
    for (int i = 0; i < 256; i++) {
        fp8[i] = coli_e4m3fn_decode((uint8_t)i);
        e8[i] = coli_e8m0_decode((uint8_t)i);
    }
    const uint8_t *data = weight->data, *scales = weight->scales;
    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < weight->rows; row++) {
        float sums[64] = {0};
        size_t row_data = (size_t)row * columns;
        size_t scale_row = (size_t)row / 128;
        for (size_t base = 0; base < columns; base += 128) {
            __m512 scale = _mm512_set1_ps(
                e8[scales[scale_row * scale_columns + base / 128]]);
            for (size_t chunk = 0; chunk < 128; chunk += 16) {
                __m128i packed_codes = _mm_loadu_si128(
                    (const __m128i *)(data + row_data + base + chunk));
                __m512i codes = _mm512_cvtepu8_epi32(packed_codes);
                __m512 values = _mm512_i32gather_ps(codes, fp8, 4);
                for (int item = 0; item < batch; item++) {
                    __m512 activation = _mm512_loadu_ps(
                        activations + (size_t)item * columns + base + chunk);
                    __m512 product = _mm512_mul_ps(
                        _mm512_mul_ps(activation, values), scale);
                    float products[16];
                    _mm512_storeu_ps(products, product);
                    for (int lane = 0; lane < 16; lane++)
                        sums[item] += products[lane];
                }
            }
        }
        for (int item = 0; item < batch; item++)
            outputs[(size_t)item * rows + (size_t)row] = sums[item];
    }
    free(activation_scales); free(activations);
    return 0;
#endif
}
