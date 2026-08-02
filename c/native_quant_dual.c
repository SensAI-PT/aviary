#include "native_quant_dual.h"

#include <stdint.h>
#include <stdlib.h>

#include "native_quant.h"
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
#include "quant.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

static int dual_same_shape(const ColiTensorView *a, const ColiTensorView *b) {
    return a && b && a->rows == b->rows && a->columns == b->columns &&
           a->block_rows == b->block_rows &&
           a->block_columns == b->block_columns;
}

int coli_fp4_dual_matvec_ref(float *output_a, float *output_b,
                             const ColiTensorView *a,
                             const ColiTensorView *b,
                             const float *input) {
    if (!output_a || !output_b || !input || !dual_same_shape(a, b) ||
        a->format != COLI_TENSOR_FP4_NATIVE_BLOCK ||
        b->format != COLI_TENSOR_FP4_NATIVE_BLOCK ||
        a->scale_format != COLI_SCALE_UE8M0 ||
        b->scale_format != COLI_SCALE_UE8M0 || !a->data || !b->data ||
        !a->scales || !b->scales || a->rows < 1 || a->columns < 1 ||
        a->columns % 128 || a->block_rows != 1 || a->block_columns != 32)
        return -1;
    size_t rows = (size_t)a->rows, columns = (size_t)a->columns;
    size_t packed_stride = columns / 2, scale_stride = columns / 32;
    if (a->data_bytes != rows * packed_stride ||
        b->data_bytes != rows * packed_stride ||
        a->scale_bytes != rows * scale_stride ||
        b->scale_bytes != rows * scale_stride) return -1;
    float *activation = malloc(columns * sizeof(*activation));
    uint8_t *activation_scales = malloc(columns / 128);
    if (!activation || !activation_scales) {
        free(activation_scales); free(activation); return -1;
    }
    if (coli_fp8_activation_qdq_ref(activation, activation_scales,
                                    input, columns, 128) != 0) {
        free(activation_scales); free(activation); return -1;
    }
    matmul_mxfp4(output_a, activation, a->data, a->scales,
                 1, (int)columns, (int)rows);
    matmul_mxfp4(output_b, activation, b->data, b->scales,
                 1, (int)columns, (int)rows);
    free(activation_scales); free(activation); return 0;
}

int coli_fp8_dual_matvec_ref(float *output_a, float *output_b,
                             const ColiTensorView *a,
                             const ColiTensorView *b,
                             const float *input) {
    if (!output_a || !output_b || !input || !dual_same_shape(a, b) ||
        a->format != COLI_TENSOR_FP8_E4M3_BLOCK ||
        b->format != COLI_TENSOR_FP8_E4M3_BLOCK ||
        a->scale_format != COLI_SCALE_UE8M0 ||
        b->scale_format != COLI_SCALE_UE8M0 || !a->data || !b->data ||
        !a->scales || !b->scales || a->rows < 1 || a->columns < 1 ||
        a->columns % 128 || a->block_rows != 128 ||
        a->block_columns != 128) return -1;
    size_t rows = (size_t)a->rows, columns = (size_t)a->columns;
    size_t scale_rows = (rows + 127) / 128, scale_columns = columns / 128;
    if (a->data_bytes != rows * columns || b->data_bytes != rows * columns ||
        a->scale_bytes != scale_rows * scale_columns ||
        b->scale_bytes != scale_rows * scale_columns) return -1;
    float *activation = malloc(columns * sizeof(*activation));
    uint8_t *activation_scales = malloc(scale_columns);
    if (!activation || !activation_scales) {
        free(activation_scales); free(activation); return -1;
    }
    if (coli_fp8_activation_qdq_ref(activation, activation_scales,
                                    input, columns, 128) != 0) {
        free(activation_scales); free(activation); return -1;
    }
    float fp8[256], e8[256];
    for (int i = 0; i < 256; i++) {
        fp8[i] = coli_e4m3fn_decode((uint8_t)i);
        e8[i] = coli_e8m0_decode((uint8_t)i);
    }
    const uint8_t *data_a = a->data, *data_b = b->data;
    const uint8_t *scales_a = a->scales, *scales_b = b->scales;
    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < a->rows; row++) {
        size_t row_data = (size_t)row * columns;
        size_t scale_row = (size_t)row / 128;
        float sum_a = 0.0f, sum_b = 0.0f;
        for (size_t base = 0; base < columns; base += 128) {
            size_t scale_index = scale_row * scale_columns + base / 128;
            float scale_a = e8[scales_a[scale_index]];
            float scale_b = e8[scales_b[scale_index]];
            for (size_t offset = 0; offset < 128; offset++) {
                size_t column = base + offset;
                float x = activation[column];
                sum_a += x * fp8[data_a[row_data + column]] * scale_a;
                sum_b += x * fp8[data_b[row_data + column]] * scale_b;
            }
        }
        output_a[row] = sum_a; output_b[row] = sum_b;
    }
    free(activation_scales); free(activation); return 0;
}
