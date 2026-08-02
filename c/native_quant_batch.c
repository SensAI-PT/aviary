#include "native_quant_batch.h"

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

static void batch_tables(float fp4[16], float fp8[256], float e8[256]) {
    for (int i = 0; i < 16; i++) fp4[i] = coli_e2m1_decode((uint8_t)i);
    for (int i = 0; i < 256; i++) {
        fp8[i] = coli_e4m3fn_decode((uint8_t)i);
        e8[i] = coli_e8m0_decode((uint8_t)i);
    }
}

int coli_fp8_matmul_batch_ref(float *outputs, const ColiTensorView *weight,
                              const float *inputs, int batch) {
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
                inputs + (size_t)item * columns, columns, 128) != 0) {
            free(activation_scales); free(activations); return -1;
        }
    float fp4_unused[16], fp8[256], e8[256];
    batch_tables(fp4_unused, fp8, e8);
    const uint8_t *data = weight->data, *scales = weight->scales;
    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < weight->rows; row++) {
        float sums[64] = {0};
        size_t row_data = (size_t)row * columns;
        size_t scale_row = (size_t)row / 128;
        for (size_t base = 0; base < columns; base += 128) {
            float scale = e8[scales[scale_row * scale_columns + base / 128]];
            for (size_t offset = 0; offset < 128; offset++) {
                size_t column = base + offset;
                float value = fp8[data[row_data + column]];
                for (int item = 0; item < batch; item++)
                    sums[item] += activations[(size_t)item * columns + column] *
                                  value * scale;
            }
        }
        for (int item = 0; item < batch; item++)
            outputs[(size_t)item * rows + (size_t)row] = sums[item];
    }
    free(activation_scales); free(activations); return 0;
}

int coli_fp4_matmul_batch_ref(float *outputs, const ColiTensorView *weight,
                              const float *inputs, int batch) {
    if (!outputs || !weight || !inputs || batch < 1 || batch > 64 ||
        weight->format != COLI_TENSOR_FP4_NATIVE_BLOCK ||
        weight->scale_format != COLI_SCALE_UE8M0 ||
        !weight->data || !weight->scales || weight->rows < 1 ||
        weight->columns < 1 || weight->columns % 128 ||
        weight->block_rows != 1 || weight->block_columns != 32) return -1;
    size_t rows = (size_t)weight->rows, columns = (size_t)weight->columns;
    size_t packed_stride = columns / 2, scale_stride = columns / 32;
    if (weight->data_bytes != rows * packed_stride ||
        weight->scale_bytes != rows * scale_stride) return -1;
    float *activations = malloc((size_t)batch * columns * sizeof(*activations));
    uint8_t *activation_scales = malloc((size_t)batch * columns / 128);
    if (!activations || !activation_scales) {
        free(activation_scales); free(activations); return -1;
    }
    for (int item = 0; item < batch; item++)
        if (coli_fp8_activation_qdq_ref(
                activations + (size_t)item * columns,
                activation_scales + (size_t)item * columns / 128,
                inputs + (size_t)item * columns, columns, 128) != 0) {
            free(activation_scales); free(activations); return -1;
        }
    matmul_mxfp4(outputs, activations, weight->data, weight->scales,
                 batch, (int)columns, (int)rows);
    free(activation_scales); free(activations); return 0;
}
