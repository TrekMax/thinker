#undef __OP__
#define __OP__ Expand
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "thinker_status.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "core/comm/utils.h"

#if THINKER_USE_ARCS || THINKER_USE_VENUSA
#include "arcs/luna/opi_psram_cpy.h"
#endif

#ifdef THINKER_USE_ARCS
#include "arcs/luna/luna_misc_math.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/luna/luna_matrix_math.h"
#include "./venusA/luna/luna_misc_math.h"
#endif

#define EXPAND_MAX_DIMS 7

static int32_t expand_checked_mul_size(size_t lhs, size_t rhs, size_t *out) {
    if (rhs != 0 && lhs > ((size_t)INT_MAX / rhs)) {
        return T_ERR_INVALID_DATA;
    }
    *out = lhs * rhs;
    return T_SUCCESS;
}

static int32_t expand_copy_bytes(uint8_t *dst, const uint8_t *src, size_t size,
                                 bool dst_in_psram) {
    if (size == 0 || dst == src) {
        return T_SUCCESS;
    }

#if THINKER_USE_ARCS || THINKER_USE_VENUSA
    if (dst_in_psram) {
        opi_psram_cpy_out(dst, (void *)src, (int32_t)size);
    } else {
        return luna_memcpy_i8o8((int8_t *)dst, (int8_t *)src, (uint32_t)size);
    }
#else
    memcpy(dst, src, size);
#endif
    return T_SUCCESS;
}

static int32_t expand_repeat_block(uint8_t *dst, size_t block_bytes,
                                   size_t repeat, bool dst_in_psram) {
    size_t filled = block_bytes;
    size_t total = block_bytes * repeat;

    while (filled < total) {
        size_t copy_bytes = total - filled;
        if (copy_bytes > filled) {
            copy_bytes = filled;
        }
        int32_t ret = expand_copy_bytes(dst + filled, dst, copy_bytes, dst_in_psram);
        if (ret != T_SUCCESS) {
            return ret;
        }
        filled += copy_bytes;
    }
    return T_SUCCESS;
}

static int32_t expand_shapes_equal_from(int32_t dim, int32_t ndim,
                                        const uint32_t *in_shape,
                                        const uint32_t *out_shape) {
    for (int32_t i = dim; i < ndim; ++i) {
        if (in_shape[i] != out_shape[i]) {
            return 0;
        }
    }
    return 1;
}

static int32_t expand_copy_dim(uint8_t *dst, const uint8_t *src, int32_t dim,
                               int32_t ndim, const uint32_t *in_shape,
                               const uint32_t *out_shape,
                               const size_t *in_stride_bytes,
                               const size_t *out_stride_bytes,
                               size_t elem_bytes, bool dst_in_psram) {
    if (dim == ndim) {
        return expand_copy_bytes(dst, src, elem_bytes, dst_in_psram);
    }

    if (expand_shapes_equal_from(dim, ndim, in_shape, out_shape)) {
        return expand_copy_bytes(dst, src, out_stride_bytes[dim] * out_shape[dim],
                                 dst_in_psram);
    }

    if (in_shape[dim] == out_shape[dim]) {
        for (uint32_t i = 0; i < out_shape[dim]; ++i) {
            int32_t ret = expand_copy_dim(dst + i * out_stride_bytes[dim],
                                          src + i * in_stride_bytes[dim],
                                          dim + 1, ndim, in_shape, out_shape,
                                          in_stride_bytes, out_stride_bytes,
                                          elem_bytes, dst_in_psram);
            if (ret != T_SUCCESS) {
                return ret;
            }
        }
    } else {
        int32_t ret = expand_copy_dim(dst, src, dim + 1, ndim, in_shape,
                                      out_shape, in_stride_bytes,
                                      out_stride_bytes, elem_bytes,
                                      dst_in_psram);
        if (ret != T_SUCCESS) {
            return ret;
        }
        return expand_repeat_block(dst, out_stride_bytes[dim], out_shape[dim],
                                   dst_in_psram);
    }
    return T_SUCCESS;
}

/**
 * Forward pass implementation for Expand operator
 * Expands input tensor to match target shape by repeating elements
 * @param op: Operator structure containing expansion attributes
 * @param tensors: Array of input/output tensors (input, output, optional workspace)
 * @param num_tensor: Total number of tensors (must be 3)
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    (void)list;

    // Validate tensor count
    CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_));

    // Get input and output tensors
    tTensor *X = (tTensor *)tensors[0];
    tTensor *Y = (tTensor *)tensors[op->num_input_];
    bool dst_in_psram = Y->mem_.type_ != 2;

    // Get shape information
    int32_t xdim = X->shape_.ndim_;
    int32_t ydim = Y->shape_.ndim_;
    const uint32_t *tShape = X->shape_.dims_;
    const uint32_t *yshape = Y->shape_.dims_;

#ifdef RUNTIME_PARAM_CHECK
    if (xdim <= 0 || ydim < xdim || xdim > EXPAND_MAX_DIMS || ydim > EXPAND_MAX_DIMS) {
        return T_ERR_INVALID_PARA;
    }

    if (X->dtype_ != Y->dtype_ || X->byte_ != Y->byte_ || X->byte_ == 0) {
        return T_ERR_INVALID_DATATYPE;
    }
#endif

    // Calculate leading dimension multiplier
    int32_t bl = ydim - xdim;
    size_t leading = 1;
    for (int32_t i = 0; i < bl; ++i) {
        if (expand_checked_mul_size(leading, yshape[i], &leading) != T_SUCCESS) {
            return T_ERR_INVALID_DATA;
        }
    }

    const uint32_t *expandshape = yshape + bl;
    for (int32_t i = 0; i < xdim; ++i) {
        if (tShape[i] != expandshape[i]) {
            if (tShape[i] != 1 || expandshape[i] == 0) {
                return T_ERR_INVALID_DATA;
            }
        }
    }

    size_t base_elems = 1;
    for (int32_t i = 0; i < xdim; ++i) {
        if (expand_checked_mul_size(base_elems, expandshape[i], &base_elems) != T_SUCCESS) {
            return T_ERR_INVALID_DATA;
        }
    }

    size_t base_bytes = 0;
    if (expand_checked_mul_size(base_elems, X->byte_, &base_bytes) != T_SUCCESS) {
        return T_ERR_INVALID_DATA;
    }

    size_t total_bytes = 0;
    if (expand_checked_mul_size(base_bytes, leading, &total_bytes) != T_SUCCESS) {
        return T_ERR_INVALID_DATA;
    }
    if (total_bytes == 0) {
        return T_SUCCESS;
    }

    uint8_t *output = (uint8_t *)Y->dptr_;
    const uint8_t *input = (const uint8_t *)X->dptr_;

    if (xdim == 0) {
        THINKER_RET_CHECK(expand_copy_bytes(output, input, X->byte_, dst_in_psram),
                          "expand_copy_bytes");
        return expand_repeat_block(output, X->byte_, leading, dst_in_psram);
    }

    size_t input_stride_bytes[EXPAND_MAX_DIMS];
    size_t output_stride_bytes[EXPAND_MAX_DIMS];
    input_stride_bytes[xdim - 1] = X->byte_;
    output_stride_bytes[xdim - 1] = X->byte_;
    for (int32_t i = xdim - 1; i > 0; --i) {
        if (expand_checked_mul_size(input_stride_bytes[i], tShape[i],
                                    &input_stride_bytes[i - 1]) != T_SUCCESS) {
            return T_ERR_INVALID_DATA;
        }
        if (expand_checked_mul_size(output_stride_bytes[i], expandshape[i],
                                    &output_stride_bytes[i - 1]) != T_SUCCESS) {
            return T_ERR_INVALID_DATA;
        }
    }

    THINKER_RET_CHECK(expand_copy_dim(output, input, 0, xdim, tShape,
                                      expandshape, input_stride_bytes,
                                      output_stride_bytes, X->byte_,
                                      dst_in_psram),
                      "expand_copy_dim");
    THINKER_RET_CHECK(expand_repeat_block(output, base_bytes, leading,
                                          dst_in_psram),
                      "expand_repeat_block");

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
