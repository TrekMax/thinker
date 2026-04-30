#undef __OP__
#define __OP__ GRUInt
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

#ifdef THINKER_USE_VENUS
#include "./venus/gruint.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/gruint.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/gruint.h"
#endif

static int32_t aligned_tensor_bytes(tTensor* tensor) {
    return ALIGN16(getShapeSize(&(tensor->shape_)) * tensor->byte_);
}

/**
 * Forward pass implementation for Gated Recurrent Unit Integer operator
 * Performs GRU computation with integer quantization
 * @param op: Operator structure containing GRU attributes
 * @param tensors: Array of input/output tensors (input, i2h_weights, h2h_weights, i2h_bias, h2h_bias, output, hidden_output, optional workspace)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list for weight data handling
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator* op, tTensor** tensors, int32_t num_tensor, tDMA_List* list) {
    // Validate tensor count
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
    
    // Get GRU attributes
    GRUIntAttrs* attr = (GRUIntAttrs*)((int8_t*)op + op->attr_offset_);
    int32_t weight_idx = 0;
    if(op->num_input_ == 6)
    {
        weight_idx = 1;
    }
    // Get all tensor pointers
    tTensor* input = tensors[0];
    tTensor* i2h_w = tensors[weight_idx + 1];
    tTensor* h2h_w = tensors[weight_idx + 2];
    tTensor* i2h_bias = tensors[weight_idx + 3];
    tTensor* h2h_bias = tensors[weight_idx + 4];

    tTensor* output = tensors[op->num_input_];
    tTensor* hidden_o = tensors[op->num_input_ + 1];

    // Get workspace tensor if present
    tTensor* workspace = NULL;
    if (num_tensor > op->num_input_ + op->num_output_) {
        workspace = tensors[op->num_input_ + op->num_output_];
    }

    // Initialize dummy tensors for hidden state and mask
    tTensor hidden_i_inst;
    hidden_i_inst.shape_.ndim_ = 0;
    tTensor *hidden_in = &hidden_i_inst;
    if(weight_idx == 1)
        hidden_in = tensors[1];
    
    tTensor mask;
    mask.shape_.ndim_ = 0;
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
    if (list->total_ != 0)
        getWeightData(list, 0);
#endif
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

#ifdef THINKER_USE_VENUS
    // Venus hardware implementation
    if (num_tensor > op->num_input_ + op->num_output_) {
        workspace = tensors[op->num_input_ + op->num_output_];
        tTensor *dma_temp   = ((tTensor**)tensors)[op->num_input_ + op->num_output_ + 1];
        tTensor i2h_w_temp  = i2h_w[0];
        i2h_w_temp.dptr_    = (addr_type)(dma_temp->dptr_);

        dma_temp  = ((tTensor**)tensors)[op->num_input_ + op->num_output_ + 2];
        tTensor i2h_bias_temp = i2h_bias[0];
        i2h_bias_temp.dptr_ = (addr_type)(dma_temp->dptr_);

        dma_temp  = ((tTensor**)tensors)[op->num_input_ + op->num_output_ + 3];
        tTensor h2h_w_temp  = h2h_w[0];
        h2h_w_temp.dptr_    = (addr_type)(dma_temp->dptr_);

        dma_temp  = ((tTensor**)tensors)[op->num_input_ + op->num_output_ + 4];
        tTensor h2h_bias_temp     = h2h_bias[0];
        h2h_bias_temp.dptr_ = (addr_type)(dma_temp->dptr_);

        THINKER_RET_CHECK(gruint_luna(input, &hidden_i_inst, i2h_w, h2h_w, i2h_bias, h2h_bias,
                      output, hidden_o, attr, workspace), "gruint_luna");
    }
#elif defined(THINKER_USE_ARCS) || defined(THINKER_USE_VENUSA)
    // ARC/VENUSA hardware implementation
    if(list->total_ > 0) {
        if (num_tensor > op->num_input_ + op->num_output_) {
            workspace = tensors[op->num_input_ + op->num_output_];
            tTensor *dma_temp   = ((tTensor**)tensors)[op->num_input_ + op->num_output_ + 1];
            tTensor i2h_w_temp  = i2h_w[0];
            i2h_w_temp.dptr_    = (addr_type)(dma_temp->dptr_);
            tTensor h2h_w_temp  = h2h_w[0];
            h2h_w_temp.dptr_    = (addr_type)((int8_t *)i2h_w_temp.dptr_ + aligned_tensor_bytes(&i2h_w_temp));
            tTensor i2h_bias_temp = i2h_bias[0];
            i2h_bias_temp.dptr_ = (addr_type)((int8_t *)h2h_w_temp.dptr_ + aligned_tensor_bytes(&h2h_w_temp));
            tTensor h2h_bias_temp     = h2h_bias[0];
            h2h_bias_temp.dptr_ = (addr_type)((int8_t *)i2h_bias_temp.dptr_ + aligned_tensor_bytes(&i2h_bias_temp));

            THINKER_RET_CHECK(gruint_luna(input, hidden_in, &i2h_w_temp, &h2h_w_temp, &i2h_bias_temp, &h2h_bias_temp,
                          output, hidden_o, attr, workspace), "gruint_luna");
        }
    }
    else {
        if (num_tensor > op->num_input_ + op->num_output_) {
            workspace = tensors[op->num_input_ + op->num_output_];
        }

        THINKER_RET_CHECK(gruint_luna(input, hidden_in, i2h_w, h2h_w, i2h_bias, h2h_bias,
                          output, hidden_o, attr, workspace), "gruint_luna");
    }
#endif

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","GruInt", total_t);  
#endif
    
    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__