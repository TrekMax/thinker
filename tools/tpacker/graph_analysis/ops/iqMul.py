import math
import numpy as np
from typing import Any, Dict, Optional, List

from ...graph import Tensor
from ...enum_defines import DevType
from .base import iqBinaryOperator, register_op

from ...xsympy import is_sympy
from .utils import QuantType, calc_expr, RoundMethod

def find_min_q(a):
    # 处理 0 的情况
    if a == 0:
        return 0  # 0 * 2^q = 0，总是整数，q 可任取，通常返回 0
    
    # 取绝对值，符号不影响是否为整数
    a = abs(a)
    
    # 将浮点数转换为分数 p / 2^k
    # 使用 float.as_integer_ratio() 得到精确的 p/q（q 是 2 的幂）
    p, denom = a.as_integer_ratio()
    
    # 现在 a = p / denom，且 denom 是 2 的幂（因为是二进制浮点数）
    # 求 denom = 2^k => k = log2(denom)
    k = denom.bit_length() - 1  # 因为 denom 是 2 的幂，bit_length - 1 = log2
    
    return k, a*pow(2,k)

@register_op
class iqMul(iqBinaryOperator):
    def infer_tensor(self, dynamic_shape: Dict[str, int]):
        """Infer output tensor shape and data."""
        inputs = self.inputs
        assert len(inputs) == 2, "iqBinaryOperator must have exactly two inputs"
        X1, X2 = inputs

        shape1 = list(X1.shape)
        shape2 = list(X2.shape)

        # Expand to the same dimension
        if len(shape1) > len(shape2):
            shape2 = [1] * (len(shape1) - len(shape2)) + shape2
        else:
            shape1 = [1] * (len(shape2) - len(shape1)) + shape1

        assert len(shape1) == len(shape2), "Input shapes must have the same dimensions after expansion"

        shape = []
        for s1, s2 in zip(shape1, shape2):
            if s1 == 1:
                shape.append(s2)
            elif s2 == 1:
                shape.append(s1)
            elif s1 == s2:
                shape.append(s1)
            elif is_sympy(s1) and is_sympy(s2):
                s1_val = calc_expr(str(s1), dynamic_shape)
                s2_val = calc_expr(str(s2), dynamic_shape)
                assert s1_val == s2_val, "Dynamic shapes must match"
                shape.append(s1_val)
            else:
                raise AttributeError("Incompatible shapes")

        scale_x = self.attrs.get('scale_x', 1.0)
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 1e-6, "scale_x must be a power of 2"
        if self.inputs[0].scale != -1:
            assert self.inputs[0].scale == int(temp), "Input scale mismatch"
        else:
            self.inputs[0].scale = int(temp)

        scale_y = self.attrs.get('scale_y', 1.0)
        temp = math.log(scale_y, 2)
        assert abs(temp - int(temp)) < 1e-6, "scale_y must be a power of 2"
        self.inputs[1].scale = int(temp)
        if self.inputs[1].has_data() and self.inputs[1].dtype == np.float32 and self.inputs[1].shape == ():
            shift, new_data = find_min_q(self.inputs[1].data)
            self.attrs['scale_y'] = scale_y * 2 ** shift
            self.inputs[1].data = new_data.astype(self.inputs[0].dtype)
            self.inputs[1].scale = int(temp+shift)
            self.inputs[1].dtype = self.inputs[0].dtype

        scale_o = self.attrs.get("scale_o", 1.0)
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 1e-6, "scale_o must be a power of 2"

        Y = X1.clone(shape=tuple(shape), scale=temp)
        self.outputs = [Y]
        if all(x.has_data() for x in inputs):
            self.forward()

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the iqMul operation."""
        input1 = self.inputs[0]
        input2 = self.inputs[1]
        workspace_size = 0

        # Check if inputs meet specific shape conditions
        if (len(input1.shape) == 4 and 
            len(input2.shape) == 4 and 
            input1.shape[0] == input2.shape[0] and 
            input1.shape[1] == input2.shape[1] and 
            (input1.shape[2] > 1 or input1.shape[3] > 1) and 
            input2.shape[2] == 1 and 
            input2.shape[3] == 1):
            
            # Calculate workspace size based on input dimensions
            workspace_size = input1.shape[2] * input1.shape[3]
            workspace_size += input1.shape[1] * input1.shape[2] * input1.shape[3]

        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, input1.mem_type)]
        return []

__all__ = ["iqMul"]