import math
import numpy as np
from typing import List, Dict

from ...graph import Tensor
from ...enum_defines import DevType, MemType
from .base import UnaryOperator, BaseLayout, register_op


@register_op
class Relu(UnaryOperator, BaseLayout):
    """Rectified Linear Unit (Relu) activation function."""
    def infer_tensor(self, dynamic_shape: Dict[str, int]):
        """Infer output tensor shape and data."""
        assert len(self.inputs) == 1, "Unary operator must have exactly one input"
        X = self.inputs[0]
        platform = self.attrs.get("platform", "venus")
        if platform == "arcs":
            assert X.dtype != np.int16, "input data type of Relu do not support int16 for arcs"
        else:
            assert X.dtype in (np.int8, np.int16, np.int32), "input data type of Relu can be int8/int16/int32"

        scale_x = self.attrs.get('scale_x', 1.0)
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 1e-6, "scale_x must be a power of 2"
        # if X.scale != -1:
        #     assert X.scale == int(temp), "Input scale mismatch"
        # else:
        #     X.scale = int(temp)

        scale_o = self.attrs.get("scale_o", 1.0)
        temp = math.log(scale_o, 2)
        # assert abs(temp - int(temp)) < 1e-6, "scale_o must be a power of 2"

        out_bits = self.attrs.get('o_bits', 8)
        if out_bits == 8:
            out_dtype = np.int8
        elif out_bits == 16:
            out_dtype = np.int16
        elif out_bits == 32:
            out_dtype = np.int32
        else:
            assert False, "output type of relu must be int8 or int16 or int32"

        # Y = X.clone(scale = int(temp), dtype = out_dtype)
        Y = X.clone()
        self.outputs = [Y]

    def get_workspace(self):
        """Calculate the required workspace size."""
        data = self.inputs[0]
        output = self.outputs[0]
        data_size = np.prod(data.shape)
        workspace_size = 0
        platform = self.attrs.get("platform", "venus")

        if data.mem_type != MemType.SHARE_MEM or output.mem_type != MemType.SHARE_MEM:
            assert data.dtype == np.int8 and output.dtype== np.int8,\
            "input data type of Relu must be int8 for venus"
            workspace_size = data_size
        else:
            workspace_size = 0

        workspace_size = min(65536, workspace_size)
        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]
        return []


@register_op
class PRelu(UnaryOperator, BaseLayout):
    """Parametric Rectified Linear Unit (PRelu) activation function."""
    def get_workspace(self):
        data = self.inputs[0]
        output = self.outputs[0]
        assert data.mem_type == MemType.SHARE_MEM and output.mem_type == MemType.SHARE_MEM,\
        "mem type of PRelu input/output must be share memory"
        return []

@register_op
class ReluX(UnaryOperator, BaseLayout):
    """Parametric Rectified Linear Unit (ReluX) activation function."""
    def get_workspace(self):
        data = self.inputs[0]
        output = self.outputs[0]
        assert data.mem_type == MemType.SHARE_MEM and output.mem_type == MemType.SHARE_MEM,\
        "mem type of ReluX input/output must be share memory"
        return []

@register_op
class Sigmoid(UnaryOperator, BaseLayout):
    pass       

@register_op
class Tanh(UnaryOperator, BaseLayout):
    pass 

__all__ = ["Relu", "PRelu", "ReluX", "Sigmoid", "Tanh"]