import math
import numpy as np
from typing import List
from ...graph import Tensor
from ...xsympy import is_sympy
from .utils import QuantType, RoundMethod, calc_expr
from ...enum_defines import DevType, MemType
from .base import Operator, OperatorAttrs, register_op
from ...resource_packer._type._ctype import tffi

class GeluOperatorAttrs(OperatorAttrs):
    """Attributes for Gelu operator."""

    def __init__(self, attrs={}):
        """Initialize the Gelu operator attributes."""
        self.attrs = attrs

    def checkparams(self) -> None:
        """Check if required parameters are present."""
        required_attrs = ["scale_x", "scale_o", "platform", "quant_mode"]
        for attr in required_attrs:
            assert attr in self.attrs, f"Missing required attribute: {attr}"


@register_op
class QGelu(Operator):
    """Quantized Gaussian Error Linear Unit (GELU) activation function.

    GELU(x) = x * Φ(x) where Φ is the Gaussian CDF.
    For quantized implementation, this is computed using integer arithmetic.

    Note: This operator only supports venusA platform.

    Attributes:
        o_bits: output bit width (default: 8)
        platform: target platform (only 'venusA' supported)
        quant_mode: quantization mode ('floor', 'floor_add', 'round', 'ceil')
        scale_o: output scale (power of 2)
        scale_x: input scale (power of 2)
        x_bits: input bit width (default: 8)
    """

    def __init__(self, attrs={}):
        """Initialize the Gelu operator with given attributes."""
        self.attrs = GeluOperatorAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on inputs."""
        inputs = self.inputs
        assert len(inputs) == 1, "Gelu operator must have exactly one input"
        platform = self.attrs.get("platform", "venus")

        # Only venusA platform is supported
        assert platform == "venusA", "Gelu operator only supports venusA platform"

        X = inputs[0]

        # Check input data type based on x_bits
        x_bits = self.attrs.get("x_bits", 8)
        assert x_bits in {8, 16, 32}, "input bits must be 8, 16 or 32"

        if x_bits == 8:
            expected_dtype = np.int8
        elif x_bits == 16:
            expected_dtype = np.int16
        else:
            expected_dtype = np.int32

        assert X.dtype == np.dtype(expected_dtype), \
            f"input data type of Gelu must be {expected_dtype} for x_bits={x_bits}"

        # Process input scale
        scale_x = self.attrs.get("scale_x", 1.0)
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
        assert X.scale == int(temp), "Input scale must match attribute scale_x"

        # Process output scale
        scale_o = self.attrs.get("scale_o", 1.0)
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"

        # Check output bits
        o_bits = self.attrs.get("o_bits", 8)
        assert o_bits in {8, 16, 32}, "output bits must be 8, 16 or 32"

        if o_bits == 8:
            out_dtype = np.int8
        elif o_bits == 16:
            out_dtype = np.int16
        else:
            out_dtype = np.int32

        Y = X.clone(scale=int(temp), dtype=out_dtype)
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:

        X = self.inputs[0]
        data_size = np.prod(X.shape)
        
        if X.dtype == np.int8:
            workspace_size = data_size * 6
        elif X.dtype == np.int16:
            workspace_size = data_size * 4
        elif X.dtype == np.int32:
            if X.scale != 27:
                workspace_size =  data_size * 4
            else:
                workspace_size = 0
        return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]

    def flops_counter(self, dynamic_shape) -> int:
        """Calculate the number of floating-point operations (FLOPs) for the Gelu operation."""
        X = self.inputs[0]
        Y = self.outputs[0]
        xshape = list(X.shape)
        yshape = list(Y.shape)

        # Resolve symbolic expressions in shapes
        for i, s in enumerate(xshape):
            if is_sympy(s):
                xshape[i] = calc_expr(str(s), dynamic_shape)
        for i, s in enumerate(yshape):
            if is_sympy(s):
                yshape[i] = calc_expr(str(s), dynamic_shape)

        # GELU requires multiple operations: multiply, add, erf approximation
        # Estimated as ~8 operations per element
        flops = int(np.prod(yshape)) * 8
        return flops


__all__ = ["QGelu"]