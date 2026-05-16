import numpy as np

from ...xsympy import is_sympy
from .base import Operator, OperatorAttrs, register_op

@register_op
class Tile(Operator):
    def __init__(self, attrs={}):
        """Initialize Tile operator with given attributes."""
        self.attrs = OperatorAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer output tensor shape by tiling input tensor according to repeats."""
        inputs = self.inputs
        assert len(inputs) == 2, "Tile operation requires exactly two inputs"
        X = inputs[0]
        repeats = inputs[1].data
        platform = self.attrs.get("platform", "venus")
        if platform == "venus":
            assert X.dtype in (np.int8, np.float32), "Tile on venus only supports int8 or float32 input"
            assert inputs[1].dtype == np.int64, "Tile on venus requires int64 repeats"
            assert len(X.shape) == inputs[1].shape[0], "Tile repeat length must equal input rank"

        # Calculate output shape by tiling input shape
        yshape = X.shape * repeats

        # Convert symbolic dimensions to integers if possible
        yshape = [int(s) if not is_sympy(s) else s for s in yshape]

        # Create output tensor with the new shape
        Y = X.clone(shape=tuple(yshape))
        self.outputs = [Y]
