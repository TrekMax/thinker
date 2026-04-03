import math
import numpy as np
from typing import List
from ...graph import Tensor
from .utils import calc_expr
from ...enum_defines import DevType, MemType
from ...xsympy import is_sympy
from .base import iqBinaryOperator, register_op, BaseLayout

@register_op
class iqAdd(iqBinaryOperator, BaseLayout):
    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on inputs."""
        inputs = self.inputs
        assert len(inputs) == 2, "iqAdd operator must have exactly two inputs"

        X1 = inputs[0]
        X2 = inputs[1]

        # Expand shapes to the same dimension
        shape1 = list(X1.shape)
        shape2 = list(X2.shape)
        if len(shape1) > len(shape2):
            shape2 = [1] * (len(shape1) - len(shape2)) + shape2
        else:
            shape1 = [1] * (len(shape2) - len(shape1)) + shape1

        assert len(shape1) == len(shape2), "Shapes must have the same dimensions after expansion"

        # Check shape compatibility
        diff_count = 0
        for i in range(len(shape1)):
            if is_sympy(shape1[i]) and is_sympy(shape2[i]):
                temp_shape1 = calc_expr(str(shape1[i]), dynamic_shape)
                temp_shape2 = calc_expr(str(shape2[i]), dynamic_shape)
                if temp_shape1 != temp_shape2:
                    diff_count += 1
            elif shape1[i] != shape2[i]:
                assert shape1[i] == 1 or shape2[i] == 1, "Incompatible dimensions"
                diff_count += 1
        assert diff_count <= 1, "iqAdd does not support this type of broadcasting"

        # Process scales
        scale_x = self.attrs.get('scale_x', 1.0)
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
        if X1.scale != -1:
            assert X1.scale == int(temp), "Input scale must match attribute scale_x"
        else:
            X1.scale = int(temp)

        scale_y = self.attrs.get('scale_y', 1.0)
        temp = math.log(scale_y, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
        X2.scale = int(temp)

        scale_o = self.attrs.get("scale_o", 1.0)
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"

        Y = X1.clone(shape=tuple(shape1), scale=temp)
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the iqAdd operation."""
        x1 = self.inputs[0]
        x2 = self.inputs[1]
        size = x1.nbytes
        Y = self.outputs[0]

        scale_x = self.attrs["scale_x"]
        scale_y = self.attrs["scale_y"]
        scale_o = self.attrs["scale_o"]
        platform = self.attrs.get("platform", "venus")

        workspace_size = 0
        if platform == "venusA":
            if (x1.mem_type==x2.mem_type) and scale_x==scale_o and scale_y==scale_o:
                if Y.mem_type!=MemType.SHARE_MEM:   
                    workspace_size = Y.nbytes
            elif scale_y==scale_o and x2.mem_type==MemType.SHARE_MEM:
                if Y.mem_type!=MemType.SHARE_MEM:   
                    workspace_size = Y.nbytes
            elif scale_x==scale_o and x1.mem_type==MemType.SHARE_MEM:    
                if Y.mem_type!=MemType.SHARE_MEM:   
                    workspace_size = Y.nbytes
            else:
                if Y.mem_type!=MemType.SHARE_MEM:
                    workspace_size = Y.nbytes * 2
                else:
                    workspace_size = Y.nbytes

            workspace_size = min(workspace_size, 65536)
        elif platform == "venus":
            # Venus platform workspace calculation based on iqadd.h logic
            # Check if inputs need PSRAM to SHARE_MEM copy or scale conversion
            x1_need_workspace = (scale_x != scale_o) or (x1.mem_type != MemType.SHARE_MEM)
            x2_need_workspace = (scale_y != scale_o) or (x2.mem_type != MemType.SHARE_MEM)
            y_in_psram = (Y.mem_type != MemType.SHARE_MEM)

            if y_in_psram:
                # Output in PSRAM needs workspace for computation
                if x1_need_workspace and x2_need_workspace:
                    # Need space for Y + processed X1 + processed X2 = 2*size
                    # (Y at offset 0, X2 at offset size since X1 already processed in-place)
                    workspace_size = size * 2
                else:
                    # Only need space for Y result before copying to PSRAM
                    workspace_size = size
            else:
                # Output in SHARE_MEM, only need workspace for input processing
                if x1_need_workspace and x2_need_workspace:
                    # Need space for both processed inputs
                    workspace_size = size
        elif platform == "arcs":
            # Arcs platform workspace calculation based on arcs/iqadd.h implementation
            # Arcs uses chunked processing and requires workspace for PSRAM inputs/scale conversion

            x1_in_psram = (x1.mem_type != MemType.SHARE_MEM)
            x2_in_psram = (x2.mem_type != MemType.SHARE_MEM)
            y_in_psram = (Y.mem_type != MemType.SHARE_MEM)

            scale_x_eq = (scale_x == scale_o)
            scale_y_eq = (scale_y == scale_o)

            workspace_size = 0

            if y_in_psram:
                # Y in PSRAM: need workspace for temporary output
                workspace_size = size
                # If need to process either input, need additional space
                if ((not scale_x_eq) or x1_in_psram) and ((not scale_y_eq) or x2_in_psram):
                    workspace_size = size * 2
            elif ((not scale_x_eq) or x1_in_psram) and ((not scale_y_eq) or x2_in_psram):
                workspace_size = size

            workspace_size = min(workspace_size, 65536)

        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]
        return []

__all__ = ["iqAdd"]