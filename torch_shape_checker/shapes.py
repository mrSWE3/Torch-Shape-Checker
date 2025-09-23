
from typing import Callable, Dict
import torch

shape_type = torch.Size
shape_transform_type = Callable[[list[shape_type]], shape_type]

ATEN_SHAPE_FUNCS: Dict[str, shape_transform_type] = {}

def register_aten(*op_names: str) -> Callable[[shape_transform_type], shape_transform_type]:
    def decorator(fn: shape_transform_type) -> shape_transform_type:
        for name in op_names:
            ATEN_SHAPE_FUNCS[name] = fn
        return fn
    return decorator

def broadcast_shape(a: shape_type, b: shape_type) -> shape_type:
    max_len = max(len(a), len(b))
    a_pad = (1,) * (max_len - len(a)) + a
    b_pad = (1,) * (max_len - len(b)) + b

    result_shape: list[int] = []
    for dim_a, dim_b in zip(a_pad, b_pad):
        if dim_a == dim_b or dim_a == 1 or dim_b == 1:
            result_shape.append(max(dim_a, dim_b))
        else:
            raise ValueError(f"Shapes {a} and {b} are not broadcastable")

    return torch.Size(result_shape)


@register_aten("aten::add", "aten::mul", "aten::sub", "aten::div", "aten::pow")
def shape_elementwise(shapes: list[shape_type]) -> shape_type:
    assert len(shapes) == 2
    return broadcast_shape(shapes[0], shapes[1])


# Register aten::tensor
@register_aten("aten::tensor")
def shape_tensor(shapes: list[shape_type]) -> shape_type:
    # For aten::tensor, the output shape is usually a scalar (empty shape),
    # but if the input is a list of values, the shape is (N,) where N is the length of the list.
    # Here, we assume the input shape is provided as a single shape argument.
    assert len(shapes) == 1
    return shapes[0]

# Register aten::linear
@register_aten("aten::linear")
def shape_linear(shapes: list[shape_type]) -> shape_type:
    # aten::linear(%input, %weight, %bias)
    # input: (N, in_features)
    # weight: (out_features, in_features)
    # bias: (out_features) or None
    # Output: (N, out_features)
    assert len(shapes) == 3
    input_shape, weight_shape, bias_shape = shapes
    assert len(input_shape) >= 2, f"input must have at least 2 dims, got {input_shape}"
    assert len(weight_shape) == 2, f"weight must have 2 dims, got {weight_shape}"
    in_features = input_shape[1]
    weight_out, weight_in = weight_shape
    assert in_features == weight_in, f"input's in_features ({in_features}) must match weight's in_features ({weight_in})"
    N = input_shape[0]
    out_features = weight_out
    # Output shape: (N, out_features) + input_shape[2:]
    return torch.Size((N, out_features) + tuple(input_shape[2:]))