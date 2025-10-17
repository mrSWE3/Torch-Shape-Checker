
from typing import Callable, Dict, Any
import torch

shape_type = torch.Size
shape_transform_type = Callable[[list[shape_type]], shape_type]

ATEN_SHAPE_FUNCS: Dict[str, shape_transform_type] = {}

def eval_aten(aten_name:str, typ: str, *args: Any) -> Any:
    aten_op = getattr(torch.ops.aten, aten_name)
    try:
        aten_op = getattr(aten_op, typ)
        kwargs = {a.name:v for a,v in zip(aten_op._schema.arguments, args)}
        return aten_op(**kwargs)
    except:
        return aten_op(*args)
    



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


#@register_aten("add", "mul", "sub", "div", "pow")
def shape_elementwise(shapes: list[shape_type]) -> shape_type:
    assert len(shapes) == 2
    return broadcast_shape(shapes[0], shapes[1])


# Register tensor
#@register_aten("tensor")
def shape_tensor(shapes: list[shape_type]) -> shape_type:
    # For tensor, the output shape is usually a scalar (empty shape),
    # but if the input is a list of values, the shape is (N,) where N is the length of the list.
    # Here, we assume the input shape is provided as a single shape argument.
    assert len(shapes) == 1
    return shapes[0]

#@register_aten("linear")
def shape_linear(shapes: list[shape_type]) -> shape_type:
    # linear(%input, %weight, %bias)
    # input: (..., in_features)
    # weight: (out_features, in_features)
    # bias: (out_features) or None
    # Output: (..., out_features)
    assert len(shapes) == 3
    input_shape, weight_shape, bias_shape = shapes
    assert len(weight_shape) == 2, f"weight must have 2 dims, got {weight_shape}"
    in_features = input_shape[-1]
    weight_out, weight_in = weight_shape
    assert in_features == weight_in, f"input's in_features ({in_features}) must match weight's in_features ({weight_in})"
    # Output shape: input_shape[:-1] + (out_features,)
    return torch.Size(tuple(input_shape[:-1]) + (weight_out,))