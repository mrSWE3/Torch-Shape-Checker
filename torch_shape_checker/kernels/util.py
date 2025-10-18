from typing import Callable
import torch._C
from torch_shape_checker.ShapeContext import ShapeContext
 
kernel_type = Callable[[ShapeContext, torch._C.Node], None]
def register[T](op_names: list[str], dispatch: dict[str, T]) -> Callable[[T], T]:
    def decorator(fn: T) -> T:
        for name in op_names:
            dispatch[name] = fn
        return fn
    return decorator

def not_implemented(obj: "ShapeContext", node: torch._C.Node):
    raise NotImplementedError(f"{node.kind()} not implemented")
