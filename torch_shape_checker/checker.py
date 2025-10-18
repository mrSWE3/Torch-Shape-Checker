
import torch
from torch import jit
from typing import Callable, Any
from torch_shape_checker.kernels.kernels import compile_ShapeContext, ShapeContext

def script_func(f: Callable[..., Any]) -> jit.ScriptFunction: # type: ignore
    return jit.script(f) # type: ignore

def script_module(f: torch.nn.Module ) -> jit.ScriptModule:
    return jit.script(f) # type: ignore

def check_shape_module(
                module_init: type[torch.nn.Module],
                init_args: list[Any] = [],
                init_kwargs: dict[str, Any] = {},
                forward_args: list[Any] = []) -> ShapeContext:
    with torch.device("meta"):
        m = module_init(*init_args,**init_kwargs)
        s = script_module(m)
        sc = compile_ShapeContext(s, [m, *forward_args])
    return sc

def check_shape_func(func: Callable[..., torch.Tensor],
                forward_args: list[Any]) -> ShapeContext:
    with torch.device("meta"):
        sc = compile_ShapeContext(script_func(func), forward_args)
    return sc