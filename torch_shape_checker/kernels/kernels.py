from collections import defaultdict
from torch_shape_checker.kernels.aten import compile_aten_kernal
from torch_shape_checker.kernels.prim import KERNELS as PRIM_KERNELS
from torch_shape_checker.kernels.util import kernel_type, not_implemented
from collections import defaultdict
from torch_shape_checker.ShapeContext import ShapeContext
import torch
from typing import Any

_cached_dispatch: dict[str, defaultdict[str, "kernel_type"]] | None = None

def compile_shape_kernel() -> dict[str, defaultdict[str, "kernel_type"]]:
    global _cached_dispatch
    if _cached_dispatch is not None:
        return _cached_dispatch

    dispatch: dict[str, defaultdict[str, "kernel_type"]] = defaultdict(
        lambda: defaultdict(lambda: not_implemented)
    )

    dispatch["aten"] = compile_aten_kernal()
    dispatch["prim"] = defaultdict(lambda: not_implemented, PRIM_KERNELS)


    _cached_dispatch = dispatch
    return dispatch

def compile_ShapeContext(
                 scripted: torch.jit.ScriptModule | torch.jit.ScriptFunction,  # type: ignore
                 args: list[Any] = [],)-> ShapeContext:
    return ShapeContext(scripted, compile_shape_kernel(), args)
    
