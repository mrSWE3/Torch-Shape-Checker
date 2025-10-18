from typing import Any
import torch
from torch_shape_checker.ShapeContext import ShapeContext
from collections import defaultdict
from torch_shape_checker.kernels.util import kernel_type
def compile_aten_kernal() -> defaultdict[str, kernel_type]:
    return defaultdict(lambda: eval_aten)

def eval_aten(context: ShapeContext, node: torch._C.Node): # type: ignore
    args: list[Any]= []
    for n in node.inputs():
        attr = context.context[n.debugName()]
        if isinstance(attr, tuple):
            args.append(attr[0])
        else:
            args.append(attr)
    aten_name = node.kind()[len("aten::"):]
    typ = node.output().type().annotation_str
    a = list(node.outputs())[0].debugName()
    
    
    aten_op = getattr(torch.ops.aten, aten_name)
    try:
        aten_op = getattr(aten_op, typ)
        kwargs = {a.name:v for a,v in zip(aten_op._schema.arguments, args)}
        context.context[a]  =  aten_op(**kwargs)
    except:
        context.context[a] = aten_op(*args)