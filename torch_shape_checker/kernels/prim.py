from torch_shape_checker.ShapeContext import ShapeContext
from torch_shape_checker.kernels.util import register, kernel_type, not_implemented
import torch._C
import torch
from typing import Any, Callable
import inspect
from torch_shape_checker.util import int_to_dtype
from collections import defaultdict
KERNELS: dict[str, kernel_type] = defaultdict(lambda: not_implemented)

def register_prim(*op_names: str) -> Callable[[kernel_type], kernel_type]:
    return register([*op_names], KERNELS)

@register_prim("GetAttr")
def GetAttr(obj: ShapeContext, node: torch._C.Node):
    
    attr_name: str = node.s("name")
    attr = getattr(obj.scripted, attr_name)  # type: ignore
    
    debug_name:str = node.output().debugName()
    if isinstance(attr, torch.nn.Parameter):
        obj.context[debug_name] = attr.data
    else:
        obj.context[debug_name] = attr
        
        
@register_prim("Constant")
def Constant(obj: ShapeContext, node: torch._C.Node):   
    typ:str = node.output().type().annotation_str
    if "torch.nn.functional" in typ:
        v = getattr(torch.nn.functional, typ.split(".")[-1])
    else:
        v = node.output().toIValue()
    obj.context[node.output().debugName()] = v
    
@register_prim("CallMethod")
def CallMethod(obj: ShapeContext, node: torch._C.Node):
    submodule_node = list(node.inputs())[0]
    node_debug_name = submodule_node.debugName()
    submodule = obj.context[node_debug_name]
    
    inputs = [obj.context[i.debugName()] for i in node.inputs()]
    a = node.output().debugName()
    
    sub_sc: ShapeContext = ShapeContext(submodule, obj.dispatch, inputs)
    obj.context[a] = sub_sc.get_result()
    obj.subContexts[submodule_node.debugName()] = sub_sc
    
@register_prim("CallFunction")
def CallFunction(obj: ShapeContext, node: torch._C.Node):
    subfunction_node = list(node.inputs())[0]
    node_debug_name = subfunction_node.debugName()
    subfunc = obj.context[node_debug_name]
    
    inputs: list[Any] = []
    for i, t in zip(list(node.inputs())[1:], list(inspect.signature(subfunc).parameters.values())):
        v = obj.context[i.debugName()]
        if t.name == "dtype":
            inputs.append(int_to_dtype(v))
        else:
            inputs.append(v)
        
    a = node.output().debugName()
    #kwargs = {p:i for p,i in zip(list(inspect.signature(subfunc).parameters.keys()),inputs)}
    try:
        obj.context[a] = subfunc(*inputs)
    except Exception as e:
        print(f"Failed {subfunc} with args {inputs}")
        raise e
                    