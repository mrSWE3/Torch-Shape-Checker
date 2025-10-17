
import torch
from torch import jit
import torch._C
from typing import Callable, Any
from shapes import eval_aten
import time
import inspect


def script_func(f: Callable[..., Any]) -> jit.ScriptFunction: # type: ignore
    return jit.script(f) # type: ignore

def script_module(f: torch.nn.Module ) -> jit.ScriptModule:
    return jit.script(f) # type: ignore


def int_to_dtype(i: int):
    mapping = {
        0: torch.float32,
        1: torch.float64,
        2: torch.float16,
        3: torch.uint8,
        4: torch.int8,
        5: torch.int16,
        6: torch.int32,
        7: torch.int64,
        8: torch.bool,
        # Add more if needed
    }
    return mapping.get(i, None)



class ShapeContext:
    def __init__(self,
                 scripted: torch.jit.ScriptModule | torch.jit.ScriptFunction, # type: ignore
                 args: list[Any] = []) -> None:
        self.scripted:torch.jit.ScriptModule | torch.jit.ScriptFunction = scripted
        self.graph: torch._C.Graph = scripted.graph
        assert len(list(self.graph.inputs())) == len(args)
        self.subContexts: dict[str, ShapeContext] = {}
        self.context: dict[str, Any] = {i.debugName():arg for i, arg in zip(self.graph.inputs(), args)}
    
    def get_result(self):
        if not self.graph.return_node().input().debugName() in self.context.keys():
            raise Exception("Result not yet calculated")
        return self.context[self.graph.return_node().input().debugName()]
    
    
    
    def check_shape(self):
  
        nodes: list[torch._C.Node] = list(self.graph.nodes())
        
        for node in nodes:
            
            kind: str = node.kind()
            if  kind.startswith("prim"):
                if kind.startswith("prim::GetAttr"):
                    attr_name: str = node.s("name")
                    attr: ref_type = getattr(self.scripted, attr_name)  # type: ignore
                    
                    debug_name:str = node.output().debugName()
                    if isinstance(attr, torch.nn.Parameter):
                        self.context[debug_name] = attr.data
                    else:
                        self.context[debug_name] = attr
                if kind == "prim::Constant":
                    typ:str = node.output().type().annotation_str
                    if "torch.nn.functional" in typ:
                        v = getattr(torch.nn.functional, typ.split(".")[-1])
                    else:
                        v = node.output().toIValue()
                    self.context[node.output().debugName()] = v
                elif kind == "prim::CallMethod":
                    submodule_node = list(node.inputs())[0]
                    node_debug_name = submodule_node.debugName()
                    submodule = self.context[node_debug_name]
                    
                    inputs = [self.context[i.debugName()] for i in node.inputs()]
                    a = node.output().debugName()
                    
                    sub_sc: ShapeContext = ShapeContext(submodule, inputs)
                    sub_sc.check_shape()
                    self.context[a] = sub_sc.get_result()
                    self.subContexts[submodule_node.debugName()] = sub_sc
                    
                elif kind == "prim::CallFunction":
                    subfunction_node = list(node.inputs())[0]
                    node_debug_name = subfunction_node.debugName()
                    subfunc = self.context[node_debug_name]
                    
                    inputs: list[Any] = []
                    for i, t in zip(list(node.inputs())[1:], list(inspect.signature(subfunc).parameters.values())):
                        v = self.context[i.debugName()]
                        if t.name == "dtype":
                            inputs.append(int_to_dtype(v))
                        else:
                            inputs.append(v)
                        
                    a = node.output().debugName()
                    #kwargs = {p:i for p,i in zip(list(inspect.signature(subfunc).parameters.keys()),inputs)}
                    try:
                        self.context[a] = subfunc(*inputs)
                    except Exception as e:
                        print(f"Failed {subfunc} with args {inputs}")
                        raise e
                    
            elif kind.startswith("aten"):
                args: list[Any]= []
                for n in node.inputs():
                    attr = self.context[n.debugName()]
                    if isinstance(attr, tuple):
                        args.append(attr[0])
                    else:
                        args.append(attr)
                
                ret = eval_aten(kind[len("aten::"):],node.output().type().annotation_str, *tuple(args))
                a = list(node.outputs())[0].debugName()
                self.context[a] = ret
            
    def __str__(self):
        return "\n".join(self.str_depth())
    
    def __repr__(self):
        return self.__str__()
    
    def str_depth(self,max_lvl:int = -1) -> str:
        return "\n".join(self._str_helper(max_lvl))
    
    def _str_helper(self, max_lvl:int) -> list[str]:
        out: list[str] = []
        for name,ref in self.context.items():
            if not name.isnumeric():
                if (isinstance(ref, (int, float, bool, str, bytes, type(None), torch.Tensor))):
                    out.append(f"[Ref]{name}: {ref.shape if isinstance(ref, torch.Tensor) else ref}")
            
        for name, context in self.subContexts.items():
            out.append(f"[Context]{name}({type(self.context[name]).__name__}):")
            if max_lvl != 0:
                out.extend(["  " + s for s in context._str_helper(max_lvl-1)])
                
        result = self.get_result()
        out.append(f"[Result]: {result.shape if isinstance(result, torch.Tensor) else result}")
        return out
  



    

def check_shape_module(
                module_init: type[torch.nn.Module],
                init_args: list[Any] = [],
                init_kwargs: dict[str, Any] = {},
                forward_args: list[Any] = []) -> ShapeContext:
    with torch.device("meta"):
        m = module_init(*init_args,**init_kwargs)
        s = script_module(m)
        sc = ShapeContext(s, [m, *forward_args])
        sc.check_shape()
    return sc

def check_shape_func(func: Callable[..., torch.Tensor],
                forward_args: list[Any]) -> ShapeContext:
    with torch.device("meta"):
        sc = ShapeContext(script_func(func), forward_args)
        sc.check_shape()
    return sc

from torch import nn
from torch.nn import functional as F
class TestModel(torch.nn.Module):
    def __init__(self, fi: int, fo: int) -> None:
        super().__init__() # type: ignore
        # Multiple sequential layers
        self.l1 = nn.Sequential(
            nn.Linear(fi, fo),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(fo, fo),
            nn.ReLU()
        )
        self.scale = 0.5  # a constant scalar
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply multiple layers and operations
        out1 = self.l1(x)
        out2 = self.l2(out1)
        added = out1 + out2 * self.scale  # scalar multiplication
        # softmax with optional dtype argument
        return F.softmax(added, dim=1, dtype=torch.float32)

from torch import jit

from typing import Callable, Any

import psutil
import os

def get_mem_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024



in_size = torch.Size((500,100))
init_args = [100,1000000000]
# Time and memory shape checking
start = time.time()
mem_start = get_mem_mb()
with torch.device("meta"):
    shape_result = check_shape_module(TestModel,init_args=init_args, forward_args=[torch.rand(in_size)])
mem_end = get_mem_mb()
end = time.time()

print("Shape checker result")
print(shape_result.str_depth())
print("Shape checker time:", end - start)
print("Shape checker memory (MB):", mem_end - mem_start)


