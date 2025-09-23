import torch
from torch import jit
import torch._C
from typing import Callable, Any
from shapes import ATEN_SHAPE_FUNCS, shape_type

def script_func[R](f: Callable[..., R]) -> jit.ScriptFunction:
    return jit.script(f) # type: ignore

def script_module(f: torch.nn.Module ) -> jit.ScriptModule:
    return jit.script(f) # type: ignore

class test_moudal(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(5, 10)
        self.y = torch.zeros((3,5))

    def forward(self):
        return self.l1(self.y)



def check_shape(scripted: torch.jit.ScriptModule | torch.jit.ScriptFunction,
                args: list[Any]) -> shape_type:
    graph: torch._C.Graph = scripted.graph
    assert len(list(graph.inputs())) == len(args)

    attr_context: dict[str, Any] = {i.debugName():arg for i, arg in zip(graph.inputs(), args)}
    nodes = list(graph.nodes())
    for node in nodes:
        
        kind: str = node.kind()
        print(node)
        if  kind.startswith("prim"):
            if kind.startswith("prim::GetAttr"):
                attr_name = node.s("name") 
                attr = getattr(scripted, attr_name)  # type: ignore
                if isinstance(attr, torch.Tensor):
                    attr_context[attr_name] = attr.shape
                else:
                    attr_context[attr_name] = attr
            if kind == "prim::CallMethod":
                submodule_node = list(node.inputs())[0]
                inputs = [attr_context[i.debugName()] for i in node.inputs()]
                submodule = getattr(scripted, submodule_node.debugName())
                attr_context[node.output().debugName()] = check_shape(submodule, inputs)
        elif kind.startswith("aten"):
            shapes = [attr_context[i.debugName()] for i in node.inputs() if isinstance(attr_context[i.debugName()], shape_type)]
            size = ATEN_SHAPE_FUNCS[kind](shapes)
            a = list(node.outputs())[0].debugName()
            attr_context[a] = size
    ret = attr_context[graph.return_node().input().debugName()]
    return ret

m = test_moudal()
scripted = script_module(m)
print(check_shape(scripted, [m]), m.forward().shape)