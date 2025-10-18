
from typing import Callable, Any
import torch
from collections import defaultdict
import torch._C

class ShapeContext:
    def __init__(self,
                 scripted: torch.jit.ScriptModule | torch.jit.ScriptFunction,  # type: ignore
                 dispatch: dict[str, defaultdict[str, Callable[['ShapeContext', torch._C.Node], None]]],
                 args: list[Any] = [],
                 ) -> None:
        self.scripted:torch.jit.ScriptModule | torch.jit.ScriptFunction = scripted
        self.graph: torch._C.Graph = scripted.graph 
        assert len(list(self.graph.inputs())) == len(args)
        self.subContexts: dict[str, ShapeContext] = {}
        self.context: dict[str, Any] = {i.debugName():arg for i, arg in zip(self.graph.inputs(), args)}
        self.dispatch = dispatch
        
        nodes: list[torch._C.Node] = list(self.graph.nodes())
        for node in nodes:
            name_sapace, operation_name = node.kind().split("::")[:2]
            self.dispatch[name_sapace][operation_name](self, node)
            
    def get_result(self):
        if not self.graph.return_node().input().debugName() in self.context.keys():
            raise Exception("Result not yet calculated")
        return self.context[self.graph.return_node().input().debugName()]
    
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




    
    



