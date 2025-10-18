
import torch
from torch import nn
from torch.nn import functional as F
import psutil
import os
import time
from torch_shape_checker.checker import check_shape_module


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


