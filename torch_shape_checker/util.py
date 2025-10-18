import torch

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


