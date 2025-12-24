from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class KVCache:
    k: Tensor
    v: Tensor
    cur_len: int = 0

    def reset(self) -> None:
        self.cur_len = 0

