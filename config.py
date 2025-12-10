import dataclasses
import torch

# 只负责创建 cfg 抽象
# dataclass 装饰器的作用是 自动生成构造函数，没有这个装饰器就要自己写一大堆等于，非常丑
# TODO 未来可以通过 cfg.backend = "nccl" | "torch" | "triton" 去一键实现不同后端的替换

@dataclasses.dataclass
class MoEConfig:
    num_experts: int
    experts_per_token: int
    hidden_dim: int
    max_num_tokens: int
    in_dtype: torch.dtype = torch.float16
    out_dtype: torch.dtype = torch.float16

