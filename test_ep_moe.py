import torch
import torch.distributed as dist

from config import MoEConfig
from reference import PyTorchAllToAll

cfg = MoEConfig(
    num_experts=16,
    experts_per_token=2,
    hidden_dim=256,
    max_num_tokens=128,
    in_dtype=torch.float16,
    out_dtype=torch.float16,
)

# need init_process_group to get rank and world_size

ata = PyTorchAllToAll(cfg, rank, world_size)