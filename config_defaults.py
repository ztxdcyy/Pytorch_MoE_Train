import torch
from config import MoEConfig


def get_default_cfg() -> MoEConfig:
    return MoEConfig(
        num_experts=16,
        experts_per_token=2,
        hidden_dim=256,
        max_num_tokens=128,
        in_dtype=torch.float32,
        out_dtype=torch.float32,
    )


def get_default_train_cfg():
    return {
        "steps": 10000,
        "bsz": 32,
        "lr": 5e-4,
        "aux_alpha": 1e-2,
        "log_interval": 1000,  
        "profile": True,
    }
