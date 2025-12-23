import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from layers.moe_layer import MoE
from config import MoEConfig
from config_defaults import get_default_cfg, get_default_train_cfg


def init_dist():
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def build_model(input_dim=1024, hidden_dim=4096, output_dim=1024, num_experts=16, top_k=2, expert_capacity=4, device="cuda"):
    moe = MoE(input_dim, num_experts, top_k, expert_capacity, hidden_dim, output_dim).to(device)
    # gate 用 DDP，同步参数；experts 保持本地
    moe.gate = DDP(moe.gate, device_ids=[device] if device != "cpu" else None, find_unused_parameters=False)
    return moe


def train_dp(cfg: MoEConfig, steps=1000, lr=5e-4, aux_alpha=1e-2, bsz=32, log_interval=100, profile: bool=False):
    rank = dist.get_rank()
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    model = build_model(
        input_dim=cfg.hidden_dim,
        hidden_dim=cfg.hidden_dim * 4,
        output_dim=cfg.hidden_dim,
        num_experts=cfg.num_experts,
        top_k=cfg.experts_per_token,
        device=device,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    # 固定目标投影
    target_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False).to(device)
    target_proj.requires_grad_(False)

    model.train()
    start_time = time.perf_counter()
    for step in range(steps):
        x = torch.randn(bsz, cfg.hidden_dim, device=device, dtype=cfg.in_dtype)
        with torch.no_grad():
            y_true = target_proj(x)

        y_pred, aux = model(x)
        task_loss = mse(y_pred, y_true)
        loss = task_loss + aux_alpha * aux

        opt.zero_grad(set_to_none=True)
        loss.backward()        # gate在DDP里封装，autograd-engine自动异步地管理分布式梯度
        # 阻塞式同步,计算和通信串行，此时GPU资源闲置
        world_size = dist.get_world_size()
        for p in model.experts.parameters():            # 只遍历专家的参数
            if p.grad is None:          # 单独处理没有梯度（本轮没有被激活）的专家
                p.grad = torch.zeros_like(p.data)               # 保证所有专家参数都有梯度张量（即使是零），避免后续分布式通信操作报错。
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)       # 用all reduce手动同步本地专家之间的梯度
            p.grad /= world_size            # 该param在各rank上的梯度加起来之后取均值
        opt.step()

        if rank == 0 and step % log_interval == 0:
            print(f"[dp] step {step:05d} | task {task_loss.item():.4f} | aux {aux.item():.4f} | total {loss.item():.4f}")
    
    if profile and rank == 0:
        elapsed = time.perf_counter() - start_time
        print(f"[dp] total {steps} steps time: {elapsed:.2f}s | {elapsed/steps*1000:.2f} ms/step")

def main():
    init_dist()
    cfg = get_default_cfg()
    train_cfg = get_default_train_cfg()
    train_dp(cfg, **train_cfg)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
