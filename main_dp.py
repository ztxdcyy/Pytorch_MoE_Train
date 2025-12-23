import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn

from layers.moe_layer import MoE


def init_dist():
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def build_model(input_dim=1024, hidden_dim=4096, output_dim=1024, num_experts=16, top_k=2, expert_capacity=4, device="cuda"):
    moe = MoE(input_dim, num_experts, top_k, expert_capacity, hidden_dim, output_dim).to(device)
    ddp_model = DDP(moe, device_ids=[device] if device != "cpu" else None)
    return ddp_model


def train_dp(steps=1000, lr=5e-4, aux_alpha=1e-2, bsz=32, print_every=100):
    rank = dist.get_rank()
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    model = build_model(device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    # 固定目标投影
    target_proj = nn.Linear(1024, 1024, bias=False).to(device)
    target_proj.requires_grad_(False)

    model.train()
    for step in range(steps):
        x = torch.randn(bsz, 1024, device=device)
        with torch.no_grad():
            y_true = target_proj(x)

        y_pred, aux = model(x)
        task_loss = mse(y_pred, y_true)
        loss = task_loss + aux_alpha * aux

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if rank == 0 and step % print_every == 0:
            print(f"[dp] step {step:05d} | task {task_loss.item():.4f} | aux {aux.item():.4f} | total {loss.item():.4f}")


def main():
    init_dist()
    train_dp()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
