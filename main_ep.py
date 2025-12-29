import os
import time
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from config import MoEConfig
from config_defaults import get_default_cfg, get_default_train_cfg
from layers.moe_ep_layer import EPMoE


def init_distributed():
    """Init process group if not already done."""
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def train_tiny_ep(
    cfg: MoEConfig,
    steps: int = 10,
    bsz: int = 16,
    lr: float = 5e-4,
    log_interval: int = 1000,
    aux_alpha: float = 1e-2,
    profile: bool = False,
):
    """Minimal EP-only training loop to sanity check all-to-all wiring."""
    # 分布式相关初始化
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # cuda: 0, 1
    device = (
        torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device.index)     # 绑定 rank 到cuda device

    model = EPMoE(cfg, rank=rank, world_size=world_size).to(device)
    if world_size > 1:
        model.gate = DDP(model.gate, device_ids=[device] if device.type == "cuda" else None)

    # 训练相关（优化器和模拟的任务——训练moe拟合一个nn.Linear()）初始化
    opt = torch.optim.AdamW(list(model.gate.parameters()) + list(model.experts.parameters()), lr=lr)
    target_proj = torch.nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False).to(device)  # 给定一个模拟训练任务：使得moe网络近似拟合该linear
    target_proj.requires_grad_(False)                                                       # 用于生成label，并不需要训练
    mse = torch.nn.MSELoss()
    writer = SummaryWriter(log_dir=os.environ.get("TB_LOGDIR")) if rank == 0 else None

    start_time = time.perf_counter()
    last_log_time = start_time
    for step in range(steps):
        # 1) synthetic data
        x = torch.randn(bsz, cfg.hidden_dim, device=device, dtype=cfg.in_dtype)
        with torch.no_grad():
            y = target_proj(x.float()).to(cfg.out_dtype)

        out_tokens, aux_loss = model(x)

        # 6) loss & step
        task_loss = mse(out_tokens.float(), y)
        total_loss = task_loss + aux_alpha * aux_loss
        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        opt.step()

        interval_s = None
        if rank == 0 and step % log_interval == 0:
            now = time.perf_counter()
            interval_s = 0.0 if step == 0 else (now - last_log_time)
            last_log_time = now
            print(
                f"[step {step:05d}] task={task_loss.item():.4f} aux={aux_loss.item():.4f} "
                f"total={total_loss.item():.4f} interval_s={interval_s:.2f}"
            )
        if writer:
            writer.add_scalar("loss/task", task_loss.item(), step)
            writer.add_scalar("loss/aux", aux_loss.item(), step)
            writer.add_scalar("loss/total", total_loss.item(), step)
            if interval_s is not None:
                writer.add_scalar("time/interval_s", interval_s, step)
    
    if profile and rank == 0:
        elapsed = time.perf_counter() - start_time
        print(f"[ep] total {steps} steps time: {elapsed:.2f}s | {elapsed/steps*1000:.2f} ms/step")

    dist.barrier()
    if writer:
        writer.flush()
        writer.close()


def main():
    init_distributed()
    cfg = get_default_cfg()
    train_cfg = get_default_train_cfg()
    train_tiny_ep(cfg, **train_cfg)
    if dist.get_rank() == 0:
        print("EP MoE tiny training finished.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
    
