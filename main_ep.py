import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from config import MoEConfig
from config_defaults import get_default_cfg, get_default_train_cfg
from reference import PyTorchAllToAll
from layers.expert import Expert


def init_distributed():
    """Init process group if not already done."""
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def build_modules(cfg: MoEConfig, hidden_dim: int):
    """Build gate (global) and local experts (per rank)."""
    gate = nn.Linear(hidden_dim, cfg.num_experts)       # gate 用 DDP 同步
    # num_local_experts is per-rank
    world_size = dist.get_world_size()
    num_local_experts = cfg.num_experts // world_size
    experts = nn.ModuleList(                            # 每个rank上有num_local_experts个本地专家
        [Expert(hidden_dim, hidden_dim * 4, hidden_dim) for _ in range(num_local_experts)]
    )
    return gate, experts


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
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = (
        torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device.index)

    ata = PyTorchAllToAll(cfg, rank, world_size)
    gate, experts = build_modules(cfg, cfg.hidden_dim)
    gate.to(device)
    gate = DDP(gate, device_ids=[device] if device.type == "cuda" else None)
    experts.to(device)

    opt = torch.optim.AdamW(list(gate.parameters()) + list(experts.parameters()), lr=lr)
    target_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False).to(device)          # 给定一个模拟训练任务：使得moe网络近似拟合该linear
    target_proj.requires_grad_(False)           # 用于生成label，并不需要训练
    mse = nn.MSELoss()
    writer = SummaryWriter(log_dir=os.environ.get("TB_LOGDIR")) if rank == 0 else None

    start_time = time.perf_counter()
    for step in range(steps):
        # 1) synthetic data
        x = torch.randn(bsz, cfg.hidden_dim, device=device, dtype=cfg.in_dtype)
        with torch.no_grad():
            y = target_proj(x.float()).to(cfg.out_dtype)

        # 2) gating -> top-k route
        logits = gate(x)
        probs = torch.softmax(logits, dim=-1)   # 温度平滑，避免过尖导致路由难训练
        weights, indices = torch.topk(probs, cfg.experts_per_token, dim=-1)
        indices = indices.to(torch.int64)  # scatter 需要 int64 索引
        weights = weights.to(torch.float32)

        # 2.1 aux loss（负载均衡）
        importance = probs.sum(0)  # [num_experts]
        importance_loss = torch.var(importance) / (cfg.num_experts ** 2)
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        routing_probs = probs * mask
        expert_usage = mask.float().mean(0)
        routing_weights = routing_probs.mean(0)
        load_balance_loss = cfg.num_experts * (expert_usage * routing_weights).sum()
        aux_loss = importance_loss + load_balance_loss

        # 3) dispatch
        if world_size == 1:
            # 绕过 all_to_all，纯本地分桶，保留对 x 的计算图
            token_map = [[] for _ in range(ata.num_local_experts)]
            for t, expert_list in enumerate(indices.tolist()):
                for k, e in enumerate(expert_list):
                    local_eid = e % ata.num_local_experts
                    token_map[local_eid].append((t, k, e))

            expert_num = torch.tensor(
                [len(lst) for lst in token_map], device=device, dtype=torch.int32
            )
            expert_meta = torch.zeros(
                (ata.num_local_experts, ata.max_recv, ata.META_DIM),
                device=device,
                dtype=torch.int32,
            )
            expert_inputs = []
            for local_eid, lst in enumerate(token_map):
                for pos, (t, k, e) in enumerate(lst):
                    expert_meta[local_eid, pos, 0] = e
                    expert_meta[local_eid, pos, 1] = rank
                    expert_meta[local_eid, pos, 2] = t
                    expert_meta[local_eid, pos, 3] = k
                idx = [t for t, _, _ in lst]
                expert_inputs.append(x[idx] if idx else None)
            expert_x = None  # 占位，单卡不依赖 expert_x
        else:
            expert_num, expert_x, expert_meta = ata.dispatch(x, indices)
            expert_inputs = None

        # 4) local expert forward
        expert_y = torch.zeros(
            (ata.num_local_experts, ata.max_recv, cfg.hidden_dim),
            device=device,
            dtype=cfg.out_dtype,
        )
        for local_eid in range(ata.num_local_experts):
            cnt = int(expert_num[local_eid].item())
            if cnt == 0:
                continue
            if world_size == 1:
                x_slice = expert_inputs[local_eid].to(torch.float32)
            else:
                x_slice = expert_x[local_eid, :cnt].to(torch.float32)
            y_slice = experts[local_eid](x_slice).to(cfg.out_dtype)
            expert_y[local_eid, :cnt] = y_slice

        # 5) combine back
        out_tokens = torch.zeros(cfg.max_num_tokens, cfg.hidden_dim, device=device, dtype=cfg.out_dtype, requires_grad=True)
        out_tokens = ata.combine(out_tokens, weights, expert_meta, expert_y, expert_num)
        out_tokens = out_tokens[: x.shape[0]]

        # 6) loss & step
        task_loss = mse(out_tokens.float(), y)
        total_loss = task_loss + aux_alpha * aux_loss
        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        opt.step()

        if rank == 0 and step % log_interval == 0:
            print(
                f"[step {step:05d}] task={task_loss.item():.4f} aux={aux_loss.item():.4f} total={total_loss.item():.4f}"
            )
        if writer:
            writer.add_scalar("loss/task", task_loss.item(), step)
            writer.add_scalar("loss/aux", aux_loss.item(), step)
            writer.add_scalar("loss/total", total_loss.item(), step)
    
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
    