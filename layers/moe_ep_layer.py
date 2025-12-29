import torch
import torch.distributed as dist
import torch.nn as nn

from config import MoEConfig
from reference import PyTorchAllToAll
from .expert import Expert


class EPMoE(nn.Module):
    """
    Expert-Parallel MoE layer (correctness-first).

    - gate: replicated across ranks (wrap with DDP outside if world_size > 1)
    - experts: sharded across ranks (each rank owns num_experts/world_size experts)
    - comm: dispatch/combine via PyTorchAllToAll
    """

    def __init__(self, cfg: MoEConfig, rank: int | None = None, world_size: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.rank = dist.get_rank() if rank is None else rank
        self.world_size = dist.get_world_size() if world_size is None else world_size

        if cfg.num_experts % self.world_size != 0:
            raise ValueError("num_experts must be divisible by world_size")

        self.ata = PyTorchAllToAll(cfg, rank=self.rank, world_size=self.world_size)
        self.gate = nn.Linear(cfg.hidden_dim, cfg.num_experts)

        self.num_local_experts = cfg.num_experts // self.world_size
        self.experts = nn.ModuleList(
            [Expert(cfg.hidden_dim, cfg.hidden_dim * 4, cfg.hidden_dim) for _ in range(self.num_local_experts)]
        )

    def _aux_loss(self, probs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        importance = probs.sum(0)  # [num_experts]
        importance_loss = torch.var(importance) / (cfg.num_experts**2)
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        routing_probs = probs * mask
        expert_usage = mask.float().mean(0)
        routing_weights = routing_probs.mean(0)
        load_balance_loss = cfg.num_experts * (expert_usage * routing_weights).sum()
        return importance_loss + load_balance_loss

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, cfg.experts_per_token, dim=-1)
        indices = indices.to(torch.int64)
        weights = weights.to(torch.float32)

        aux_loss = self._aux_loss(probs, indices) if self.training else torch.tensor(0.0, device=x.device)

        if self.world_size == 1:
            token_map = [[] for _ in range(self.ata.num_local_experts)]
            for t, expert_list in enumerate(indices.tolist()):
                for k, e in enumerate(expert_list):
                    local_eid = e % self.ata.num_local_experts
                    token_map[local_eid].append((t, k, e))

            expert_num = torch.tensor([len(lst) for lst in token_map], device=x.device, dtype=torch.int32)
            expert_meta = torch.zeros(
                (self.ata.num_local_experts, self.ata.max_recv, self.ata.META_DIM),
                device=x.device,
                dtype=torch.int32,
            )
            expert_inputs = []
            for local_eid, lst in enumerate(token_map):
                for pos, (t, k, e) in enumerate(lst):
                    expert_meta[local_eid, pos, 0] = e
                    expert_meta[local_eid, pos, 1] = self.rank
                    expert_meta[local_eid, pos, 2] = t
                    expert_meta[local_eid, pos, 3] = k
                idx = [t for t, _, _ in lst]
                expert_inputs.append(x[idx] if idx else None)
            expert_x = None
        else:
            expert_num, expert_x, expert_meta = self.ata.dispatch(x, indices)
            expert_inputs = None

        expert_y = torch.zeros(
            (self.ata.num_local_experts, self.ata.max_recv, cfg.hidden_dim),
            device=x.device,
            dtype=cfg.out_dtype,
        )
        for local_eid in range(self.ata.num_local_experts):
            cnt = int(expert_num[local_eid].item())
            if cnt == 0:
                continue
            if self.world_size == 1:
                x_slice = expert_inputs[local_eid].to(torch.float32)
            else:
                x_slice = expert_x[local_eid, :cnt].to(torch.float32)
            y_slice = self.experts[local_eid](x_slice).to(cfg.out_dtype)
            expert_y[local_eid, :cnt] = y_slice

        out_tokens = torch.zeros(cfg.max_num_tokens, cfg.hidden_dim, device=x.device, dtype=cfg.out_dtype)
        out_tokens = self.ata.combine(out_tokens, weights, expert_meta, expert_y, expert_num)
        out_tokens = out_tokens[: x.shape[0]]
        return out_tokens, aux_loss

