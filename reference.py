# pytorch_all2all.py
import os
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from config import MoEConfig

# ---------------- All2All pytorch impl ----------------
class PyTorchAllToAll:
    META_DIM = 5  # global_exp, src_rank, src_token, src_k, pad

    # 初始化一些分布式需要的变量
    def __init__(self, cfg: MoEConfig, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        # num experts per rank
        self.num_local_experts = cfg.num_experts // world_size
        # max recv tokens per rank，不满足的用 padding
        self.max_recv = cfg.max_num_tokens * world_size

    # ---------- dispatch ----------
    # dp_x 当前 rank（gpu）拥有的 token，shape = [num_tokens,hidden_dim]
    # indices: 每个 token 选中的全局专家ID列表，形状 [num_tokens, experts_per_token]，值域 [0, num_experts)。
    # TODO experts_per_token 就是 topk，能不能改成 topk？并且作为一个可以从外部传入的参数。
    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        device = dp_x.device
        cfg = self.cfg

        # 1) 构建扁平的发送 buffer 与 meta
        send_tokens = []
        send_meta = []
        for t, expert_list in enumerate(indices.tolist()):
            for k, e in enumerate(expert_list):
                send_tokens.append(dp_x[t].unsqueeze(0))
                send_meta.append([e, self.rank, t, k, 0])
        if send_tokens:
            send_buf_flat = torch.cat(send_tokens, dim=0)
            send_meta_flat = torch.tensor(send_meta, device=device, dtype=torch.int32)
        else:
            send_buf_flat = torch.empty((0, cfg.hidden_dim), device=device, dtype=cfg.in_dtype)
            send_meta_flat = torch.empty((0, self.META_DIM), device=device, dtype=torch.int32)

        # 2) 交换各 rank 发送条数，pad 到统一长度
        send_items = torch.tensor([send_meta_flat.size(0)], device=device, dtype=torch.long)
        all_items = [torch.zeros_like(send_items) for _ in range(self.world_size)]
        dist.all_gather(all_items, send_items)
        send_counts = [int(c.item()) for c in all_items]
        max_items = max(send_counts)
        pad_len = max_items - send_meta_flat.size(0)
        if pad_len > 0:
            pad_buf = torch.zeros(pad_len, cfg.hidden_dim, device=device, dtype=cfg.in_dtype)
            pad_meta = torch.zeros(pad_len, self.META_DIM, device=device, dtype=torch.int32)
            send_buf_flat = torch.cat([send_buf_flat, pad_buf], dim=0)
            send_meta_flat = torch.cat([send_meta_flat, pad_meta], dim=0)

        # 3) all_gather 数据与 meta（autograd-aware）
        gathered_buf = dist_nn.all_gather(send_buf_flat)
        gathered_meta = dist_nn.all_gather(send_meta_flat)
        concat_buf = torch.cat(gathered_buf, dim=0)
        concat_meta = torch.cat(gathered_meta, dim=0)

        # 4) 过滤目标为本 rank 的条目（根据全局专家 ID 映射 rank）
        global_eids = concat_meta[:, 0].to(torch.long)
        dst_ranks = global_eids // self.num_local_experts
        mask = dst_ranks == self.rank
        valid_buf = concat_buf[mask]
        valid_meta = concat_meta[mask]
        total_recv = valid_buf.size(0)

        # 5) 落桶到本地专家
        expert_num_tokens = torch.zeros(self.num_local_experts, dtype=torch.int32, device=device)
        expert_x = torch.empty(
            (self.num_local_experts, self.max_recv, cfg.hidden_dim),
            dtype=cfg.in_dtype,
            device=device,
        )
        expert_meta = torch.empty(
            (self.num_local_experts, self.max_recv, self.META_DIM),
            dtype=torch.int32,
            device=device,
        )
        for i in range(total_recv):
            geid = int(valid_meta[i, 0].item())
            local_eid = geid % self.num_local_experts
            pos = expert_num_tokens[local_eid]
            expert_x[local_eid, pos] = valid_buf[i]
            expert_meta[local_eid, pos] = valid_meta[i]
            expert_num_tokens[local_eid] += 1

        return expert_num_tokens, expert_x, expert_meta

    # ---------- combine ----------
    def combine(
        self,
        out_tokens: torch.Tensor,  # output, (max num tokens, hidden_dim)
        weights: torch.Tensor,  # topk weight
        expert_meta: torch.Tensor,  # input
        expert_y: torch.Tensor,  # input, (num_local_experts, max_num_tokens * num_dp, hidden_dim)
        expert_num_tokens: torch.Tensor,
    ):  # input
        device = out_tokens.device
        cfg = self.cfg

        # 单机单卡直接聚合，避免通信写入破坏计算图
        if self.world_size == 1:
            total_recv = int(expert_num_tokens.sum().item())
            if total_recv == 0:
                return out_tokens
            idx = []
            upd = []
            for local_eid in range(self.num_local_experts):
                cnt = int(expert_num_tokens[local_eid].item())
                for j in range(cnt):
                    meta = expert_meta[local_eid, j]
                    src_token = int(meta[2].item())
                    src_k = int(meta[3].item())
                    w = weights[src_token, src_k].to(torch.float32)
                    idx.append(src_token)
                    upd.append(expert_y[local_eid, j].to(torch.float32) * w)
            idx = torch.tensor(idx, device=device, dtype=torch.long)
            updates = torch.stack(upd, dim=0)
            out = torch.zeros_like(out_tokens, dtype=torch.float32)
            out = out.index_add(0, idx, updates)
            out = out + out_tokens.to(torch.float32)
            return out.to(out_tokens.dtype)

        # 构建扁平的发送 buffer 与 meta（目标 rank = meta[:,1]）
        send_tokens = []
        send_meta = []
        for local_eid in range(self.num_local_experts):
            cnt = int(expert_num_tokens[local_eid].item())
            if cnt == 0:
                continue
            send_tokens.append(expert_y[local_eid, :cnt])
            send_meta.append(expert_meta[local_eid, :cnt])
        if send_tokens:
            send_buf_flat = torch.cat(send_tokens, dim=0)
            send_meta_flat = torch.cat(send_meta, dim=0)
        else:
            send_buf_flat = torch.empty((0, cfg.hidden_dim), device=device, dtype=cfg.out_dtype)
            send_meta_flat = torch.empty((0, self.META_DIM), device=device, dtype=torch.int32)

        # 1) 交换条数，pad 到统一长度
        send_items = torch.tensor([send_meta_flat.size(0)], device=device, dtype=torch.long)
        all_items = [torch.zeros_like(send_items) for _ in range(self.world_size)]
        dist.all_gather(all_items, send_items)
        send_counts = [int(c.item()) for c in all_items]
        max_items = max(send_counts)
        pad_len = max_items - send_meta_flat.size(0)
        if pad_len > 0:
            pad_buf = torch.zeros(pad_len, cfg.hidden_dim, device=device, dtype=cfg.out_dtype)
            pad_meta = torch.zeros(pad_len, self.META_DIM, device=device, dtype=torch.int32)
            send_buf_flat = torch.cat([send_buf_flat, pad_buf], dim=0)
            send_meta_flat = torch.cat([send_meta_flat, pad_meta], dim=0)

        # 2) all_gather 数据和元信息
        gathered_buf = dist_nn.all_gather(send_buf_flat)
        gathered_meta = dist_nn.all_gather(send_meta_flat)
        concat_buf = torch.cat(gathered_buf, dim=0)
        concat_meta = torch.cat(gathered_meta, dim=0)

        # 3) 过滤目标为本 rank 的条目（meta[1] 是 src_rank，作为回传目的地）
        dst_mask = concat_meta[:, 1].to(torch.long) == self.rank
        if not torch.any(dst_mask):
            return out_tokens
        recv_buf = concat_buf[dst_mask]
        recv_meta = concat_meta[dst_mask]

        # 4) 聚合回源 token
        idx = recv_meta[:, 2].to(torch.long)      # src_token
        src_k = recv_meta[:, 3].to(torch.long)    # topk 序号
        weight = weights[idx, src_k].to(torch.float32)
        updates = recv_buf.to(torch.float32) * weight.unsqueeze(1)

        out = torch.zeros_like(out_tokens, dtype=torch.float32)
        out = out.index_add(0, idx, updates)
        out = out + out_tokens.to(torch.float32)
        return out.to(out_tokens.dtype)
