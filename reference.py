# pytorch_all2all.py
import os
import torch
import torch.distributed as dist
import dataclasses
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

        # ------ 1. 创建回传要用到的三个关键变量：send_counts（存数量），token_map（存tokendata），meta_map（存metadata）--------
        # send_counts[r]记录本 rank 需要给 rank r 发送多少 token，初始化为全 0 的 Python list（dim=worldsize）
        send_counts = [0] * self.world_size
        # 嵌套列表 token_map[r] 存储需要发送给 rank r 的所有tokenid
        token_map = [[] for _ in range(self.world_size)]
        # 嵌套列表 meta_map[r] 存储需要发送给 rank r 的所有token对应的metadata【combine 要用到，等combine看完再来复习下】
        meta_map = [[] for _ in range(self.world_size)]

        # ------- 2. 统计 send_counts_t & recv_counts_t -----------
        # 遍历 dispatch 输入 indices，填写 dst_rank 对应位置上的信息
        for t, expert_list in enumerate(indices.tolist()):      # 从indices中提取每个token选中专家的情况，t是第几个token，expert_list是每个token选中专家的list
            for k, e in enumerate(expert_list):             # 从expert_list中提取专家用于记录metadata：k代表list中的第几个专家，e代表具体的专家id
                dst_rank = e // self.num_local_experts      # 每个rank负责num_local_experts个专家，先确定去哪个rank
                send_counts[dst_rank] += 1                  # 计算完dst_rank之后，在三个关键变量中填充
                token_map[dst_rank].append(t)               
                meta_map[dst_rank].extend(                  
                    [e, self.rank, t, k, 0]
                )  # srcGobalExpert, srcRank, srcTokenIndex, expert index，padding
        send_counts_t = torch.tensor(send_counts, dtype=torch.long, device=device)      # 转成 tensor，torch.long = int64

        # recv_counts_t[i]：本 rank 从第 i 个 rank 处接收 recv_counts_t[i] 个 token。_t 代表 pytorch tensor
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        
        # 调用 torch.dist.alltoall API 将 send_counts_t “转置 transpose” 得到 recv_counts_t
        # alltoall示意图：https://user-images.githubusercontent.com/8791375/72598353-a73c0580-3935-11ea-84e3-03c61f4d4935.png
        """
        rank 0: [0, 5, 3, 2]  # 要发给rank 0的0个token，rank 1的5个token，rank 2的3个token，rank 3的2个token
        rank 1: [4, 0, 1, 3]  # 要发给rank 0的4个token，rank 1的0个token，rank 2的1个token，rank 3的3个token
        rank 2: [2, 2, 0, 4]  # 要发给rank 0的2个token，rank 1的2个token，rank 2的0个token，rank 3的4个token
        rank 3: [1, 3, 5, 0]  # 要发给rank 0的1个token，rank 1的3个token，rank 2的5个token，rank 3的0个token

        执行dist.all_to_all_single后，每个rank的recv_counts_t会变为：

        rank 0: [0, 4, 2, 1]  # 将从rank 0接收0个token，rank 1接收4个token，rank 2接收2个token，rank 3接收1个token
        rank 1: [5, 0, 2, 3]  # 将从rank 0接收5个token，rank 1接收0个token，rank 2接收2个token，rank 3接收3个token
        rank 2: [3, 1, 0, 5]  # 将从rank 0接收3个token，rank 1接收1个token，rank 2接收0个token，rank 3接收5个token
        rank 3: [2, 3, 4, 0]  # 将从rank 0接收2个token，rank 1接收3个token，rank 2接收4个token，rank 3接收0个token
        """
        dist.all_to_all_single(recv_counts_t, send_counts_t)          
        # 对于每个rank来说，send_counts_t 和 recv_counts_t都只是 world_size 维度的向量，代表本 rank 需要向其他 rank 发送/接收 token 的数量。  

        # ---------3. 组织 send_buf&recv_buf + send_meta&recv_meta ----------
        # 3.1 组织 send_buf，从 token_map 里找 tokenid，再从 dp_x 中提取对应token，then concat
        # token_map[r]存储需要发送给rank r的所有token索引
        # 而里面的每个子列表就是 idx_list，就是每个 rank 要发送的 tokenid
        # 根据 idx_list,找到 dp_x 中对应的 token，将这些 token 拼接在一起，得到一个大的 send buffer
        """
        假设有4个rank，当前rank的token分配如下：
        token_map = [
            [],           # 发往rank 0的token索引
            [5, 7, 9],    # 发往rank 1的token索引
            [2, 4],       # 发往rank 2的token索引
            [1, 3, 6, 8]  # 发往rank 3的token索引
        ]
        生成的send_buf将按照以下顺序排列：
        [token5, token7, token9, token2, token4, token1, token3, token6, token8]
        rank 1的token           | rank 2的token | rank 3的token
        """
        send_buf = torch.cat([dp_x[idx_list] for idx_list in token_map], dim=0)         # dp_x[idx_list].shape() = [len(idx_list), hidden_dim]
        
        # 计算当前 rank 需要接收的 token 的总数量，用于创建 recv_buf
        # tensor.item()：从标量张量提取 Python 原生数值（pytorch tensor的方法）；int() 保证是 int 类型（python 的方法）
        total_recv = int(recv_counts_t.sum().item())

        # 3.2 组织接收缓冲区 recv_buf，total_recv 个 hidden_dim 大小的 tensor。empty仅分配内存而不设置初始值
        recv_buf = torch.empty(
            total_recv, cfg.hidden_dim, dtype=cfg.in_dtype, device=device
        )

        # 3.3 组织 send_meta，从 meta_map 里面抽取
        # 双重嵌套列表推导式
        # meta_map有world_size个rank，每个rank有一个sub_meta_map的list。每个list里有num_tokens*expert_per_token个token的metadata（5dim）
        # srcGobalExpert, srcRank, srcTokenIndex, expert index, pad
        # 使用 view，reshape 成[num_tokens（当前 rank 持有多少个 token） × experts_per_token（每个 token 选中的专家数量，等价于 TopK）, 5]的 send_meta_data
        send_meta = torch.tensor(
            [v for sub in meta_map for v in sub], dtype=torch.int32, device=device
        ).view(-1, self.META_DIM)

        # 3.4 组织 recv_meta：预先创建一块大的连续的区域，后面用来写入“本rank接收到的，来自其他所有rank的token对应的metadata”
        recv_meta = torch.empty(
            total_recv, self.META_DIM, dtype=torch.int32, device=device
        )

        # ---------4. 调用 all_to_all_single通信，往 recv_buf & recv_meta 里写入。注意 uneven split 要额外引入 split_sizes --------------
        dist.all_to_all_single(
            recv_buf,
            send_buf,                                       # 因为每个专家收到的token数量不一致，而每个rank又持有相同数量的本地专家，因此每个rank收到的（相应的，发送的）token数量也不一致
            output_split_sizes=recv_counts_t.tolist(),      # output_split_sizes 记录了该 rank 需要从所有 rank 中接收多少 token
            input_split_sizes=send_counts_t.tolist(),       # input_split_sizes 记录了该 rank 需要向其余每个 rank 发送多少个 token
        )

        dist.all_to_all_single(
            recv_meta.view(-1),     # 展平成向量
            send_meta.view(-1),
            # recv_counts_t 原本是 tensor，先转 pythonlist，再用列表推导式提取每个 slot 上的数量 count（c）
            output_split_sizes=[c * self.META_DIM for c in recv_counts_t.tolist()],
            input_split_sizes=[c * self.META_DIM for c in send_counts_t.tolist()],
        )
        recv_meta = recv_meta.view(-1, self.META_DIM)


        # ---------5. 创建输出张量：expert_num_tokens（存数量），expert_x（存data），expert_meta（存metadata）------------
        # Per-rank 上有 num_local_experts 个专家，记录本 rank 每个专家“当前”收到的 token 个数（可以看成一种指针，区别于recv_counts_t）
        expert_num_tokens = torch.zeros(
            self.num_local_experts, dtype=torch.int32, device=device
        )

        # 为每个 expert 具体收到的 token tensor 预留一块大的 buffer，用 max_recv 设定上限，未用部分会被填充
        # shape = [num_local_experts, max_recv, hidden_dim]
        expert_x = torch.empty(
            (self.num_local_experts, self.max_recv, cfg.hidden_dim),
            dtype=cfg.in_dtype,
            device=device,
        )

        # 和 expert_x 对齐的元信息缓冲，shape 一致
        # META_DIM 的 5 个槽依次是（全局专家ID、源rank、源tokenID、top-k序号、pad位）
        expert_meta = torch.empty(
            (self.num_local_experts, self.max_recv, self.META_DIM),
            dtype=torch.int32,
            device = device,
        )

        # ---------6. 将 per-rank 接收到的 recv_buf&recv_meta 写到本地专家上------
        # 遍历本 rank 所有接收 token
        for i in range(total_recv):                             # total_recv：本 rank 从 other rank 接收到的所有 token 数量总和，i 代表一个 token
            # 6.1 计算 local_expert id (local_eid)，服务于计数器 expert_num_tokens
            global_eid = int(recv_meta[i, 0].item())            # 专家 全局id，从 recv_meta中提取，recv_meta.shape = [total_recv, META_DIM] 而 meata 元组的第一个分量就是专家的全局 id
            local_eid = global_eid % self.num_local_experts     # 专家 本地id
            
            # 6.2 token&metadata 的存储（根据计数器，控制写入位置）
            # expert_num_tokens[local_eid] 代表 本 rank 下，第 local_eid 个专家，当前记录了多少个 token 的信息。
            expert_x[local_eid, expert_num_tokens[local_eid]] = recv_buf[i]             # 写入一个 token（形状为[hidden_dim]的切片）expert_x.shape=[num_local_experts, max_recv, hidden_dim]
            expert_meta[local_eid, expert_num_tokens[local_eid]] = recv_meta[i]         # 写入一份metadata 
            expert_num_tokens[local_eid] += 1       # 这里加一代表成功写入一个 token 的 hidden_dim+metadata，起到指针偏移的作用，为写入下一个 token 做准备

        # 好像目前为止没有看到 pad？？？也就是 meta 的第五个分量，肯定是在combine里面
        # 总结下 return 的三个变量：
        # expert_num_tokens：每个 local_expert 已经处理的 token 数量；
        # expert_x：每个专家存储的 token 数据；expert_meta：与 expert_x 对齐的 metadata
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

        # ------ 1. 创建回传要用到的三个关键变量：send_counts（存数量），y_map（存tokendata），meta_map（存metadata）--------
        send_counts = [0] * self.world_size
        # y_map：长度为 world_size 的嵌套列表，用于按目标 rank 组织要发送的 token 数据
        y_map = [[] for _ in range(self.world_size)]
        # meta_map：和 y_map 一一对应的用于存放 metadata 的嵌套列表
        meta_map = [[] for _ in range(self.world_size)]

        # ------ 2. 统计 send_counts_t 和 recv_counts_t ------
        # 遍历本rank所有本地专家的所有接收token，将它们和 metadata[1] 中的 dst_rank 关联起来
        for local_eid in range(self.num_local_experts):
            cnt = int(expert_num_tokens[local_eid].item())      # 该本地专家接收到的token总数
            for j in range(cnt):                                # 遍历该本地专家收到的所有 token（j）
                meta = expert_meta[local_eid, j]                # 该本地专家的第j个token
                dst_rank = int(meta[1].item())                  # 提取 dst_rank
                send_counts[dst_rank] += 1                      # 计数加一

                # 回传 token j and its meta 到 dst rank/local eid
                # expert_y：专家计算后的输出张量，形状为 (num_local_experts, max_num_tokens * num_dp, token_dim)
                # unsqueeze(0) 是为了后面在 dim=0 上 concat
                y_map[dst_rank].append(expert_y[local_eid, j].unsqueeze(0))
                # pythonlist.extend() 将一个可迭代对象的所有元素 展开以后 分别添加到列表末尾 
                meta_map[dst_rank].extend(meta.tolist())

        # token nums that cur rank plan to send to other ranks
        send_counts_t = torch.tensor(send_counts, dtype=torch.long, device=device)
        # token nums that will recv from other ranks
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        
        # 依旧“转置”：由perrank发送情况，得到perrank接收情况
        dist.all_to_all_single(recv_counts_t, send_counts_t)

        # ---------3. 组织 send_buf&recv_buf + send_meta&recv_meta ----------
        y_map_tensors = []
        for sub_list in y_map:
            if sub_list:
                y_map_tensors.append(torch.cat(sub_list, dim=0))
            else:
                y_map_tensors.append(
                    torch.empty((0, cfg.hidden_dim), dtype=cfg.out_dtype, device=device)
                )
        send_buf = torch.cat(y_map_tensors, dim=0)

        # 根据 meta_map 整理得到 send_meta 这样一个 tensor
        send_meta = torch.tensor(
            [v for sub in meta_map for v in sub], dtype=torch.int32, device=device
        ).view(-1, self.META_DIM)

        total_recv = int(recv_counts_t.sum().item())
        
        recv_buf = torch.empty(
            total_recv, cfg.hidden_dim, dtype=cfg.out_dtype, device=device
        )
        recv_meta = torch.empty(
            total_recv, self.META_DIM, dtype=torch.int32, device=device
        )

        # ---------4. 调用 all_to_all_single 通信，往 recv_buf & recv_meta 里写入。--------------
        dist.all_to_all_single(
            recv_buf,
            send_buf,
            output_split_sizes=recv_counts_t.tolist(),
            input_split_sizes=send_counts_t.tolist(),
        )
        
        dist.all_to_all_single(
            recv_meta.view(-1),
            send_meta.view(-1),     # 注意 uneven split 要额外引入 split_sizes 
            output_split_sizes=[c * self.META_DIM for c in recv_counts_t.tolist()],
            input_split_sizes=[c * self.META_DIM for c in send_counts_t.tolist()],
        )
        
        recv_meta = recv_meta.view(-1, self.META_DIM)

        # ------ 5. 将接收结果按源 token 聚合（避免对叶子张量做就地写，便于反向传播）-----------
        if total_recv == 0:
            return out_tokens

        # idx: 源 token id；src_k: topk 序号；weight: gating 权重
        idx = recv_meta[:total_recv, 2].to(torch.long)
        src_k = recv_meta[:total_recv, 3].to(torch.long)
        weight = weights[idx, src_k].to(torch.float32)
        updates = recv_buf[:total_recv].to(torch.float32) * weight.unsqueeze(1)  # [N, hidden_dim]

        # 非就地累加，返回新张量，避免 leaf 原地修改导致梯度丢失
        out = torch.zeros_like(out_tokens, dtype=torch.float32)
        out = out.index_add(0, idx, updates)        # TODO Pytorch 向量化操作代替for循环
        out = out + out_tokens.to(torch.float32)  # 保留原 out_tokens 内容（通常为 0）
        return out.to(out_tokens.dtype)


# def generate_input(
#     num_experts, experts_per_token, hidden_dim, max_num_tokens, seed, rank, world_size
# ):
#     device = torch.device(f"cuda:{rank}")
#     gen = torch.Generator(device=device)
#     gen.manual_seed(seed + rank)

#     cfg = MoEConfig(
#         num_experts=num_experts,
#         experts_per_token=experts_per_token,
#         hidden_dim=hidden_dim,
#         max_num_tokens=max_num_tokens,
#         in_dtype=torch.float16,
#         out_dtype=torch.float16,
#     )
#     rank_data = RankTestData(cfg, gen, rank)
#     return cfg, rank_data, rank, world_size


# def ref_kernel(data: input_t) -> output_t:
#     cfg, rank_data, rank, world_size = data

#     ata = PyTorchAllToAll(cfg, rank, world_size)

#     expert_num, expert_x, expert_meta = ata.dispatch(rank_data.x, rank_data.indices)
#     expert_y = expert_x.to(cfg.out_dtype) * (1 + rank)
#     y = torch.zeros(
#         cfg.max_num_tokens,
#         cfg.hidden_dim,
#         dtype=cfg.out_dtype,
#         device=rank_data.x.device,
#     )

#     ata.combine(y, rank_data.weights, expert_meta, expert_y, expert_num)

#     return y[: rank_data.num_tokens]


# def check_implementation(data: input_t, output: output_t):
#     expected = ref_kernel(data)
#     if output.device != expected.device:
#         return False, f"Output device mismatch: {output.device} != {expected.device}"
#     res = torch.allclose(output, expected, rtol=1e-2, atol=5e-3)
#     if not res:
#         return False, f"Output values mismatch, {output} != {expected}"

#     return True, ""
