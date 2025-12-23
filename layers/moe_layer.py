# 主要来自 draft.py，需要根据 ata kernel，构建一些 kernel 需要的输入输出张量 

import torch
import torch.nn as nn
from .expert import Expert

# MoE 核心模块
class MoE(nn.Module):
    def __init__(self, input_dim, num_experts, top_k, expert_capacity, hidden_dim, output_dim):
        super().__init__()
        self.num_experts = num_experts  
        self.top_k = top_k              
        self.expert_capacity = expert_capacity  
        
        self.gate = nn.Linear(input_dim, num_experts)  
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        
    def forward(self, x):
        batch_size, input_dim = x.shape
        device = x.device

        # 先给变量起名字 
        # dispatch_input_tokens = dp_x; dispatch_input_indices = indices
        
        # combine_output_tokens = out_tokens
        # topk_weight = weights
        # expert_meta
        # expert_y
        # expert_num_tokens
        
        # 1. 路由计算：完成“输入→专家匹配概率→Top-K 专家选择”
        logits = self.gate(x)  # [batch_size, num_experts]：门控输出各专家的原始匹配度（无范围约束）
        probs = torch.softmax(logits, dim=-1)  # 将 logits 归一化为 0-1 概率：确保路由权重可解释（概率越高越匹配）
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)  # 取 Top-K 专家：实现稀疏激活，降低计算量

        # 2. 负载均衡损失（仅训练时）：防止专家闲置，确保模型充分利用容量
        if self.training:
            importance = probs.sum(0)  # [num_experts]：每个专家的总路由概率（反映整体重要性）
            importance_loss = torch.var(importance) / (self.num_experts ** 2)  # 归一化方差：避免数值过大
            
            # 创建 Top-K 掩码：标记哪些专家被选中（用于计算usage从而计算load_balance_loss）
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(1, topk_indices, True)  # scatter_：按 topk_indices 将 mask 对应位置设为 True
            routing_probs = probs * mask  # [batch_size, num_experts]：仅保留选中专家的概率
            expert_usage = mask.float().mean(0)   # [num_experts]：专家使用率（分配样本占比）
            routing_weights = routing_probs.mean(0)  # [num_experts]：专家的平均路由权重（分配样本的依赖度）
            load_balance_loss = self.num_experts * (expert_usage * routing_weights).sum()  # 归一化损失
            aux_loss = importance_loss + load_balance_loss  # 总辅助损失：与主任务损失加权求和
        else:
            aux_loss = 0.0  # 推理时无需更新参数，关闭负载均衡损失

        # 3. 专家分配逻辑：将二维的专家选择结果展平，便于按专家分组计算
        flat_indices = topk_indices.view(-1)  # [batch_size*top_k]：将[batch_size, top_k]的专家索引展平为一维
        flat_probs = topk_probs.view(-1)      # [batch_size*top_k]：对应的专家权重也展平，与索引一一对应

        # 展平样本索引：每个样本对应 top_k 个专家，需标记每个专家索引属于哪个样本
        # tensor.expand(*sizes) 沿着一个size=1（必须）的某个维度做复制，仅修改tensor的stride等信息，让它在view的角度“看起来复制了”，但是底层并没有真正写入数据（repeat），所以做起来非常快。
        sample_indices = torch.arange(batch_size, device=device)[:, None]\
                            .expand(-1, self.top_k).flatten()  # [batch_size*top_k] 这里[:, None]等价于.unsqueeze(1) 假如bs=4，topk=2，则sampleindices初始化=[0,0,1,1,2,2,3,3]

        # 4. 专家并行计算：按专家分组处理样本，独立计算后聚合结果
        output_dim = self.experts[0].net[-1].out_features                   # 获取输出维度，out_features 是 nn.Linear() 的一个内置属性
        outputs = torch.zeros(batch_size, output_dim, device=device)        # 初始化输出张量
        
        for expert_idx in range(self.num_experts):
            # 找到分配给当前专家的样本：通过掩码筛选出属于该专家的样本索引
            expert_mask = flat_indices == expert_idx        # [batch_size*top_k] == 逐个元素比较，相等则mask对应位置上标记为 True
            expert_samples = sample_indices[expert_mask]    # 提取mask为true的expert对应的tokenlist
            expert_weights = flat_probs[expert_mask]        # 提取mask为true的expert对应的gating权重
            
            # 容量控制（丢弃超额样本）：避免单个专家处理过多样本导致计算过载或 OOM
            if len(expert_samples) > self.expert_capacity:
                expert_samples = expert_samples[:self.expert_capacity]  # 截断至最大容量
                expert_weights = expert_weights[:self.expert_capacity]
            
            if len(expert_samples) == 0:
                continue                # 无样本分配给当前专家，跳过计算
                
            # 专家计算并加权输出：按公式 y=sum(w_i*E_i(x))，先计算单个专家的加权输出
            expert_output = self.experts[expert_idx](x[expert_samples])             # [num_samples, output_dim]：专家处理样本 expert咱们上面定义了，expert(x)代表fwd
            weighted_output = expert_output * expert_weights.unsqueeze(-1)          # 权重广播到输出维度（匹配维度后相乘）

            # 聚合结果：将当前专家的加权输出累加到对应样本的位置（一个样本会累加 K 个专家的输出）
            outputs.index_add_(0, expert_samples, weighted_output)  # index_add_：按样本 ID 累加，避免循环赋值

        return outputs, aux_loss
    