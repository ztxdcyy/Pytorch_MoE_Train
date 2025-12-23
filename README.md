# Pytorch_MoE_Train

快速运行
---------
- 进入目录：`cd zomi/my_moe`
- 数据并行 (DP)：`MODE=dp NPROC=2 ./run_moe.sh`
- 专家并行 (EP，可导 all_gather+padding 版)：`MODE=ep NPROC=2 ./run_moe.sh`
- 调整端口/地址：设置 `MASTER_ADDR`、`MASTER_PORT` 环境变量。

项目结构与设计思路
-------------------
- `main_dp.py`：DP 入口。Gate 用 DDP 保持全局一致，experts 不包 DDP，梯度在 backward 后手动 all_reduce（未命中的专家补零再同步），避免 DDP unused 参数报错，也减少全量同步带宽。
- `main_ep.py`：EP 入口。基于 all_gather+padding 的可导实现，通信使用 autograd-aware 的 all_gather，避开 `all_to_all_single` 在当前后端缺失反传的问题。Gate 同样用 DDP，同步小模型参数。
- `reference.py`：EP 通信核心。dispatch/combine 改为「扁平化→pad→all_gather→过滤/落桶」，全链路可导；单卡分支直接聚合。
- `layers/`：MoE 组件（gate、experts 等）。配置默认值在 `config_defaults.py`，DP/EP 共用。
- `run_moe.sh`：统一入口脚本，通过 `MODE` 切换 DP/EP。

EP 实现与问题剖析
-----------------
- 问题：`all_to_all_single` 在 gloo/CPU 及部分 NCCL 组合下无 autograd 支持，recv 成为叶子张量，专家梯度丢失，loss 震荡。
- 方案：用 all_gather 兜底。步骤为：
  1) 扁平化需要发送的 tokens/meta。
  2) all_gather 交换各 rank 的条数，pad 到统一长度。
  3) `dist.nn.functional.all_gather` 收集数据+meta，concat 后按目标 rank 过滤，落桶（dispatch）或聚合（combine）。
  4) 非原地 `index_add` 聚合输出，保持计算图。
- 代价：通信量增大（每 rank 收全集），但在当前后端下保证梯度可传。

DP 实现要点
-----------
- Gate 很小且需一致：用 DDP，同步权重。
- Experts 本地实例：不包 DDP，backward 后手动 all_reduce 梯度（未命中的专家补零），减少带宽并避免 unused 参数错误。

配置
-----
- 默认 MoE/训练配置在 `config_defaults.py`，DP/EP 共用。需修改时直接调整该文件或在入口覆盖。

备注
-----
- 原 Python 测试脚本已移除，统一使用 `run_moe.sh` 调度。
- 如需性能优化，可在支持 autograd 的后端/版本上再尝试 `all_to_all_single` 或自定义 autograd.Function 包装 split 语义。
