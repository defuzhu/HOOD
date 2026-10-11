# 2026-10-11 实验系统短程验证

使用 `/home/zhudf/miniconda3/envs/hood_clean/bin/python`，两张 RTX 3090 Ti，seed=57，真实 VTO/SMPL 数据。
没有启动新的 60000 batch 训练。Baseline 和 Modified 共享网络/数据/种子/预算，仅接触 criterion 配置不同。
完整操作与字段说明见 [EXPERIMENT_SYSTEM.md](EXPERIMENT_SYSTEM.md)。

## 已实际运行

根目录为 `/home/zhudf/data/hood_data/hood_data/experiments`。

| 记录 | 目录 | 结果 |
|---|---|---|
| Baseline 配对短测 | `smoke_system_baseline_seed57_20261011T064343Z_648c85e4` | 完成 4 batch、8 子步、7 次 Adam，保存 step 2/4 checkpoint；TensorBoard 403 tags。 |
| Modified 配对短测 | `smoke_system_modified_seed57_20261011T064346Z_f9d75c57` | 相同预算完成；TensorBoard 405 tags，额外记录 relaxed-barrier 参数。 |
| Modified 中途恢复 | `smoke_system_modified_resume_seed57_20261011T064438Z_65e8731c` | 从 step 2 恢复到 step 4，新增 2 batch/5 子步/5 次 Adam，累计 Adam=7。 |
| Baseline 独立验证 | `smoke_common_baseline_seed57_20261011T064559Z_a461d183` | 加载 step 4，三条相同动作各 3 帧，保存 CSV/JSON、3 个 NPZ、3 个 MP4。 |
| Modified 独立验证 | `smoke_common_modified_seed57_20261011T064601Z_c2951a31` | 同协议完成，对应 3 个 NPZ 和 MP4。 |

配对训练及恢复的代码 commit 为 `4ffa578`；独立验证使用 `604d643`（增加已验证的编码器回退）。
后续 `a7cd567` 修复微调恢复来源记录并加入公平比较测试，不改变本次物理损失公式。
记录的有效配置、commit、环境和数据资产哈希均在各目录内。
两组共同评价 signature 为 `170bef1364f45a9a9e65423fc2f7244caca0bcb19ccc4a06b7bd6c3aa7ea17b1`。

每组短测子步数为 1/2/2/3，Adam 更新数为 0/2/2/3：预热没有虚构更新或学习率。
逐子步核对 `collision_raw_loss × collision_weight = collision_penalty_loss`，均通过。
日志检查覆盖所有六项物理 loss、mean/last、实际学习率、裁剪前梯度、诊断与时间；训练和验证均有 TensorBoard 记录。
checkpoint 包含 schema 2 的模型/Normalizer、Adam、scheduler、RNG、sampler、布料/dataset 缓存、数据指纹、配置、step/epoch/commit/best 信息。

## 恢复的一致性

恢复与连续训练的后两步逐行核对：sample_index、sequence、garment、rollout 子步、timestep、材料参数、LR 和更新次数一致。
最终 scheduler、sampler、Torch CPU/CUDA RNG 状态一致。
物理项的最大相对差约 `1.44e-5`（接触项）；total 最大相对差约 `5.53e-6`。
模型权重最大绝对差约 `2.80e-5`；Normalizer 累加状态最大绝对差约 `5.76e-4`。
这些不是逐位一致结果，CUDA scatter 原有非确定性仍存在。
CPU checkpoint 测试包含随机数据、dropout、Adam/scheduler、Normalizer buffer、非空缓存与中途 sampler，恢复后逐位一致。

实际恢复测试曾发现缓存默认工厂丢失导致新衣服 KeyError，已修复为保留 defaultdict 并重测通过。
早期失败的实验目录保留失败状态，未覆盖；上表为完成的最终记录。

## 日志与统一验证

真实数据 CPU 双子步训练，开启/关闭 observation 的 loss 和完整模型 state_dict 完全一致。
重复固定 GPU 验证的物理/几何量在浮点容差内一致，训练参数与布料缓存未被改变。
独立验证关闭梯度，评价器覆盖为共同原始物理模型和单位权重 signed-mesh cubic，评价签名不依赖两组不同训练势能。
逐帧点—三角形距离/内外符号、异常伸长、速度/加速度、推理耗时均实际写出。

自动比较输出：

- `comparisons/smoke_system_seed57_resumed_updates_20261011`：按实际优化更新数对齐；拼接父日志到 step 2 后接恢复日志，136 张图、801 行逐项对比表。
- `comparisons/smoke_system_common_checkpoints_20261011`：独立验证目录作为输入，自动找回来源训练记录，136 张图、267 行逐项对比表。

曲线包括总量/各项 mean 与 last、LR、梯度、碰撞权重、训练耗时、统一验证和逐帧穿透。
两个配对 seed 的人工构造结果测试验证均值/样本标准差、基准为零的百分比处理、拒绝不同 seed/超参数/评价签名，以及恢复父日志截断。
多 seed 支持已检查，但本次没有声称跑过多 seed 的正式训练。

## 自动检查与限制

测试集覆盖接触区域（面/边/点/退化）、闭合网格符号、距离梯度与 CUDA 查询、延拓势能导数、面积权重、ramp、RNG 隔离、配置覆盖、实验不覆盖、CSV/TensorBoard 与恢复/对比。
最终发现测试共 29 项，27 项通过、2 项真实数据集成检查在发现测试中跳过；这两项另行显式执行也通过（29 个不同用例均已执行成功）。

这次只验证基础设施与短程数值链路；每条验证动作 3 帧，不足以判断长序列稳定性，更不能宣称 Modified 优于或不劣于原版。
之后须用相同预算 checkpoint 进行完整协议评价，并重复多个配对 seed，检查穿透、拉伸、运动、稳定性及效率。
短测并发执行，中位 batch 耗时约 Baseline 450 ms、Modified 720 ms，不能据此给正式速度提升/退化结论。

未改动初始碰撞修正器、接触图和摩擦，没有新增布料自碰撞、CCD 或严格可行性修正。
碰撞评价是衣服顶点到身体网格的有符号距离，不能认证整片衣服三角形无交叉。
当前 MP4 使用服务器可用的 MPEG-4 编码；H.264 原有动态库不兼容，独立编码器的网络下载失败，因此浏览器直接播放兼容性不作保证，离线播放器可查看。
历史 60000 batch Baseline 未覆盖；旧 checkpoint 可统一重新评价，缺 sampler/RNG/cache 时不能宣称精确续训。
