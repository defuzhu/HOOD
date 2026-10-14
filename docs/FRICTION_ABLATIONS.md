# 摩擦三组消融

三组是三种训练目标和三个独立配置，不是同一次训练的三个阶段，也不是三种推理开关。
三组都使用 signed_mesh + area_relaxed_barrier 法向接触，其他训练控制相同。
原始 HOOD baseline 配置不变；这里 A 组是新法向接触配旧摩擦。

| 组 | 配置（config= 后的值） | friction.mode |
|---|---|---|
| A | friction/legacy_seed57 | legacy |
| B | friction/geometry_mg_seed57 | geometry_mg |
| C | friction/lagged_coulomb_seed57 | lagged_coulomb |

A 完全沿用原摩擦实现。B 更换为有限三角形最近点、冻结重心坐标、统一
cloth.pos→pred_pos 和 body.pos→target_pos、单位法向投影，使用与法向接触一致的
2 mm 有符号距离支持域；保留原 mg 和固定轴方向因子以及未平滑位移范数。
因此 A→B 是几何/时间/投影/接触支持域的一组修正，不能单独归因某一小项。
旧代码的 KNN 平方距离阈值和面中心距离不被保留到 B。

C 与 B 使用同一接触几何和支持域，改成从法向势能导出的载荷和正则化库仑摩擦，
移除 mg 和固定轴方向因子。B→C 评估这一整组模型改变；若要分别隔离载荷和平滑
的贡献，需要追加消融。配置中的 friction_radius 只在 legacy 有效。

## 数值约定

- 当前状态的接触面、重心坐标、法向、载荷都冻结；仅预测位置参与摩擦反传。
- C 的载荷 λ=-dE_contact/dd，包含静止顶点面积和 collision_weight/weight_start，
  直接绑定实际 collision criterion，不复制一份可能失配的 YAML 刚度。
- λ 是每顶点 N；不使用 batch 平均后的损失求载荷，不重复乘面积/碰撞权重。
- 默认不限制载荷，不关闭负距离接触；穿透过深可能造成过大载荷，必须看日志。
- C 的 velocity_epsilon=0.01 m/s 是待验证的初值，不是已标定的最优值。
  平滑位移阈值 a=velocity_epsilon*timestep，初始 1/3 s 和常规 1/30 s 分别处理。
- 势采用 f0(r)-f0(0)，让零滑移的记录为零；冻结载荷时不改变力。
  friction_energy_loss 是增量伪势，不是实际耗散功，不同定义不能直接比较好坏。
- C 的梯度是正则化库仑切向力，仍有低速滑移近似；不保证精确静摩擦/无穿透。
- 接触于 n 时冻结：本步新形成接触会有摩擦滞后；没有加入固定点迭代或 CCD。
- 这里没有衣服自摩擦。

## 训练和对比

在服务器运行：

```bash
cd /home/zhudf/projects/HOOD
source /home/zhudf/miniconda3/etc/profile.d/conda.sh
conda activate hood_clean
export HOOD_PROJECT="$PWD"
export HOOD_DATA=/home/zhudf/data/hood_data/hood_data
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

正式三组分别启动（独立输出目录，每组预算 60000 步，seed=57）：

```bash
python train_baseline.py config=friction/legacy_seed57
python train_baseline.py config=friction/geometry_mg_seed57
python train_baseline.py config=friction/lagged_coulomb_seed57
```

多种子使用 `seed=58` 等覆盖，并保持三组的 seed 相同。不要只让某一组从旧 checkpoint
微调，另外两组从头训练。正式配置相同数据/网络/材料/优化器/预算，每组每次验证
都执行相同三条 tshirt、三条 longskirt（当前配置每条 120 帧；正式完整序列可共同设
`validation.n_steps=-1`）。所有训练写 CSV 和 TensorBoard。

短程功能检查模板（MODE 替换为 legacy / geometry_mg / lagged_coulomb）：

```bash
python train_baseline.py config=friction/MODE_seed57 experiment.name=smoke_friction_MODE experiment.max_iter=4 experiment.save_checkpoint_every=2 runner.postcvpr.warmup_steps=0 runner.postcvpr.increase_roll_every=2 runner.postcvpr.roll_max=3 validation.every=4 validation.n_steps=3 validation.save_videos=false
```

对同一 checkpoint 重新独立验证：

```bash
python validate_experiment.py checkpoint=/absolute/path/step_0000060000.pth
```

把三组的运行目录分别填为 A_DIR、B_DIR、C_DIR，使用已有脚本做 A→B 和 B→C：

```bash
python compare_experiments.py --baseline A_DIR --modified B_DIR --output /tmp/friction_A_vs_B
python compare_experiments.py --baseline B_DIR --modified C_DIR --output /tmp/friction_B_vs_C
```

输出目录必须是新目录。比较器会检查配置、预算、seed、评价器签名；不要用
`--exploratory` 绕过正式对照的限制。旧 60000 步新接触+旧摩擦模型方法上属于 A，
但它原来的三序列验证不等于新六序列评价；须重新统一评价，并核对训练控制，
不能仅凭名称把它当成新的严格三组实验之一。

## 怎样判断改善

统一评价器增加下面四个指标，在 CSV/TensorBoard/比较表中自动保存：

- common_contact_free_fraction：非固定衣服顶点中，当前有符号间距小于 2 mm 的比例。
- common_contact_slip_speed_mean_m_s：上述顶点相对人体同一材料点的平均切向速度。
- common_contact_slip_speed_rms_m_s：同一速度的 RMS。
- common_contact_slip_speed_max_m_s：同一速度的最大值。

没有接触顶点时速度记录零，必须同时看接触比例，避免把脱离人体误判为少滑动。
穿透点也在支持域内，必须结合已有穿透、拉伸/压缩、速度/加速度和视频评价。
上述滑移指标不使用各组训练摩擦公式、载荷或 mu，因此定义对三组一致。
比较器将其当作描述指标，不自动把“越小”当作“越好”：合理滑动不能被过度粘滞替代。
历史 common_postcvpr_mesh_v2 与新 mesh_slip_v3 签名不同，须重评两端后再比较。

训练 B/C 新增独立监控量（不进入 backward 的 loss_dict）：

- friction_active_fraction：当前接触支持域内顶点比例（训练包含固定点，区别于评价）。
- friction_load_sum_n / friction_load_max_n：各图载荷总和/峰值，跨 batch 等图平均；
  B 为 mg 尺度（尚未乘方向因子），C 为实际势能导数；不含 mu 或 friction.weight。
- friction_inside_fraction：当前在人体内部的顶点比例。
- friction_slip_speed_active_mean_m_s：训练接触顶点平均相对切向速度。
- friction_low_speed_active_fraction：接触顶点中速度低于 velocity_epsilon 的比例，
  仅说明进入平滑区，不代表满足静摩擦。

CSV 保存每个子步及 batch mean/last；TensorBoard 写 friction/*。A 不伪造不存在的
法向接触载荷日志。正式判断需多个相同 seed 对照、完整 rollout 与视频，不以摩擦
loss 或总 loss 下降作结论。

## 实现文件与验证顺序

- utils/friction_contact.py：载荷导数、冻结接触运动、平滑位移势。
- criterions/postcvpr/friction_contact.py：B/C 新实现。
- criterions/postcvpr/friction_energy.py：默认 legacy；新模式转给新实现。
- runners/postcvpr.py：把摩擦与实际 collision criterion 绑定。
- utils/training_observer.py、utils/tensorboard_logging.py：独立诊断日志。
- utils/fixed_validation.py：与训练模式无关的共同滑移评价及版本签名。
- configs/friction/：三份受控训练配置。
- tests/test_friction_contact.py：先检查导数、力方向/界限、零滑移/无接触、共同运动、
  重心跟踪、时间步、旋转不变性、batch 归一化、权重只计一次、旧公式不变。

先运行物理测试，再做三组短程前后向和六组验证，检查 CSV/TB/checkpoint 及断点恢复，
最后才做正式训练。改变 YAML 摩擦配置不会立即改变现有模型的裸推理输出；必须训练
相应模型，或另行实现物理修正层，才能改变运动。

## 本次执行结果（2026-10-14）

实现提交 `cbf9e13` 上完成三组真实 GPU 短程训练，各 4 batch、8 次 Adam 更新，
均在 step 0/4 执行六组各 3 帧固定验证。对应实验目录位于 `$HOOD_DATA/experiments/`：

- `smoke_friction_legacy_seed57_20261014T082319Z_d84aa3cd`
- `smoke_friction_geometry_mg_seed57_20261014T082354Z_ae9f32f4`
- `smoke_friction_lagged_coulomb_seed57_20261014T082438Z_af8dcb51`

验证三组初始共同滑移指标相同；每组 4 行 batch 日志、8 行子步日志、12 行验证日志，
末次六份 NPZ 均有限。原版 TensorBoard 759 个 scalar tags，B/C 各 771 个，包含
新增摩擦诊断和共同滑移指标。B/C 实际训练中的摩擦 loss 均出现非零值。
38 项自动测试中 34 项通过、4 项因其 opt-in/CUDA 条件跳过；新摩擦 6 项测试全部通过。
新增比较器测试确认滑移指标只输出描述性变化，不自动声称“减少就是改善”。

C 组从 step 2 恢复到 step 4 的目录：
`smoke_friction_lagged_coulomb_resume_seed57_20261014T082707Z_50e6df0c`。
恢复样本索引和总更新次数一致，载荷绑定、优化器、验证均正常。
CUDA 连续运行与恢复运行并非逐位一致：模型非 Normalizer 张量最大绝对差约
`1.24e-4`，Normalizer 累计量最大绝对差 `0.00390625`，摩擦子步 loss 最大差
`2.03e-4`。本次未证明严格数值等价；正式实验应记录恢复历史，不能宣称位级复现。

自动比较结果：
`$HOOD_DATA/experiments/comparisons/friction_smoke_20261014T162813/{A_vs_B,B_vs_C}/`。
包含训练物理项、状态/摩擦诊断、统一评价和逐帧穿透/接触滑移曲线。
这里只是短程功能验证，不是性能优劣证据。

C 组随机初始化短测中，单顶点载荷峰值约 8.28 N，摩擦增量势峰值约 7.16 J；
这来自有穿透的软接触状态，尚未标定。没有偷偷限幅或改变法向势能。
正式训练前/中应检查载荷分布、梯度裁剪、拉伸和粘滞情况；需要时将限幅作为
单独消融，而不是把它混入上述三组定义。

已核对旧完整 Modified 60000 步实验与 A 配置：模型、runner、seed、预算、原有
criterion 参数相同；数据根目录仅相对/绝对写法不同；资产及训练动作哈希相同。
因此可优先复用旧 Modified 为 seed57 的 A 组。仍应以新的统一六序列评价重评，
确保与 B/C 使用相同帧数和材料；旧官方 baseline 不能替代这里的 A。

日常项目中的 `HOOD_Universal_Inference.ipynb` 有用户未提交改动，本次保持原样。
`recording.require_clean=true` 会拒绝脏工作区正式训练；不必为此删除 notebook 改动。
可以在干净研究工作树运行上述命令，只把启动位置改为：

```bash
cd /home/zhudf/.codex/worktrees/1ce8/HOOD
export HOOD_PROJECT="$PWD"
```

如果采用断点恢复，请始终明确预算，例如：

```bash
python train_baseline.py resume=/absolute/path/checkpoint.pth experiment.max_iter=60000
```
