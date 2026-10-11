# HOOD 对照实验系统：使用与指标说明

本入口用于后续物理损失与接触算法研究。代码版本交给 Git，算法开关交给 YAML，一次实际运行使用独立目录。
2026-10-11 已完成配对 GPU 短测、恢复测试、统一验证、视频和自动对比；没有启动新的 60000 次训练。
短测报告见 [EXPERIMENT_SYSTEM_VERIFICATION.md](EXPERIMENT_SYSTEM_VERIFICATION.md)。

## 1. 检查结果与实现范围

| 检查文件 | 发现与处理 |
|---|---|
| `runners/postcvpr.py` 的 `forward/criterion_pass/optimizer_step/run_epoch` | 原版只返回最后一个 rollout 子步。保留该接口，用观察回调记录所有子步；保持每子步 detach、backward、Adam、scheduler 的顺序。非有限损失/梯度会报错。 |
| `train_baseline.py`、`utils/experiment.py`、`utils/tensorboard_logging.py` | 已有 CSV、mean/last 与 TensorBoard，直接扩展，未开发第二套日志；增加实际更新计数、计时、碰撞原始量及诊断。 |
| `criterions/postcvpr/*`、`utils/contact_geometry.py`、`utils/contact_potential.py` | 保留原始六项损失；仅接触项可切换几何和势能。诊断保存在 `last_diagnostics`，不加入参与求和反传的返回字典。 |
| `utils/common.py::save_checkpoint`、`train.py` | 原逻辑保存模型、配置及辅助模块，但不足以恢复中途采样与随机状态。保留旧入口/保存函数；新入口外包原子保存和完整实验状态。 |
| `ValidationSequences.ipynb`、`utils/metrics.py`、`utils/make_comparison.py`、`configs/aux/metrics.yaml`、`criterions/aux/collision_metrics.py` | 复用其逐帧 rollout 和物理评价思想、现有 runner/SMPL/布料构造；旧碰撞指标是面中心平面近似，不能作为真实穿透深度。原论文比较依赖下载的特定比较数据/静止形状，继续保留。 |
| `utils/fixed_validation.py`、`validate_experiment.py` | 独立统一评价器：原始物理模型、统一单位权重的几何碰撞诊断、固定序列/材料/初始条件。保存序列及逐帧结果。 |
| `compare_experiments.py` | 检查配对 seed、训练控制变量、数据哈希、初始化、预算与评价签名，拼接恢复历史，生成单项曲线和配对统计。 |

Baseline 在研究分支中选择原始物理公式及原始碰撞修正，实现共用的记录和评价基础设施。
官方 `main=9bc1076` 保留，研究从既有工作基线 `c929c50` 发展，兼容/记录修改不冒充官方文件逐字未改。
本次未 reset/clean，开发通过独立分支进行，未覆盖之前提交。

本次所谓“新碰撞”具体指 **训练接触项的距离几何**。
初始碰撞修正器、GNN body-garment 接触边、摩擦模型仍使用原实现；没有加入布料自碰撞或 CCD。
Modified 是有限延拓的 relaxed barrier 势能，能够接收穿透预测；不是严格不可穿透的 interior-point barrier。

## 2. 环境与正式训练命令

在服务器执行：

```bash
cd /home/zhudf/projects/HOOD
source /home/zhudf/miniconda3/etc/profile.d/conda.sh
conda activate hood_clean
export HOOD_PROJECT="$PWD"
export HOOD_DATA=/home/zhudf/data/hood_data/hood_data
```

已在 `hood_clean` 实测；`hood` 尚未验证本次新增依赖。TensorBoard 为 2.20.0。
正式实验要求干净 Git checkout，修改后先提交。每次创建时间戳和唯一 ID 目录，名称相同也不会覆盖。
`taskset -c 0` 是本服务器沿用的 CPU 绑定，不是 HOOD 的物理要求；其他服务器可移除。

以下 **60000 batch 正式训练命令本次未执行**。使用前确认 GPU 空闲：

```bash
# Baseline：原始距离近似 + 原始三次接触势能
CUDA_VISIBLE_DEVICES=0 taskset -c 0 python -u train_baseline.py \
  config=baseline/postcvpr_seed57 experiment.name=baseline seed=57

# Modified：signed mesh 距离 + 面积加权 relaxed barrier
CUDA_VISIBLE_DEVICES=0 taskset -c 0 python -u train_baseline.py \
  config=contact/relaxed_barrier_v2_seed57 experiment.name=modified seed=57
```

两组默认 batch_size=1、num_workers=0、Adam 配置 lr=5e-5、相同随机材料范围、相同训练 split/网络、max_iter=60000。
`max_iter` 数的是 batch/global_step，不是 Adam 次数。预热与多子步 rollout 导致两者不同。
原始 scheduler 保留，实际初始 LR 因其乘数含 `+0.01` 是 5.05e-5，不能把配置 LR 当成每步实际 LR。
默认第 100 个 batch 起更新；每 5000 个 batch 增加 rollout 长度，上限 5。
默认每 2000 batch 保存；每 5000 batch 和末步验证。
日常验证最多 120 帧，正式末次完整验证使用下文独立脚本。

消融配置：

| config | 距离几何 | 接触势能 |
|---|---|---|
| `baseline/postcvpr_seed57` | 原始面中心平面 | 原始三次罚能 |
| `contact/geometry_v1_seed57` | signed mesh | 原始三次罚能 |
| `contact/potential_only_v2_seed57` | 原始面中心平面 | 面积加权 relaxed barrier |
| `contact/relaxed_barrier_v2_seed57` | signed mesh | 面积加权 relaxed barrier |
| `contact/area_cubic_v2_seed57` | signed mesh | 面积加权三次罚能，用于区分面积权重和势能形状的影响 |

每组修改 `config=` 和独立的 `experiment.name=` 即可。
后续 seed=58/59 也可直接覆盖；文件名 seed57 是模板名，最终生效的 seed 写入配置及元数据。
正式配对实验必须两边使用相同 seed 集合、训练预算、数据和评价协议。

仅快速检查流程：

```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0 python train_baseline.py config=smoke/baseline_seed57
CUDA_VISIBLE_DEVICES=0 taskset -c 0 python train_baseline.py config=smoke/modified_seed57
```

这两个配置各运行 4 batch：预热阈值 2、rollout 增长间隔 2、上限 3、LR 衰减间隔 2，仅用于验证各代码路径，不能与正式 60000 batch 结果比较。
完整的实际配置总会保存为 `config.yaml`，包含模块默认值和 CLI 覆盖。

## 3. 每个训练日志字段

`logs/train_substeps.csv` 逐子步，`logs/train_rollout_steps.csv` 是它的符号链接，避免存两份不同步的数据。
`logs/train.csv` 每 batch 一行；物理量与接触诊断的 `_mean` 为本 batch 所有子步平均，`_last` 为最后子步。
原始 `Runner.forward()` 返回值仍是最后子步，完整日志通过独立 observer 获取。

| 字段或字段族 | 定义 |
|---|---|
| `global_step` | 已处理的 batch 序号，从 1 起。验证的 step=0 表示未训练模型。 |
| `epoch` | 从 0 起的数据遍历编号。 |
| `sample_index` | 本 batch 在训练 dataset 中的索引，用于核对恢复采样。 |
| `sequence/garment` | 当前动作序列与衣服标识，位于子步 CSV。 |
| `rollout_substep` | batch 内子步索引，从 0 起。 |
| `roll_steps` | 本 batch 实际执行的 rollout 长度。 |
| `optimized` | 此子步是否执行 Adam，0 表示预热。batch 行为末子步标志。 |
| `optimizer_step_count` | 截止此行真实完成的 Adam 更新累计数，恢复后连续计数。 |
| `optimizer_updates_in_batch` | batch CSV 是该 batch 总更新数；子步 CSV 是截至该子步的累计更新数。 |
| `learning_rate/lr_used` | 该更新实际使用的 LR；预热为空。batch 的 `learning_rate` 是最后子步 LR。 |
| `lr_next` | 子步更新和 scheduler 执行后，下一次更新将使用的 LR；预热仍可记录。 |
| `grad_norm_preclip/grad_norm_before_clip` | backward 后、梯度裁剪前的全网络梯度 L2 范数；两者同义。预热为空。 |
| `step_time_ms` | batch 数据加载、搬运、所有前向/损失/反向/更新和诊断的同步耗时；不含随后验证、checkpoint 和最后 CSV/TB 写出。 |
| `rollout_substep_time_ms` | 子步之间的监控耗时；第一个还包含加载与准备，不是纯模型前向耗时。 |
| `elapsed_s` | 整个实验经过时间，包含验证和保存；恢复时累计父实验的时间。 |
| `timestep` | 此子步积分时间间隔，初始化默认 1/3 s，常规默认 1/30 s。 |
| `lame_mu/lame_lambda/bending_coeff` | 本 batch 抽样材料的 Lamé 参数和弯曲系数。抽样范围/密度范围记录在完整配置。 |
| `stretching_energy_loss` | 原始膜内弹性能，沿用对应 criterion 的标量值。 |
| `bending_energy_loss` | 原始离散弯曲能，沿用对应 criterion。 |
| `inertia_loss` | 相对惯性预测的二次代价，沿用原始质量和时间步处理。 |
| `gravitational_energy_loss` | 重力势能；可以为负，降低不自动代表更合理。 |
| `collision_penalty_loss` | 当前实验定义下、乘当前碰撞权重后的接触项，实际进入训练目标。 |
| `friction_energy_loss` | 原始摩擦能项；本次未更换其计算方法。 |
| 其他 `<criterion>_loss` | 新 criterion 返回的物理标量会自动出现在子步、mean/last 和 TensorBoard；诊断不要从 criterion 返回。 |
| `total_loss/total_physical_loss` | 上述参与训练的物理项之和，两者同义；不是数据拟合误差。batch 保存 `_mean` 和 `_last`。 |
| `collision_raw_loss` | 当前接触定义中、乘动态碰撞权重之前的标量，保证 raw×weight=collision_penalty_loss。 |
| `collision_weight` | 原始 ramp 的当前权重，依据 outer global_step，同一 batch 各子步相同。 |
| `contact_training_active_fraction` | 训练距离 d<eps 的顶点比例，包含近接触和穿透。 |
| `contact_training_inside_fraction` | 训练距离 d<0 的比例；原始平面模式只是近似，不能冒充真实几何穿透率。 |
| `contact_training_distance_min_m` | 训练距离的最小值，单位 m；符号/几何取决于实验模式。 |
| `contact_extended_fraction` | relaxed barrier 中 d<eps×tau、使用二次延拓的顶点比例；其他势能为 0。 |
| `barrier_stiffness_current` | κ×collision_weight/weight_start，仅 relaxed barrier 模式记录。 |

特别注意 raw 的量纲/定义随势能变化：原始 cubic 为 Σmax(eps−d,0)^3；面积 cubic 加入 A_i/A_ref。
relaxed barrier 的 raw 是 `P/weight_start`，P 为面积加权势能，因此加权项为 `P×weight/weight_start`。
它不是把 P 再乘一次 5000；不同模式的 raw、contact loss 和 total loss 不能当作共同质量指标直接比较。

全部监控数值从 detach 后取出；loss 字典仅包含实际参与训练的 criterion 返回量。
每个 rollout 子步单独反传和更新，上一子步预测作为下一子步初态时已 detach，不做跨整段 rollout 的 BPTT。

## 4. Checkpoint 与中断恢复

新 checkpoint 为 `checkpoint_schema=2`，通过 `.pth.tmp` 写完整后原子重命名。
包含模型权重/Normalizer buffer、Adam、scheduler、global_step、真实 Adam 次数、epoch、seed、完整配置、commit、最佳验证、累计时间、数据哈希，以及：

- Python、NumPy、Torch CPU 和所有可见 CUDA 的随机状态；
- 当前随机排列、sampler cursor，只有完整完成一个 batch 后才提交 cursor；
- 衣服的静止几何/质量/连接等运行缓存和 dataset 动态 coarse-edge 缓存，保留 defaultdict 的自动建缓存行为。

这些缓存会影响下一次损失，恢复时不能只恢复网络。
新入口暂要求 batch_size=1、num_workers=0，避免 worker 预取导致未记录的采样/RNG 状态。
CUDA scatter 不保证逐位确定性：恢复采样和随机状态不等于承诺 GPU 浮点结果完全相同。

```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0 python train_baseline.py \
  resume=/absolute/path/experiment/checkpoints/step_0000020000.pth \
  experiment.name=modified_resume
```

配置自动取自 checkpoint，不需要重新指定 `config=`。默认要求代码 commit、数据内容和训练设置一致。
可以变更名称、设备、训练预算；若 checkpoint 已达到原预算，添加例如 `experiment.max_iter=80000`。
变更预算后做正式比较，两边必须匹配。
恢复创建新目录，在 metadata 中记录 parent_run/from_step/checkpoint hash；对比脚本截断父日志到恢复点后拼接，避免重复步骤。

代码有新提交时，优先在旧 commit 的独立工作树恢复，勿为恢复而 reset 当前研究分支：

```bash
git worktree add --detach /tmp/hood_resume_worktree CHECKPOINT_GIT_COMMIT
cd /tmp/hood_resume_worktree
export HOOD_PROJECT="$PWD"
# 使用同一 Conda 环境、HOOD_DATA 和上面的 resume 命令
```

只有明确接受代码不同才用 `resume_allow_code_change=true`，元数据会标记，不能称为同版本精确恢复。
旧 checkpoint 缺 RNG/sampler/cache 时会拒绝 resume；仍可独立验证。
`init_checkpoint=/path/old.pth` 可作显式标注的微调：加载权重/Normalizer，重新建立 Adam/scheduler，不能伪称继续原训练或和随机初始化直接作等条件对照。

Ctrl+C 或正常 SIGTERM 会在当前 batch 完成后保存并退出；SIGKILL、断电或 batch 中途异常只能从最近一次完整 checkpoint 恢复。
不会把部分子步已更新的 batch 冒充完成状态保存。

查看管理：

```bash
ls -lh /absolute/path/experiment/checkpoints/
cat /absolute/path/experiment/checkpoints/best_checkpoint.json
cat /absolute/path/experiment/metadata.json
```

`metadata.latest_checkpoint` 指向最近保存；best JSON 指向按固定验证选择的 checkpoint。
默认 best 是三条序列常规帧 `mesh_free_vertex_penetration_fraction` 的等序列平均，越小越好。
这只是选择标准，仍需同时检查拉伸、稳定性和视频；step=0 不作为最佳训练 checkpoint。
没有自动删除旧 checkpoint，避免误删研究结果；实验可整体归档，但保留恢复 lineage 路径以便自动拼接。

## 5. 独立完整验证

```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0 python validate_experiment.py \
  checkpoint=/absolute/path/experiment/checkpoints/step_0000060000.pth \
  experiment.name=baseline_full_evaluation
```

默认 `protocol=configs/evaluation/common_physics.yaml`，`validation.n_steps=-1` 评估完整序列。
对 Modified 的相同预算 checkpoint 执行同一命令，仅换 checkpoint 和名称。
可指定同格式协议文件；公平比较要求两边协议一致。
`validation.n_steps=3` 仅用于调试，不是长序列验证。
验证关闭梯度，独立 runner 复制权重/Normalizer，不更新 Adam，并隔离所有训练 RNG；每序列重新建立固定材料/静止缓存。

统一评价保留原始拉伸、弯曲、惯性、重力、摩擦；接触能诊断固定为单位权重 signed-mesh cubic，所有模型相同。
它不读取 Modified 的训练势能来决定评价标准。评价定义、数据哈希、材料、时间步等形成 signature，不一致则拒绝正式比较。

| 主要验证字段 | 定义 |
|---|---|
| `mesh_vertex_penetration_fraction` | signed point-to-body-mesh distance < −1e−7 m 的衣服顶点比例。 |
| `mesh_free_vertex_penetration_fraction` | 同上，但排除 pinned/handle 顶点，避免固定点接触误差掩盖模型能力。 |
| `mesh_vertex_penetration_depth_mean_m` | 所有衣服顶点 max(−d,0) 的平均，未穿透顶点为 0。 |
| `mesh_vertex_penetration_depth_max_m` | 所有衣服顶点最大穿透深度。 |
| `mesh_penetrating_depth_mean_m/p95_m` | 只在穿透顶点上计算平均/95 分位深度，无穿透为 0。 |
| `mesh_penetration_area_fraction` | 穿透顶点静止 dual area 占总面积的比例。 |
| `mesh_vertex_clearance_min_m` | 最小有符号距离，负值为穿透；不是越小越好。 |
| `approx_*` | 保留原面中心平面近似，仅用于诊断，与 mesh 指标不是同一定义。 |
| `principal_stretch_max/min` | F 的奇异值最大/最小，衡量局部拉伸/压缩。 |
| `principal_strain_rms` | 所有面主伸长相对 1 的 RMS。 |
| `principal_extreme_strain_fraction` | 主伸长 >2 或 <0.5 的比例，用于检测异常。 |
| `pinned_error_max_m` | 固定点相对目标位置的最大误差。 |
| `cloth_speed_rms_m_s` | 帧位移/时间步的顶点速度 RMS。HOOD 原名 velocity 字段实际是位移，已换成 SI 时间单位。 |
| `cloth_acceleration_rms_m_s2` | 本帧位移相对原惯性位移的变化/时间步² 的 RMS。高值是诊断，不等于有真值的加速度误差。 |
| `frame_inference_time_ms` | 同步计时的采样推进、首帧碰撞预处理及模型调用，不含评价指标与轨迹 CPU 转存。 |
| `model_forward_time_ms` | 模型调用的同步耗时，包括其内部图构建/网络处理。 |
| `inference_frames_total_s` | 上述逐帧推理时间之和。 |
| `evaluation_wall_time_s` | 整序列推理、物理/几何评价和 CPU 转存耗时；不含之后写 NPZ/视频。 |
| `evaluation_cuda_peak_allocated_mib` | 验证进程的 CUDA 张量峰值，包含训练/评价两份模型；不能冒充单模型部署显存。 |
| `rollout_cuda_peak_increment_mib` | 本次 rollout 相对开始时已分配显存的峰值增量。 |
| `model_parameters` | 模型参数总数。 |

几何定义：全人体三角形参与最近有限三角形搜索，距离为最近点欧氏距离；winding number 决定内外。
要求闭合且一致定向的人体拓扑；自交人体保留 winding 的定义，不是修复实体并集的 SDF。
这是顶点采样评价，不能保证没有衣服边/面穿过人体；不含 CCD 或布料自碰撞评价。
本次算法未改自碰撞，因此不生成虚构的自碰撞结果；以后研究自碰撞时需另外加入三角形交叉/接近检测。
漂移/震荡由完整轨迹、速度/加速度、能量与视频共同判断，目前没有声称某个单一 RMS 能证明稳定。

输出位于新评价目录：

- `logs/validation.csv`：每 checkpoint/每序列汇总，保存 `_mean/_max/_regular_mean`；regular 排除初始化帧。
- `validation/step_*/collision_per_frame.csv`、`physics_metrics.csv`、各序列 `*_frames.csv`：完整逐帧记录。
- `validation/step_*/metrics.json`、`*_rollout.npz`：指标汇总和全部预测/人体顶点及 faces。
- `videos/step_*/*.mp4`：采样预览；默认 stride=4、最多 60 帧，NPZ 始终保留所有 rollout 帧。

首帧初始化时间步不同，第二帧按原版把初始速度重置为零。
视频默认采用 30/stride fps 的常规帧播放速度，初始化帧已标注，不能将其当作精确时间重建。
服务器 FFmpeg 的 H.264 动态库不兼容，当前可用 MPEG-4 回退，适合本地播放器。
安装 `imageio-ffmpeg==0.6.0` 后优先用其独立 H.264 编码器；当前网络下载失败没有阻止 MP4 输出。
如需保存全部帧视频，覆盖 `validation.video_stride=1 validation.video_max_frames=100000`；渲染会更慢。

## 6. 自动比较与多 seed

既可传训练目录，也可传独立验证目录；后者会根据 checkpoint 来源加载训练记录。
恢复实验传最后一个恢复目录，不要把父目录当作另一随机种子。

```bash
python compare_experiments.py \
  --baseline /absolute/path/baseline_run \
  --modified /absolute/path/modified_run \
  --axis optimizer_step_count \
  --output "$HOOD_DATA/experiments/comparisons/baseline_modified_seed57_updates"
```

输出目录必须尚不存在。`--axis global_step` 按 batch 对齐；`optimizer_step_count` 按实际 Adam 次数对齐，略过未更新的预热点。
正式比较要求相同非目标超参数/预算、seed 集合、资产/数据内容、初始化来源和统一评价 signature，并在相同 checkpoint step 比较。
历史评价 CSV 缺新 signature 时会拒绝；先对两边 checkpoint 重新执行统一验证。不能把旧 approx 指标与新 mesh 指标合并比较。
`--exploratory` 允许标注为探索性诊断的部分预算/控制变量不一致；评价定义仍必须一致，不输出正式改进结论。

```bash
python compare_experiments.py \
  --baseline /path/baseline57 /path/baseline58 /path/baseline59 \
  --modified /path/modified57 /path/modified58 /path/modified59 \
  --axis global_step \
  --output "$HOOD_DATA/experiments/comparisons/paired_three_seeds"
```

自动输出：

- `training_curves/`：total 和各物理项的 mean/last，分开的 PNG 与曲线 CSV。
- `training_state/`：LR、梯度范数、碰撞权重、raw 接触量及训练耗时。
- `validation_comparison/`、`per_frame/`：统一验证趋势与逐帧穿透率/深度。
- `metrics_comparison.csv`：每 seed/序列/checkpoint 的两组值、绝对变化、相对变化、适用时的改善百分比。
- `paired_seed_statistics.csv`：先每 seed 等序列平均，再在 seed 间取均值、样本标准差和样本数。
- `comparison.json`：配对来源、控制变量检查、评价定义与坐标轴。

基准为 0 时相对百分比未定义，留空；单 seed 的样本标准差未定义，曲线阴影置零但不冒充统计显著性。
距离、速度/加速度、重力以及不同定义的训练能量不自动判定“越小越好”。
短测耗时受并行作业、GPU 负载和初始化影响；正式速度比较需在相同硬件空闲时独立测量。
正式优劣至少检查：穿透率/深度、异常拉伸、完整长序列稳定性、视频、计算成本与多个配对 seed。
不能以不同总 loss 更低或三帧 rollout 作为证明；“不劣”还需预先指定可接受退化范围和统计方案。

## 7. TensorBoard

以后训练统一使用此 recorded 入口，自动记录各项 loss、total、LR、梯度、接触诊断和固定验证。
CSV 保留逐子步，TensorBoard 横轴默认 batch/global_step；按 Adam 次数的对比图由比较脚本生成。

```bash
tensorboard --logdir "$HOOD_DATA/experiments" --host 127.0.0.1 --port 6006
```

服务器已运行 `hood-tensorboard.service`；先查看 `systemctl --user status hood-tensorboard.service`，不要重复占用端口。
在自己的电脑做 SSH 端口转发：`ssh -N -L 6006:127.0.0.1:6006 zhudf@服务器地址`，浏览器打开 http://127.0.0.1:6006。
选择多个实验即可叠加同一 tag；各物理项、total、学习率与 validation 使用不同图。
历史 Baseline CSV 已转成事件记录；之后如需另转一份用 `python csv_to_tensorboard.py /path/run --output /path/new_events`，不要覆盖已有事件。

完整依赖版本在 `environment.txt`，Python/PyTorch/CUDA/设备/确定性设置在 `metadata.json`；配置和数据哈希与 commit 均保存。
后续新增物理项时添加 criterion 与独立配置；其日志会自动收集，但共同评价定义只有经过明确版本更新才能改变。
