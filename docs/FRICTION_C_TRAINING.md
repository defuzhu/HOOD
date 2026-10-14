# C 组正式训练：扩展摩擦验证

本次配置 `friction/lagged_coulomb_extended_seed57`：seed=57、60000 batch、
batch_size=1、原网络/训练数据/随机材料/Adam 和 rollout 调度，随机初始化训练。
运行时必须确认：

- collision.contact_geometry=signed_mesh
- collision.contact_potential=area_relaxed_barrier
- friction.mode=lagged_coulomb，mu=0.5，velocity_epsilon=0.01 m/s
- 摩擦绑定同一 collision criterion，载荷包含实际碰撞权重调度。
- 不启用载荷限幅，不新增 CCD、自摩擦或推理接触修正；不宣称严格无穿透。

原六组 T-shirt/长裙验证保留，另外增加：

| 动作文件 ID | 衣服 | 选择依据（预先选择，不根据新模型表现挑选） |
|---|---|---|
| tshirt_shape00_55_27 | tshirt | 整体位移较小的留出动作，补充局部相对运动/过度粘滞检查 |
| tshirt_shape00_55_27 | longskirt | 检查相同动作下较长自由布料的随动、拉伸及粘滞 |
| tshirt_shape00_07_02 | tight_dress | 贴身服装的持续接触与滑移压力测试 |
| tshirt_shape00_104_54 | tight_dress | 本地四种留出动作中轴角逐帧变化较大的动作，检查快速接触变化 |

动作类型不凭文件 ID 猜测；上述依据来自本地 SMPL 参数统计。轴角差只作为粗筛选，
不是角速度的严格 SO(3) 测量。shape00 的四种动作都没有出现在训练 split 中（跨 shape
按动作 ID 检查）。训练服装仍为 tshirt，longskirt/tight_dress 是跨服装泛化测试。
未增加训练样本，也未改变训练预算。

每 5000 batch 执行十组验证，每组最多 120 个预测帧；短序列自然取全长。
保留原六组帧范围，新增组单独分析，不能把十组整体平均直接与历史三/六组平均比较。
T-shirt 六组中的三个动作属于更贴近训练服装的测试，长裙/贴身裙用于暴露失效情况。
最佳 checkpoint 仍按配置的十组平均非固定顶点穿透比例选取，它不代表摩擦综合最优；
正式消融优先比较相同 60000 batch checkpoint，并分服装、分动作报告。

保存逐子步与平均/末步物理 loss、lr、裁剪前梯度范数、优化器次数、碰撞权重、
摩擦载荷/接触比例/内部顶点比例/滑移速度和低速区比例，CSV 与 TensorBoard 同时写。
统一验证保存穿透深度/比例、拉伸/压缩、惯性/弯曲、速度/加速度、接触滑移、
推理时间和显存，以及每帧 CSV、NPZ 轨迹；每组另渲染最多 30 帧视频（stride=4）。
视频仅为抽帧预览，完整验证轨迹保存在 NPZ。每 2000 batch 保存 checkpoint，
独立目录包含 Git commit、最终配置、环境、数据哈希和完整恢复状态。

旧 A 模型必须用同一协议重评：

```bash
python validate_experiment.py \
  checkpoint=/home/zhudf/data/hood_data/hood_data/experiments/modified_contact_relaxed_barrier_seed57_20261011T084658Z_fc628647/checkpoints/step_0000060000.pth \
  protocol=configs/evaluation/friction_extended.yaml \
  experiment.name=friction_reference_A_extended
```

## 值得观察的动作/服装假设

- T-shirt 的反复摆臂、躯干转动/弯曲：关注肩部、腋下和衣摆的相对滑动，可能受益于
  人体运动时间对齐和重心跟踪，但也可能被过大载荷拉住。
- 贴身裙的行走、较大步幅、腿部弯曲：预计更容易显现法向载荷驱动摩擦的差异；
  同时更容易产生粘滞、异常拉伸和初始穿透放大的问题。
- 长裙的快速转向、跨步、摆腿：适合检查裙摆随动与高频抖动；本次没有自碰撞，
  裙摆互相穿过不能据此归咎于/声称被新摩擦解决。
- 平缓动作或接触很少的宽松服装：可能变化较小，作为弱响应对照有价值。

以上是物理机制推测，不是已验证的改善。滑移减少需同时检查接触比例、穿透、
拉伸和视频；不以总 loss 或摩擦增量势下降直接判断成功。
