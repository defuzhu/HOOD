# 精简 TensorBoard（只读镜像）

原 6006 保留全量数据。新增 6007 精简视图读取三次训练的 CSV，每 60 秒同步，
不改变正在训练的冻结代码、优化器、checkpoint 或原始 events。

阅读顺序：

1. `01_core3_tshirt`：共同三条动作、相同步数，首先看这一组。
2. 组内 `01_contact`（穿透）→`02_deformation`（变形）→`03_sliding`（滑移）。
3. `04_secondary` 是弯曲/加速度；`05_legacy_approx` 为旧平面近似；`06_efficiency` 为共享 GPU 下的时间。
4. `02_extra_tshirt_55_27`、`03_longskirt_4motions`、`04_tight_dress_2motions` 单独看泛化压力测试。
5. `10_training_losses` 仅看训练过程/尖峰；不同定义的 total/contact/friction 不直接比较好坏。
6. `11_training_state` 看学习率、梯度、碰撞权重和摩擦载荷。

每张标量图带 summary_description 说明；Text 页有 `00_read_me` 中文说明。
验证剔除初始化帧，按组内动作等权平均。只有完整的预定义动作组才输出；
不把三条长裙平均误标成四条，不把缺失历史指标填零。
baseline 旧日志没有精确网格历史指标；contact 旧日志没有滑移历史指标。
这两种缺失不是零穿透/零滑移。已保存轨迹的统一补算结果另外存于 comparisons。

接触区滑移越小不一定越好，需联读接触比例、拉伸、穿透。`mean_frame_max_depth_mm`
是逐帧最大穿透深度的均值，不是全序列最坏深度。百分比和毫米已转换为阅读单位。
建议 Smoothing=0 先观察真实尖峰；默认 Scalar 卡片按数字前缀阅读。

从自己的电脑使用与原 TensorBoard 相同的服务器地址，转发新端口：

```bash
ssh -N -L 6007:127.0.0.1:6007 zhudf@服务器地址
```

浏览器打开 http://127.0.0.1:6007 。原 6006 不受影响。

脚本：`tensorboard_summary.py --run 标签=/absolute/run/path ... --output /new/path --watch`。
输出必须是新目录，防止重启时重复写历史点；重建时换新目录并让 TensorBoard 指向新目录。
脚本不自动发现新的独立评价目录；旧模型完成十组重评后可把该目录作为单独 label 加入。
CSV 是原始记录，精简 events 可随时重建。
