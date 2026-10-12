# 六组固定验证与通用推理

新训练的 baseline、四个 contact 消融配置和两份 smoke 配置现在都包含六组验证：
动作 `tshirt_shape00_01_01`、`tshirt_shape00_07_02`、`tshirt_shape00_104_54`，
分别配 `tshirt` 和 `longskirt` 模板。源动作文件只提供 SMPL 参数；文件名中的
`tshirt` 不限制服装模板。长裙是 `longskirt`，不是 `dress`。

## 配置在哪里

- `configs/baseline/postcvpr_seed57.yaml`
- `configs/contact/{geometry_v1,area_cubic_v2,potential_only_v2,relaxed_barrier_v2}_seed57.yaml`
- `configs/smoke/{baseline,modified}_seed57.yaml`
- `configs/evaluation/common_physics.yaml`：独立验证使用的共同协议。

在原有 `validation.sequences` 后加入：

```yaml
- id: tshirt_shape00_01_01
  garment: longskirt
- id: tshirt_shape00_07_02
  garment: longskirt
- id: tshirt_shape00_104_54
  garment: longskirt
```

同一 `validation` 下新增：

```yaml
rollout_name_template: '{experiment}_seed{seed}_{sequence}_{garment}_rollout.npz'
```

`utils/rollout_io.py` 生成安全文件名；`utils/fixed_validation.py` 检查名称不重复，
按模板保存 NPZ，并让视频使用同一文件名前缀。`validate_experiment.py` 保留
checkpoint 中的训练实验名称，避免独立验证的临时名称替换模型身份。
例如：

```
modified_contact_relaxed_barrier_seed57_tshirt_shape00_01_01_longskirt_rollout.npz
```

每次验证仍保存在独立的 step 子目录；不会覆盖不同训练步的结果。
本次仅改变验证集合和输出命名，不改变训练损失。
已完成训练及旧 checkpoint 内的三序列配置不会自动改写；恢复训练保留原配置。
若对旧模型进行六组评估，使用 `validate_experiment.py` 的共同协议即可。
比较 baseline 与 modified 时，两者必须采用相同六组协议；不能把旧三组均值
直接与新六组均值比较。

## 通用 notebook

在项目根目录打开 `HOOD_Universal_Inference.ipynb`，选择 `Python (hood_clean)` 内核。
首个代码单元配置项目、数据目录和 GPU；第二个代码单元修改：

- `CHECKPOINT`：兼容的 HOOD postcvpr 模型。
- `MOTION_FILE`：标准 SMPL 动作；支持可信 `.pkl` 字典、数值 `.npz`、完整 pose `.npy`。
- `GARMENT`：模板字典中的衣服，如 `tshirt`、`longskirt`。
- `BETAS_OVERRIDE`：None 沿用动作体型，或给十维 betas / 体型文件。
- `SMPL_MODEL`：可覆盖兼容 SMPL 身体模型文件；需与衣服模板的蒙皮绑定兼容。
- `SOURCE_FPS`：输入动作帧率；`FRAME_START/STOP/STRIDE` 控制截取。
- `MAX_STEPS`：-1 完整序列；初次建议 30。
- `SAVE_VIDEO`：是否渲染 MP4。

依次运行全部单元。每次运行在 `$HOOD_DATA/inference/` 建立唯一目录，保存
规范化输入、最终配置、来源及体型元数据、NPZ 和可选 MP4。
标准输入是轴角 `body_pose[T,69] + global_orient[T,3] + transl[T,3] + betas[10]`，
或完整 `poses[T,72]`（也支持相应三维关节数组）。缺失平移、根旋转和体型会明确
提示使用零值。SMPL-X、SMPL-H、BVH、旋转矩阵等必须先显式转换，不能直接截断维度。
数据采用米和弧度，坐标系需匹配 HOOD。不同帧率/极端体型/陌生动作可能超出模型训练分布。

保留 HOOD 原有初始化及手部姿态处理；首帧有特殊松弛时间。T 个输入帧产生 T−2
个预测帧，NPZ 保存源帧索引及 timestep。裸推理不额外执行训练损失或物理评价。
`utils/inference_inputs.py` 负责格式验证和复制转换，不更改源动作。
`render_baseline.py` 新增 `--fps`，抽帧后按相应播放帧率输出。
服务器 MPEG-4 回退编码可能不能在浏览器播放，可下载后用 VLC 查看。
