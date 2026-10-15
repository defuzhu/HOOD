"""Read-only CSV -> curated TensorBoard mirror; original logs are never changed."""
import argparse
import json
import time
from pathlib import Path
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata

CORE = {'tshirt_shape00_01_01', 'tshirt_shape00_07_02', 'tshirt_shape00_104_54'}
# source, tag, conversion, explanation. Equal-sequence means, excluding initialization.
METRICS = [
 ('mesh_free_vertex_penetration_fraction_regular_mean','01_contact/free_penetration_pct',100.,'非固定顶点穿透比例；百分比，越低越好。baseline 旧记录缺此项，不补零。'),
 ('mesh_vertex_penetration_depth_max_m_regular_mean','01_contact/mean_frame_max_depth_mm',1000.,'每帧最大穿透深度的时间均值，再对序列等权平均；不是全局最坏值。'),
 ('principal_strain_rms_regular_mean','02_deformation/principal_strain_rms',1.,'主应变 RMS；结合运动与材料判断，重点防止摩擦引入额外拉伸。'),
 ('stretching_energy_loss_regular_mean','02_deformation/stretch_energy',1.,'统一评价器的拉伸能量，三组定义一致。'),
 ('common_contact_slip_speed_mean_m_s_regular_mean','03_sliding/relative_speed_m_s',1.,'接触区相对切向速度；更低不一定更好，可能过度粘滞。'),
 ('common_contact_free_fraction_regular_mean','03_sliding/contact_free_pct',100.,'非固定顶点中进入2毫米接触范围的比例；无接触时滑移为零，需联读。'),
 ('bending_energy_loss_regular_mean','04_secondary/bending_energy',1.,'统一弯曲能量；并非单调越低越好。'),
 ('cloth_acceleration_rms_m_s2_regular_mean','04_secondary/acceleration_rms',1.,'衣服加速度 RMS，辅助观察异常振荡；受动作影响。'),
 ('approx_penetration_fraction_regular_mean','05_legacy_approx/penetration_pct',100.,'旧面中心平面近似，三组均有，仅作辅助；不是精确网格穿透。'),
 ('frame_inference_time_ms_regular_mean','06_efficiency/inference_ms',1.,'逐帧推理毫秒；共享GPU，性能比较仅供参考。'),
]
TRAIN = ['total_physical_loss_mean','stretching_energy_loss_mean','bending_energy_loss_mean',
         'inertia_loss_mean','collision_penalty_loss_mean','friction_energy_loss_mean']
STATE = ['learning_rate','grad_norm_preclip_mean','collision_weight_mean',
         'friction_load_max_n_mean','friction_slip_speed_active_mean_m_s_mean','friction_inside_fraction_mean']
GUIDE = '''# 精简视图阅读顺序

1. **01_core3_tshirt**：三模型共同的三条T-shirt动作，首先比较相同步数。
2. **02_extra_tshirt_55_27 / 03_longskirt_4motions / 04_tight_dress_2motions**：额外压力测试，不与核心三条整体均值混比。旧模型没有这些完整组时不显示。
3. 每组内部按编号看 **contact → deformation → sliding → secondary → legacy_approx → efficiency**。

所有验证曲线排除初始化帧、对该组动作等权平均。没有记录的指标不补零；因此 baseline 没有精确网格历史曲线，旧 contact 没有滑移历史曲线。

滑移减少不自动表示改善，需联读接触比例、拉伸和穿透。mean_frame_max_depth_mm 是每帧最大深度的均值，不是最坏一帧。

**10_training_losses 的 total/contact/friction 定义可能不同，不能据曲线高低判断模型优劣。** 学习率/梯度/载荷在 11_training_state。

原始逐动作、mean/max/regular_mean 全量曲线仍保留在原 TensorBoard 6006。该视图只读取 CSV，每60秒更新。横轴为 global_step，显示值未经额外平滑；建议 UI Smoothing=0 看尖峰。
'''


def validation_groups(frame):
    specs = [('01_core3_tshirt','tshirt',CORE),
             ('02_extra_tshirt_55_27','tshirt',{'tshirt_shape00_55_27'}),
             ('03_longskirt_4motions','longskirt',CORE|{'tshirt_shape00_55_27'}),
             ('04_tight_dress_2motions','tight_dress',{'tshirt_shape00_07_02','tshirt_shape00_104_54'})]
    for name,garment,sequences in specs:
        group=frame[(frame.garment==garment)&frame.sequence.isin(sequences)]
        if len(group)==len(sequences) and set(group.sequence)==sequences:
            yield name,group


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='append',required=True,help='label=/absolute/run/path')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--watch',action='store_true')
    p.add_argument('--interval',type=float,default=60.)
    args=p.parse_args()
    if args.interval<10: p.error('interval must be >=10 seconds')
    runs={label:Path(path).resolve() for label,path in (r.split('=',1) for r in args.run)}
    if len(runs)!=len(args.run) or any(not x or '/' in x or x in ('.','..') for x in runs):
        p.error('Unique simple run labels required')
    # Exclusive ownership: never append duplicate historical events on restart.
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'sources.json').write_text(json.dumps({k:str(v) for k,v in runs.items()},indent=2))
    writers={k:SummaryWriter(str(args.output/k)) for k in runs}
    seen={k:{} for k in runs}
    for w in writers.values(): w.add_text('00_read_me',GUIDE,0)
    def emit(label,tag,value,step,description):
        if pd.notna(value) and step>seen[label].get(tag,-1):
            # torch 2.0 SummaryWriter.add_scalar has no description argument.
            summary=Summary(value=[Summary.Value(tag=tag, simple_value=float(value),
                              metadata=SummaryMetadata(summary_description=description))])
            writers[label].file_writer.add_summary(summary,int(step))
            seen[label][tag]=step
    try:
        while True:
            for label,run in runs.items():
                for filename in ('validation','train'):
                    path=run/'logs'/f'{filename}.csv'
                    if not path.exists() or path.stat().st_size==0:continue
                    wanted={'global_step','sequence','garment'}|set(TRAIN+STATE)|{x[0] for x in METRICS}
                    # Ignore an incomplete final line from a concurrently flushed CSV.
                    with path.open('rb') as f:
                        content=f.read();content=content[:content.rfind(b'\n')+1]
                    if not content:continue
                    import io
                    frame=pd.read_csv(io.BytesIO(content),usecols=lambda x:x in wanted)
                    if filename=='validation':
                        for step,rows in frame.groupby('global_step',sort=True):
                            for scope,group in validation_groups(rows):
                                for key,tag,scale,note in METRICS:
                                    if key in group and group[key].notna().all():
                                        emit(label,scope+'/'+tag,group[key].mean()*scale,step,note)
                    else:
                        for key in TRAIN+STATE:
                            if key not in frame:continue
                            tag=('10_training_losses/' if key in TRAIN else '11_training_state/')+key
                            subset=frame[frame.global_step>seen[label].get(tag,-1)]
                            for step,value in zip(subset.global_step,subset[key]):
                                emit(label,tag,value,step,'CSV 原始值；不同训练目标的 loss 不直接比较性能。')
                writers[label].flush()
            if not args.watch:break
            time.sleep(args.interval)
    finally:
        for w in writers.values():w.close()


if __name__=='__main__':main()
