"""Headless fixed-rollout preview; uses no interactive viewer or extra dependencies."""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def render_rollout(rollout, output, stride=4, max_frames=30, fps=30):
    if stride <= 0 or max_frames <= 0:
        raise ValueError('stride and max-frames must be positive')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and positive')
    data = np.load(rollout)
    indices = np.arange(0, len(data['pred']), stride)[:max_frames]
    # Plot y-up SMPL coordinates as (x,z,y), with identical camera/bounds across frames.
    cloth, body = data['pred'][indices][..., [0, 2, 1]], data['obstacle'][indices][..., [0, 2, 1]]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    all_points = np.concatenate([cloth.reshape(-1, 3), body.reshape(-1, 3)])
    lower, upper = all_points.min(0), all_points.max(0)
    center = (lower + upper) / 2
    radius = max(upper - lower) * 0.55
    for i, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        setter(center[i] - radius, center[i] + radius)
    ax.set_box_aspect([1, 1, 1]); ax.view_init(elev=15, azim=-65)
    ax.set_xlabel('x'); ax.set_ylabel('z'); ax.set_zlabel('y (up)')
    body_mesh = Poly3DCollection(body[0][data['obstacle_faces']], facecolor='#b7bac2', alpha=0.22, linewidth=0)
    cloth_mesh = Poly3DCollection(cloth[0][data['cloth_faces']], facecolor='#337bbd', alpha=0.9, linewidth=0)
    ax.add_collection3d(body_mesh); ax.add_collection3d(cloth_mesh)
    def update(frame):
        body_mesh.set_verts(body[frame][data['obstacle_faces']])
        cloth_mesh.set_verts(cloth[frame][data['cloth_faces']])
        ax.set_title(f'Fixed rollout: frame {indices[frame]} (initialization included)')
        return body_mesh, cloth_mesh
    animation = FuncAnimation(fig, update, frames=len(indices), interval=100)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == '.mp4':
        import shutil
        try:
            import imageio_ffmpeg
            encoder = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            encoder = shutil.which('ffmpeg')
        if encoder is None:
            import sys
            environment_encoder = Path(sys.prefix) / 'bin' / 'ffmpeg'
            if environment_encoder.is_file():
                encoder = str(environment_encoder)
        if encoder is None:
            import imageio_ffmpeg
            encoder = imageio_ffmpeg.get_ffmpeg_exe()
        plt.rcParams['animation.ffmpeg_path'] = encoder
        import subprocess
        encoders = subprocess.check_output([encoder, '-hide_banner', '-encoders'], text=True)
        # Conda's old OpenH264 build may load an incompatible shared library.
        # Native MPEG-4 remains a working offline-player fallback.
        codec = 'libx264' if 'libx264' in encoders else 'mpeg4'
        writer = FFMpegWriter(fps=fps / stride, codec=codec, bitrate=2000, extra_args=['-pix_fmt', 'yuv420p'])
    else:
        writer = PillowWriter(fps=fps / stride)
    animation.save(output, writer=writer)
    plt.close(fig)
    print(output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rollout', type=Path)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--max-frames', type=int, default=30)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fps', type=float, default=30, help='Source rollout frames per second')
    args = parser.parse_args()
    render_rollout(args.rollout, args.output or args.rollout.with_suffix('.gif'), args.stride, args.max_frames, args.fps)


if __name__ == '__main__':
    main()
