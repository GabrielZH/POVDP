import argparse
import os
import sys

sys.path.append('.')

from core.dataset import add_demo_gen_args, generate_avd_expert_demos


def get_save_path(data=None, n_episodes=None, map_bbox=None, map_res=None, ori_res=None, resize=None,
                  target=None, min_traj_len=None, max_steps=None, sample_free=None, **kwargs):
    return os.path.join(data, "pose_nav" if ori_res else "pos_nav",
                        f"{n_episodes}_{'_'.join(map(str, map_bbox))}__{'_'.join(map(str, map_res))}__{ori_res or 'pos'}"
                        f"_{'_'.join(map(str, resize))}_{target}_{min_traj_len}_{max_steps}_{sample_free}")


def gen_avd_expert_demos(config):
    config['episode_in_mem'] = True
    config['obsv_in_mem'] = True
    config['trans_in_mem'] = True
    config['in_ram'] = True
    config['domain'] = 'avd'
    config['avd_data'] = config['data']
    generate_avd_expert_demos(get_save_path, **config)


def add_avd_env_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--avd_workers', 
        type=int, 
        default=8, 
        help='store embeddings in ram'
    )
    parser.add_argument(
        '--map_bbox', '-bbox',
        type=int, 
        nargs=4, 
        default=(-15, -15, 15, 15),
        help='bounding box for map (h1, w1, h2, w2)'
    )
    parser.add_argument(
        '--map_res', '-res', 
        type=int, 
        nargs=2, 
        default=(40, 40),
        help='map resolution'
    )
    parser.add_argument(
        '--ori_res', '-ori', 
        type=int, 
        default=None, 
        help='orientation resolution'
    )
    parser.add_argument(
        '--resize', '-imsize', 
        type=int, 
        nargs=2, 
        default=None,
        help='resize images',
    )
    parser.add_argument(
        '--target', 
        type=str,
        default=None, 
        help='name of the target object'
    )
    parser.add_argument(
        '--min_traj_len', '-trajlen', 
        required=True, 
        type=int,
        help='minimum threshold of trajectory length'
    )
    parser.add_argument(
        '--target_size_ratio', '-tsr', 
        default=0.6, 
        type=float, 
        help='target size ratio'
    )
    parser.add_argument(
        '--max_steps', '-mxs', 
        type=int, 
        default=None, 
        help='max steps before end of episode'
    )
    parser.add_argument(
        '--sample_free', 
        type=int, 
        default=8,
        help='number of free space samples per pixel' 
    )
    parser.add_argument(
        '--segment_config_path', 
        default='swin3d/Swin3D/SemanticSeg/config/s3dis/swin3D_RGB_S.yaml', 
        help='path to segmentation config file'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', 
        default="data/avd", 
        help="root path to save data"
    )
    add_demo_gen_args(parser)
    add_avd_env_args(parser)
    config = parser.parse_args()

    gen_avd_expert_demos(vars(config))


if __name__ == "__main__":
    main()
