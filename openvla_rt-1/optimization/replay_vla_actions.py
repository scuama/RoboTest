"""
Replay VLA (Vision-Language-Action) model saved actions using ManiSkill (simpler_env).

Supports all VLA models: OpenVLA, RT-1, Octo, etc.

Behavior:
- Prefer `actions.npy` in the episode dir; fall back to `actions.json`.
- Load `options.json` (if present) and pass it to `env.reset(seed=..., options=options)`.
- Step through saved actions in order, render each resulting frame and save images to
  `<episode_dir>/replay_images/<t>.jpg`.
- Save `replay_log.json` with per-step info.

updated by zeqin
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image


def load_actions(episode_dir: str):
    np_path = os.path.join(episode_dir, 'actions.npy')
    json_path = os.path.join(episode_dir, 'actions.json')
    if os.path.exists(np_path):
        try:
            arr = np.load(np_path, allow_pickle=True)
            return [None if a is None else np.array(a) for a in arr]
        except Exception:
            pass
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return [None if a is None else np.array(a) for a in data]
    raise FileNotFoundError('No actions file found in ' + episode_dir)


def load_options(episode_dir: str):
    p = os.path.join(episode_dir, 'options.json')
    if os.path.exists(p):
        try:
            with open(p, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_image(arr: np.ndarray, path: str):
    # arr expected uint8 HxWx3 or float in [0,255]
    if arr is None:
        return
    if arr.dtype != np.uint8:
        try:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        except Exception:
            arr = (arr * 255).astype(np.uint8)
    im = Image.fromarray(arr)
    im.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode_dir', required=True, help='Path to episode directory (contains actions.npy or actions.json and options.json)')
    parser.add_argument('--task', required=True, help='ManiSkill task name, e.g., google_robot_pick_customizable')
    parser.add_argument('--seed', type=int, default=None, help='Optional seed to override options')
    parser.add_argument('--camera', type=str, default=None, help='Optional camera name for rendering')
    parser.add_argument('--render_every', type=int, default=1, help='Render every N steps (default 1)')
    args = parser.parse_args()

    episode_dir = args.episode_dir
    actions = load_actions(episode_dir)
    options = load_options(episode_dir) or {}

    # try import simpler_env and helper
    try:
        import simpler_env
        try:
            from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
        except Exception:
            get_image_from_maniskill2_obs_dict = None
    except Exception as e:
        raise RuntimeError('Failed to import simpler_env: ' + str(e))

    env = simpler_env.make(args.task)
    # choose seed: CLI > options['seed'] > None
    seed = args.seed if args.seed is not None else options.get('seed', None)

    try:
        obs, reset_info = env.reset(seed=seed, options=options)
    except Exception:
        # fallback to reset without options
        obs, reset_info = env.reset(seed=seed)

    out_img_dir = os.path.join(episode_dir, 'replay_images')
    os.makedirs(out_img_dir, exist_ok=True)

    replay_info = {}

    for t, a in enumerate(actions):
        # Cap length if options specified fewer steps
        if 'max_episode_steps' in options and t >= int(options['max_episode_steps']):
            break

        if a is None:
            # step with no action
            try:
                obs, reward, terminated, truncated, info = env.step(None)
            except Exception:
                # some envs expect numpy array; pass
                obs, reward, terminated, truncated, info = env.step(None)
        else:
            a = np.asarray(a, dtype=np.float64)
            try:
                obs, reward, terminated, truncated, info = env.step(a)
            except Exception:
                # try wrapping in dict if env expects dict format
                try:
                    action_dict = {"control_mode": env.agent.control_mode, "action": a}
                    obs, reward, terminated, truncated, info = env.step(action_dict)
                except Exception:
                    raise

        # 添加 check_grasp 判断（针对 move/put_on 任务）
        # 检查是否真正抓取了物体
        try:
            # 尝试获取 source_obj 并检查抓取状态
            if hasattr(env.unwrapped, 'episode_source_obj'):
                source_obj = env.unwrapped.episode_source_obj
                is_grasped = env.unwrapped.agent.check_grasp(source_obj)
                info['is_src_obj_grasped'] = is_grasped
        except Exception:
            pass  # 如果没有 source_obj 或者不是支持的任务，忽略
        
        replay_info[t] = info

        # render and save image (after step)
        if (t % args.render_every) == 0:
            img = None
            # prefer observation-based image (matches model view)
            try:
                if get_image_from_maniskill2_obs_dict is not None:
                    img = get_image_from_maniskill2_obs_dict(env, obs)
                else:
                    # try render_rgb_array
                    try:
                        img = env.render_rgb_array(camera_name=args.camera) if args.camera else env.render_rgb_array()
                    except Exception:
                        # fallback to render()
                        try:
                            rgb = env.render()
                            if isinstance(rgb, np.ndarray):
                                img = rgb
                        except Exception:
                            img = None
            except Exception:
                img = None

            img_path = os.path.join(out_img_dir, f'{t:04d}.jpg')
            try:
                if img is not None:
                    save_image(img, img_path)
            except Exception:
                pass

        if terminated:
            print('Terminated at step', t)
            break

    # save replay log (convert numpy types to native Python types for JSON serialization)
    try:
        def convert_to_serializable(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj
        
        serializable_info = convert_to_serializable(replay_info)
        with open(os.path.join(episode_dir, 'replay_log.json'), 'w') as f:
            json.dump(serializable_info, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save replay_log.json: {e}")

    env.close()


if __name__ == '__main__':
    main()

#python VLATest/optimization/replay_vla_actions.py --episode_dir /path/to/result_dir/0 --task google_robot_pick_customizable --render_every 1

#python3 optimization/replay_vla_actions.py --episode_dir /mnt/disk1/decom/VLATest/newresult/t-grasp_n-1_o-m3_s-2498586606/openvla-7b_2024/t-grasp_n-1_o-m3_s-2498586606/openvla-7b_2024/0 --task google_robot_pick_customizable --render_every 1