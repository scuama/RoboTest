#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化实验运行脚本 - 简化版本
用于重新运行已保存的BDDL配置，不需要动态场景修改
"""

import sys
import os

# ⚠️ 必须在任何导入之前设置警告抑制
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 设置环境变量抑制警告
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 抑制TensorFlow日志

# 抑制JAX/Flax的DeprecationWarning
import logging
logging.captureWarnings(True)
logging.getLogger('py.warnings').setLevel(logging.ERROR)

# 设置环境变量
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# 继续抑制robosuite日志
logging.getLogger("robosuite_logs").setLevel(logging.ERROR)
logging.getLogger("robosuite").setLevel(logging.ERROR)

import yaml
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict
import math
import cv2
import time

# 添加路径（需要用户根据自己的环境配置）
# sys.path.insert(0, '/path/to/LIBERO')
# sys.path.insert(0, '/path/to/openpi/src')

# 获取当前脚本目录
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

# 导入OpenPI
from openpi.training import config as _config
from openpi.policies import policy_config


def _quat2axisangle(quat):
    """将四元数转换为轴角表示"""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    w = np.clip(w, -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0, abs_tol=1e-6):
        return np.zeros(3)
    return (np.array([x, y, z]) * 2.0 * math.acos(w)) / den


def extract_robot_state_from_obs(obs):
    """从LIBERO环境观察中提取8维机器人状态"""
    eef_pos = obs["robot0_eef_pos"]
    eef_quat = obs["robot0_eef_quat"]
    gripper_qpos = obs["robot0_gripper_qpos"]
    
    eef_axisangle = _quat2axisangle(eef_quat)
    gripper_pos = np.mean(gripper_qpos)
    
    robot_state = np.concatenate([
        eef_pos,           # 3维：末端位置
        eef_axisangle,     # 3维：旋转轴角
        gripper_qpos[:1],  # 1维：夹爪位置
        [gripper_pos]      # 1维：夹爪平均位置
    ])
    
    return robot_state.astype(np.float32)


def process_camera_image(obs, camera_key):
    """处理相机图像"""
    image = obs[camera_key]
    
    # 旋转180度（与pi05_libero_visual_inference_fixed.py保持一致）
    image = image[::-1, ::-1]
    
    # 确保格式为uint8 (H,W,C)
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    return image


def load_pi05_model(checkpoint_dir: str):
    """加载Pi0.5模型"""
    try:
        config_name = os.path.basename(checkpoint_dir)
        config = _config.get_config(config_name)
        model = policy_config.create_trained_policy(config, checkpoint_dir)
        print("✅ Pi0.5模型加载完成")
        return model, config
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def find_task_id_by_name(task_name: str, suite: str = "libero_90") -> int:
    """通过任务名查找task_id"""
    from libero.libero import benchmark
    
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite]()
    
    for i, task in enumerate(task_suite.tasks):
        if task.name == task_name:
            return i
    
    raise ValueError(f"找不到任务: {task_name}")


def run_multi_stage_group(env, config: Dict, group_config: Dict, group_name: str,
                          output_dir: Path, model, model_config, task_suite, task_id: int) -> Dict:
    """运行多阶段实验组"""
    from libero.libero.envs import OffScreenRenderEnv
    
    group_dir = output_dir / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    
    episodes_per_group = config['execution']['episodes_per_group']
    max_steps = config['execution']['max_steps_per_episode']
    
    results = []
    stage_configs = group_config['stages']
    
    for episode_idx in range(episodes_per_group):
        print(f"\n{'='*60}")
        print(f"📍 Episode {episode_idx + 1}/{episodes_per_group}")
        print(f"{'='*60}")
        
        # 创建episode目录
        episode_dir = group_dir / f"episode_{episode_idx}"
        episode_dir.mkdir(exist_ok=True)
        
        # 创建分视角的images目录
        frontview_dir = episode_dir / "images" / "frontview"
        wrist_dir = episode_dir / "images" / "wrist"
        frontview_dir.mkdir(parents=True, exist_ok=True)
        wrist_dir.mkdir(parents=True, exist_ok=True)
        
        step_offset = 0  # 步骤偏移量（跨stage连续）
        stage_info = []  # 记录每个stage的步数范围
        episode_success = True
        
        # 逐个运行stage
        for stage_idx, stage_config in enumerate(stage_configs, 1):
            stage_name = stage_config.get('stage_name', f'stage{stage_idx}')
            bddl_file = stage_config['bddl_file']
            instruction = stage_config['instruction']
            
            print(f"\n🎬 Stage {stage_idx}/{len(stage_configs)}: {stage_name}")
            print(f"   BDDL: {bddl_file}")
            print(f"   指令: {instruction}")
            
            # 切换BDDL文件，创建新环境
            if not bddl_file.startswith('/'):
                # 尝试相对于项目根目录查找
                bddl_path = SCRIPT_DIR.parent / bddl_file
                if not bddl_path.exists():
                    bddl_path = Path(bddl_file)
            else:
                bddl_path = Path(bddl_file)
            
            env_args = {
                "bddl_file_name": str(bddl_path),
                "camera_heights": 224,
                "camera_widths": 224,
                "has_renderer": False,
                "has_offscreen_renderer": True,
                "use_camera_obs": True,
                "camera_names": ["frontview", "robot0_eye_in_hand"],
                "control_freq": 20,
            }
            
            try:
                env = OffScreenRenderEnv(**env_args)
                env.seed(config['execution']['seed_start'] + episode_idx)
            except Exception as e:
                # 捕获物体放置失败等异常
                error_msg = str(e)
                if "Cannot place all objects" in error_msg:
                    print(f"⚠️  Stage {stage_idx} 环境初始化失败：物体放置空间不足")
                    print(f"   BDDL: {bddl_file}")
                    episode_success = False
                    # 返回失败结果
                    summary = {
                        "success_count": 0,
                        "total_episodes": episodes_per_group,
                        "success_rate": 0.0,
                        "error": "placement_failed_in_stage",
                        "failed_stage": stage_idx
                    }
                    return env, summary
                else:
                    raise
            
            # 重置环境
            obs = env.reset()
            env.sim.data.qvel[:] = 0
            env.sim.forward()
            for _ in range(5):
                env.sim.step()
            
            # 重置模型状态
            model.reset()
            
            # 动作序列缓存
            action_cache = []
            action_cache_step = 0
            inference_count = 0
            stage_step_count = 0
            stage_success = False
            
            for step in range(max_steps):
                # 提取状态和图像
                robot_state = extract_robot_state_from_obs(obs)
                base_image = process_camera_image(obs, "frontview_image")
                wrist_image = process_camera_image(obs, "robot0_eye_in_hand_image")
                
                # 检查是否需要新的推理
                if len(action_cache) == 0 or action_cache_step >= len(action_cache):
                    inference_count += 1
                    
                    model_input = {
                        "observation/state": robot_state,
                        "observation/image": base_image,
                        "observation/wrist_image": wrist_image,
                        "prompt": instruction
                    }
                    
                    with torch.no_grad():
                        result = model.infer(model_input)
                        action_sequence = result["actions"]
                    
                    action_cache = action_sequence
                    action_cache_step = 0
                    
                    if step % 50 == 0 or inference_count <= 3:
                        print(f"  [推理{inference_count}] 步骤{step+1}: 生成{len(action_cache)}步动作序列")
                
                # 从缓存中获取动作
                action = action_cache[action_cache_step]
                action_cache_step += 1
                
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy().squeeze()
                
                # 执行动作
                obs, reward, done, info = env.step(action)
                stage_step_count += 1
                
                # ✅ 使用全局步骤编号（step_offset + stage_step_count）
                global_step = step_offset + stage_step_count
                
                # 保存图像
                try:
                    front_bgr = cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR)
                    wrist_bgr = cv2.cvtColor(wrist_image, cv2.COLOR_RGB2BGR)
                    
                    frontview_filename = frontview_dir / f"step_{global_step:05d}.png"
                    wrist_filename = wrist_dir / f"step_{global_step:05d}.png"
                    
                    cv2.imwrite(str(frontview_filename), front_bgr)
                    cv2.imwrite(str(wrist_filename), wrist_bgr)
                except Exception as e:
                    print(f"    ⚠️  图像保存错误: {e}")
                
                if step % 50 == 0:
                    print(f"    Step {global_step} (stage内第{stage_step_count}步)")
                
                if done:
                    stage_success = True
                    print(f"    ✓ Stage {stage_idx} 完成于步骤 {global_step}")
                    break
            
            # 记录stage信息
            stage_info.append({
                'stage_name': stage_name,
                'stage_index': stage_idx,
                'step_range': [step_offset + 1, step_offset + stage_step_count],
                'success': stage_success,
                'inference_count': inference_count
            })
            
            # 更新step_offset
            step_offset += stage_step_count
            
            if not stage_success:
                print(f"    ✗ Stage {stage_idx} 未完成，终止episode")
                episode_success = False
                break
            
            # 关闭当前stage的环境
            env.close()
        
        # 保存stage信息
        stage_info_file = episode_dir / "stage_info.json"
        with open(stage_info_file, 'w') as f:
            json.dump({
                'episode_index': episode_idx,
                'total_steps': step_offset,
                'stages': stage_info,
                'overall_success': episode_success
            }, f, indent=2)
        
        results.append({
            'episode': episode_idx,
            'success': episode_success,
            'total_steps': step_offset,
            'stages': stage_info
        })
        
        print(f"\n📊 Episode {episode_idx} 总结:")
        print(f"   总步数: {step_offset}")
        print(f"   成功: {'✓' if episode_success else '✗'}")
    
    # 生成组总结
    success_count = sum(1 for r in results if r['success'])
    summary = {
        'group_name': group_name,
        'episodes': episodes_per_group,
        'success_count': success_count,
        'success_rate': success_count / episodes_per_group if episodes_per_group > 0 else 0,
        'results': results
    }
    
    with open(group_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ 多阶段组 {group_name} 完成")
    print(f"   成功率: {summary['success_rate']*100:.1f}% ({success_count}/{episodes_per_group})")
    print(f"{'='*60}\n")
    
    return env, summary


def run_group_with_cache(env, config: Dict, group_config: Dict, group_name: str, 
                         output_dir: Path, model, model_config, task_suite, task_id: int) -> Dict:
    """运行单个实验组 - 带动作序列缓存 + 多阶段支持"""
    
    # ✅ 检测是否为多阶段实验
    if 'stages' in group_config:
        print(f"\n{'='*60}")
        print(f"🔬 运行多阶段实验组: {group_name}")
        print(f"   描述: {group_config['description']}")
        print(f"   阶段数: {len(group_config['stages'])}")
        print(f"{'='*60}\n")
        return run_multi_stage_group(env, config, group_config, group_name,
                                     output_dir, model, model_config, task_suite, task_id)
    
    print(f"\n{'='*60}")
    print(f"🔬 运行实验组: {group_name}")
    print(f"   描述: {group_config['description']}")
    print(f"   指令: {group_config['instruction']}")
    print(f"   遮挡: {'是' if group_config['use_obstruction'] else '否'}")
    
    # ✅ 支持组级别的BDDL文件切换
    if 'bddl_file' in group_config:
        group_bddl_path = group_config['bddl_file']
        if not group_bddl_path.startswith('/'):
            group_bddl_path = SCRIPT_DIR.parent / group_bddl_path
            if not group_bddl_path.exists():
                group_bddl_path = Path(group_bddl_path)
        else:
            group_bddl_path = Path(group_bddl_path)
        
        print(f"   BDDL文件: {group_bddl_path}")
        
        # 重新创建环境以使用新的BDDL文件
        from libero.libero.envs import OffScreenRenderEnv
        env_args = {
            "bddl_file_name": str(group_bddl_path),
            "camera_heights": 224,
            "camera_widths": 224,
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "camera_names": ["frontview", "robot0_eye_in_hand"],
            "control_freq": 20,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(config['execution']['seed_start'])
        print(f"   ✅ 已切换到新BDDL场景")
    
    print(f"{'='*60}\n")
    
    # ✅ 在每个组开始前，先做一次完整重置（清除之前组的影响）
    print("🔄 重置环境状态...")
    env.reset()
    env.sim.data.qvel[:] = 0
    env.sim.forward()
    for _ in range(10):
        env.sim.step()
    print("✅ 环境状态已清理\n")
    
    group_dir = output_dir / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    
    episodes_per_group = config['execution']['episodes_per_group']
    max_steps = config['execution']['max_steps_per_episode']
    
    results = []
    
    # ✅ 加载任务初始状态（修复 PyTorch 2.6 兼容性）
    init_states = None
    # 只有使用标准LIBERO任务时才加载初始状态
    if task_suite is not None:
        # 临时禁用初始状态加载，避免维度不匹配问题
        print(f"  ℹ️  使用默认随机初始化（跳过预保存状态）")
    else:
        print(f"  ℹ️  自定义BDDL场景，使用默认随机初始化")
    # try:
    #     import torch
    #     from libero.libero import get_libero_path
    #     init_states_path = Path(get_libero_path("init_states")) / task_suite.tasks[task_id].problem_folder / task_suite.tasks[task_id].init_states_file
    #     init_states = torch.load(init_states_path, weights_only=False)
    # except Exception as e:
    #     print(f"  ⚠️  警告：无法加载初始状态: {e}")
    #     print(f"  ⚠️  将使用默认随机初始化")
    
    for episode_idx in range(episodes_per_group):
        print(f"\n📍 Episode {episode_idx + 1}/{episodes_per_group}")
        
        # 创建episode目录
        episode_dir = group_dir / f"episode_{episode_idx}"
        episode_dir.mkdir(exist_ok=True)
        
        # 创建分视角的images目录
        frontview_dir = episode_dir / "images" / "frontview"
        wrist_dir = episode_dir / "images" / "wrist"
        frontview_dir.mkdir(parents=True, exist_ok=True)
        wrist_dir.mkdir(parents=True, exist_ok=True)
        
        # ✅ 完全重置环境：先reset，再加载初始状态
        obs = env.reset()
        if init_states is not None and len(init_states) > 0:
            obs = env.set_init_state(init_states[episode_idx % len(init_states)])
            print(f"  ✓ 加载初始状态 (state {episode_idx % len(init_states)})")
        
        # ✅ 确保物理状态稳定（清零所有速度）
        env.sim.data.qvel[:] = 0
        env.sim.forward()
        for _ in range(5):
            env.sim.step()
        
        # BDDL中已定义堆叠场景
        # 所有配置都使用 use_bddl_stacking: true，不需要动态修改
        if group_config.get('use_bddl_stacking', False):
            print(f"  ℹ️  使用BDDL定义的堆叠场景")
        
        # 运行推理
        instruction = group_config['instruction']
        episode_success = False
        step_count = 0
        inference_count = 0
        
        print(f"  指令: {instruction}")
        
        # 重置模型状态
        model.reset()
        
        # 动作序列缓存机制
        action_cache = []
        action_cache_step = 0
        
        for step in range(max_steps):
            # 提取状态和图像
            robot_state = extract_robot_state_from_obs(obs)
            base_image = process_camera_image(obs, "frontview_image")
            wrist_image = process_camera_image(obs, "robot0_eye_in_hand_image")
            
            # 检查是否需要新的推理
            if len(action_cache) == 0 or action_cache_step >= len(action_cache):
                # 需要新推理
                inference_count += 1
                
                # 构建模型输入
                model_input = {
                    "observation/state": robot_state,
                    "observation/image": base_image,
                    "observation/wrist_image": wrist_image,
                    "prompt": instruction
                }
                
                # Pi0.5推理
                with torch.no_grad():
                    result = model.infer(model_input)
                    action_sequence = result["actions"]  # 完整动作序列
                
                # 更新缓存
                action_cache = action_sequence
                action_cache_step = 0
                
                if step % 50 == 0 or inference_count <= 3:
                    print(f"  [推理{inference_count}] 步骤{step+1}: 生成{len(action_cache)}步动作序列")
            
            # 从缓存中获取动作
            action = action_cache[action_cache_step]
            action_cache_step += 1
            
            # 转换为numpy
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy().squeeze()
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            step_count += 1
            
            # 每步都保存图像
            try:
                # 转换颜色通道 RGB -> BGR (OpenCV格式)
                front_bgr = cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR)
                wrist_bgr = cv2.cvtColor(wrist_image, cv2.COLOR_RGB2BGR)
                
                # 保存图像文件
                frontview_filename = frontview_dir / f"step_{step_count:05d}.png"
                wrist_filename = wrist_dir / f"step_{step_count:05d}.png"
                
                cv2.imwrite(str(frontview_filename), front_bgr)
                cv2.imwrite(str(wrist_filename), wrist_bgr)
            except Exception as e:
                print(f"    ⚠️  图像保存错误: {e}")
            
            if step % 50 == 0:
                print(f"    Step {step_count}")
            
            if done:
                # ✅ 修改判断逻辑：如果提前退出（done=True），则认为成功
                episode_success = True
                print(f"    ✓ 任务提前完成于步骤 {step_count}/{max_steps}")
                break
        
        # 如果达到最大步数仍未完成，则判定为失败
        if step_count >= max_steps and not episode_success:
            print(f"    ✗ 达到最大步数 {max_steps}，任务未完成")
        
        # 保存结果
        episode_result = {
            'episode': episode_idx,
            'success': episode_success,
            'steps': step_count,
            'inference_count': inference_count,
            'instruction': instruction,
            'obstruction': group_config['use_obstruction']
        }
        
        results.append(episode_result)
        
        # 保存到文件
        with open(episode_dir / "result.json", 'w') as f:
            json.dump(episode_result, f, indent=2)
        
        status = "✅ 成功" if episode_success else "❌ 失败"
        print(f"  {status} (步数: {step_count}, 推理: {inference_count}次)")
    
    # 计算统计
    success_count = sum(1 for r in results if r['success'])
    success_rate = success_count / len(results) if results else 0
    
    summary = {
        'group_name': group_name,
        'description': group_config['description'],
        'instruction': group_config['instruction'],
        'use_obstruction': group_config['use_obstruction'],
        'episodes': len(results),
        'success_count': success_count,
        'success_rate': success_rate,
        'results': results
    }
    
    # 保存组统计
    with open(group_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 {group_name} 统计:")
    print(f"   成功: {success_count}/{len(results)} ({success_rate*100:.1f}%)")
    
    return env, summary


def run_experiment(config_path: str):
    """运行完整实验"""
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n{'='*60}")
    print(f"🚀 启动实验: {config['experiment']['name']}")
    print(f"   描述: {config['experiment']['description']}")
    print(f"{'='*60}\n")
    
    # 创建输出目录（直接使用配置中的results_dir，不再添加实验名）
    output_dir = Path(config['output']['results_dir'])
    
    # 如果目录已存在，清空它
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    print(f"📁 结果目录: {output_dir}\n")
    
    # 初始化环境
    print("🔧 初始化LIBERO环境...")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    
    task_name = config['task']['task_name']
    suite = config['task']['suite']
    
    # ✅ 支持自定义BDDL文件或custom suite
    if 'bddl_file' in config['task'] or suite == 'custom':
        # 使用自定义BDDL文件（如果指定）或者由各组自己指定
        if 'bddl_file' in config['task']:
            custom_bddl_path = config['task']['bddl_file']
            if not custom_bddl_path.startswith('/'):
                custom_bddl_path = SCRIPT_DIR.parent / custom_bddl_path
                if not custom_bddl_path.exists():
                    custom_bddl_path = Path(custom_bddl_path)
            else:
                custom_bddl_path = Path(custom_bddl_path)
            task_bddl_file = custom_bddl_path
            print(f"   任务: {task_name}")
            print(f"   使用自定义BDDL: {task_bddl_file}")
        else:
            # custom suite，由各组自己指定BDDL
            task_bddl_file = None
            print(f"   任务: {task_name}")
            print(f"   使用custom suite，各组将指定各自的BDDL文件")
        
        task_suite = None
        task_id = 0
    else:
        # 使用标准LIBERO任务
        task_id = find_task_id_by_name(task_name, suite)
        print(f"   任务: {task_name}")
        print(f"   Task ID: {task_id}")
        
        # 创建环境
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[suite]()
        task = task_suite.get_task(task_id)
        
        task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    
    # 创建初始环境（如果有默认BDDL）
    env = None
    if task_bddl_file is not None:
        env_args = {
            "bddl_file_name": str(task_bddl_file),
            "camera_heights": 224,
            "camera_widths": 224,
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "camera_names": ["frontview", "robot0_eye_in_hand"],  # ✅ 使用frontview
            "control_freq": 20,
        }
        
        env = OffScreenRenderEnv(**env_args)
        env.seed(config['execution']['seed_start'])
        
        print("✅ 环境初始化完成\n")
    else:
        print("   环境将在各组中根据BDDL文件创建\n")
    
    # 加载Pi0.5模型
    print("🤖 加载Pi0.5模型...")
    checkpoint_dir = config['execution']['checkpoint_dir']
    model, model_config = load_pi05_model(checkpoint_dir)
    
    if model is None:
        print("❌ 模型加载失败，无法继续实验")
        if env is not None:
            env.close()
        return
    
    # 运行所有实验组
    all_summaries = []
    
    for group_idx, group_config in enumerate(config['groups'], 1):
        group_name = f"group{group_idx}_{group_config['name']}"
        
        try:
            # ✅ 自定义BDDL时task_suite可能为None，不使用初始状态
            env, summary = run_group_with_cache(
                env, config, group_config, group_name,
                output_dir, model, model_config, task_suite, task_id
            )
            all_summaries.append(summary)
        except Exception as e:
            print(f"❌ 实验组 {group_name} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 生成总报告
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        'experiment': config['experiment']['name'],
        'timestamp': timestamp,
        'task': task_name,
        'groups': all_summaries
    }
    
    with open(output_dir / "report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"✅ 实验完成!")
    print(f"{'='*60}")
    print(f"\n📊 总结:")
    for summary in all_summaries:
        print(f"   {summary['group_name']}: {summary['success_rate']*100:.1f}% "
              f"({summary['success_count']}/{summary['episodes']})")
    
    print(f"\n📁 结果保存在: {output_dir}")
    print(f"\n运行分析脚本:")
    print(f"   python experiments/obstruction/scripts/analyze_results.py \\")
    print(f"       --results_dir {output_dir}")
    
    if env is not None:
        env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                       help="实验配置文件路径")
    
    args = parser.parse_args()
    
    run_experiment(args.config)
