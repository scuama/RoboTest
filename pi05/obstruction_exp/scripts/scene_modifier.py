#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
场景修改工具 - 用于在LIBERO场景中添加遮挡物
"""

import numpy as np
from typing import Dict, Optional, Tuple

class SceneModifier:
    """场景修改器：将遮挡物叠放在目标物体上"""
    
    def __init__(self, env, config: Dict):
        self.env = env
        self.config = config
        self.sim = env.sim
        self.model = env.sim.model
        
    def find_object_by_pattern(self, pattern: str) -> Optional[int]:
        """通过名称模式找到MuJoCo body_id"""
        for i in range(self.model.nbody):
            body_name = self.model.body_id2name(i)
            if pattern in body_name:
                return i
        return None
    
    def get_object_position(self, body_id: int) -> np.ndarray:
        """获取物体的世界坐标位置"""
        return self.sim.data.body_xpos[body_id].copy()
    
    def _find_joint_by_body(self, body_id: int) -> Optional[Tuple[int, int]]:
        """
        找到控制指定body的joint
        返回: (joint_qpos_addr, joint_dof_count) 或 None
        """
        body_name = self.model.body_id2name(body_id)
        
        # 遍历所有joint
        for joint_id in range(self.model.njnt):
            joint_name = self.model.joint_id2name(joint_id)
            joint_bodyid = self.model.jnt_bodyid[joint_id]
            
            # 检查joint是否属于目标body
            if joint_bodyid == body_id:
                joint_type = self.model.jnt_type[joint_id]
                qpos_addr = self.model.jnt_qposadr[joint_id]
                
                # free joint: 7 DOF (3 pos + 4 quat)
                if joint_type == 0:  # mjJNT_FREE
                    return (qpos_addr, 7)
                # slide joint: 1 DOF
                elif joint_type == 1:  # mjJNT_SLIDE
                    return (qpos_addr, 1)
                # hinge joint: 1 DOF
                elif joint_type == 2:  # mjJNT_HINGE
                    return (qpos_addr, 1)
        
        return None
    
    def _move_object(self, body_id: int, new_pos: np.ndarray) -> bool:
        """移动指定物体到新位置"""
        joint_info = self._find_joint_by_body(body_id)
        
        if joint_info is None:
            return False
        
        qpos_addr, dof_count = joint_info
        qpos = self.sim.data.qpos.copy()
        
        if dof_count == 7:  # free joint
            qpos[qpos_addr:qpos_addr+3] = new_pos  # 位置
            qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]  # 四元数(w,x,y,z)
        elif dof_count == 1:  # slide/hinge joint
            qpos[qpos_addr] = new_pos[2]  # 假设是z轴
        
        self.sim.data.qpos[:] = qpos
        return True
    
    def apply_obstruction(self) -> bool:
        """应用遮挡配置"""
        obstruction_config = self.config['scene']['obstruction']
        
        if not obstruction_config.get('enabled', False):
            return False
        
        # 找到目标物体和遮挡物
        target_pattern = self.config['scene']['target_object']['mujoco_body_pattern']
        obstruction_pattern = self.config['scene']['obstruction_object']['mujoco_body_pattern']
        
        target_body_id = self.find_object_by_pattern(target_pattern)
        obstruction_body_id = self.find_object_by_pattern(obstruction_pattern)
        
        if target_body_id is None:
            print(f"❌ 找不到目标物体: {target_pattern}")
            return False
        
        if obstruction_body_id is None:
            print(f"❌ 找不到遮挡物: {obstruction_pattern}")
            return False
        
        # 如果配置了目标物体的初始位置，先移动目标物体
        if 'initial_position' in self.config['scene']['target_object']:
            target_initial_pos = np.array(self.config['scene']['target_object']['initial_position'])
            if self._move_object(target_body_id, target_initial_pos):
                print(f"✅ 目标物体已移动到初始位置: {target_initial_pos}")
            else:
                print(f"⚠️  无法移动目标物体到初始位置")
        
        # 获取目标物体位置（如果刚移动过，使用新位置）
        target_pos = self.get_object_position(target_body_id)
        
        # 计算遮挡物新位置
        offset = np.array(obstruction_config['offset'])
        new_pos = target_pos + offset
        
        # 移动遮挡物
        if not self._move_object(obstruction_body_id, new_pos):
            print(f"❌ 找不到遮挡物的joint")
            return False
        
        # 移动其他干扰物品（如果配置了）
        if 'distractor_objects' in self.config['scene']:
            for distractor in self.config['scene']['distractor_objects']:
                distractor_pattern = distractor['mujoco_body_pattern']
                distractor_body_id = self.find_object_by_pattern(distractor_pattern)
                
                if distractor_body_id is not None:
                    distractor_pos = np.array(distractor['position'])
                    distractor_desc = distractor.get('description', distractor_pattern)
                    if self._move_object(distractor_body_id, distractor_pos):
                        print(f"✅ 干扰物已移动: {distractor_desc} -> {distractor_pos}")
                    else:
                        print(f"⚠️  无法移动干扰物: {distractor_desc}")
        
        # 重置速度（避免物体飞天）
        self.sim.data.qvel[:] = 0
        
        # 多步forward稳定物理状态
        for _ in range(10):
            self.sim.forward()
            self.sim.step()
        
        # 验证遮挡物位置
        new_actual_pos = self.get_object_position(obstruction_body_id)
        distance = np.linalg.norm(new_actual_pos - new_pos)
        
        print(f"✅ 遮挡物已移动:")
        print(f"   目标位置: {target_pos}")
        print(f"   遮挡物新位置: {new_actual_pos}")
        print(f"   偏移量: {offset}")
        print(f"   位置误差: {distance:.4f}m")
        
        return True
    
    def get_scene_info(self) -> Dict:
        """获取场景信息（用于调试）"""
        target_pattern = self.config['scene']['target_object']['mujoco_body_pattern']
        obstruction_pattern = self.config['scene']['obstruction_object']['mujoco_body_pattern']
        
        target_body_id = self.find_object_by_pattern(target_pattern)
        obstruction_body_id = self.find_object_by_pattern(obstruction_pattern)
        
        info = {
            'target_object': {
                'body_id': target_body_id,
                'position': self.get_object_position(target_body_id) if target_body_id else None,
                'body_name': self.model.body_id2name(target_body_id) if target_body_id else None
            },
            'obstruction_object': {
                'body_id': obstruction_body_id,
                'position': self.get_object_position(obstruction_body_id) if obstruction_body_id else None,
                'body_name': self.model.body_id2name(obstruction_body_id) if obstruction_body_id else None
            }
        }
        
        return info
