# Copyright (c) 2022-2025, unitree_rl_lab
"""Event functions for G1 bow + 3-point contact task."""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_robot_to_standing(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
) -> None:
    """직립 default 자세로 리셋 + 각종 버퍼 초기화."""
    asset      = env.scene[asset_cfg.name]
    joint_pos  = asset.data.default_joint_pos[env_ids].clone()
    joint_vel  = torch.zeros_like(joint_pos)
    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]

    asset.write_root_state_to_sim(root_state, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # 발 초기 위치 캐싱
    robot = env.scene[asset_cfg.name]
    names = robot.body_names
    l_idx = names.index("left_ankle_roll_link")
    r_idx = names.index("right_ankle_roll_link")

    if "init_left_foot_pos_xy" not in env.extras:
        env.extras["init_left_foot_pos_xy"]  = torch.zeros(env.num_envs, 2, device=env.device)
        env.extras["init_right_foot_pos_xy"] = torch.zeros(env.num_envs, 2, device=env.device)

    env.extras["init_left_foot_pos_xy"][env_ids]  = (
        robot.data.body_pos_w[env_ids, l_idx, :2].detach().clone()
    )
    env.extras["init_right_foot_pos_xy"][env_ids] = (
        robot.data.body_pos_w[env_ids, r_idx, :2].detach().clone()
    )

    # 3-point contact 유지 타이머 리셋
    if "hold_timer" not in env.extras:
        env.extras["hold_timer"] = torch.zeros(env.num_envs, device=env.device)
    env.extras["hold_timer"][env_ids] = 0.0

    # 쓰러짐 감지 버퍼 리셋
    if "has_fallen" not in env.extras:
        env.extras["has_fallen"] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env.extras["has_fallen"][env_ids] = False

    # 테이블 상면 중심 위치 캐싱 (env_origins 기준)
    # bow_env_cfg.py 의 TABLE 상수와 동일하게 유지
    TABLE_DISTANCE_X = 0.6    # 로봇 정면 → 테이블 앞면 (m)
    TABLE_WIDTH      = 0.4    # Y 방향 폭
    TABLE_DEPTH      = 0.4    # X 방향 깊이
    TABLE_HEIGHT     = 0.65    # 테이블 높이 (= 상면 z, 두께 없음)
    if "table_center_pos_w" not in env.extras:
        env.extras["table_center_pos_w"] = torch.zeros(env.num_envs, 3, device=env.device)

    tx = env.scene.env_origins[:, 0] + TABLE_DISTANCE_X + TABLE_DEPTH / 2.0
    ty = env.scene.env_origins[:, 1]
    tz = torch.full((env.num_envs,), TABLE_HEIGHT, device=env.device)
    env.extras["table_center_pos_w"] = torch.stack([tx, ty, tz], dim=-1)