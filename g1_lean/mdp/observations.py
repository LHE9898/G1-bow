# Copyright (c) 2022-2025, unitree_rl_lab
"""Observation functions for G1 bow + 3-point contact task.

=== Observation 벡터 구조 (총 44-dim) ===

[몸통 상태]
  joint_pos          (5)   hip_pitch L/R + waist yaw/roll/pitch
  joint_vel          (5)   위와 동일 관절 속도
  gravity_vec        (3)   torso frame의 중력 방향 (회전 상태 표현)
  torso_ang_vel      (3)   torso 각속도 (world frame)

[목표 오차]
  hip_pitch_err      (2)   hip_pitch - 목표(-20°)
  waist_pitch_err    (1)   waist_pitch - 목표(+15°)

[왼팔 — action 대상]
  left_arm_pos       (7)   관절 위치
  left_arm_vel       (7)   관절 속도

[왼손 contact / 테이블 방향]
  left_hand_contact  (1)   왼손목 contact binary
  left_hand_to_table (3)   왼손목 → 테이블 상면 상대 벡터 (방향 힌트)

[발 contact]
  foot_contact       (1)   양발 contact AND binary

[오른팔 — obs only, disturbance]
  right_arm_pos      (7)   관절 위치만 (속도 불필요)

합계: 5+5+3+3+2+1+7+7+1+3+1+7 = 45-dim

제외한 항목:
  - torso_rot_mat(9) → gravity_vec(3) 으로 대체 (동일 정보, 3배 압축)
  - right_arm_vel(7) → disturbance는 위치만으로 충분
"""
from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_HIP_PITCH_TARGET   = -0.1 + math.radians(-20.0)
_WAIST_PITCH_TARGET = 0.0  + math.radians(15.0)
TABLE_TOP_Z         = 0.8   # bow_env_cfg.py TABLE_HEIGHT 와 동일


# ── 몸통 관절 상태 ────────────────────────────────────────────────────────────

def body_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """지정 관절 위치 → (N, k)."""
    return env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]


def body_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """지정 관절 속도 → (N, k)."""
    return env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]


# ── Torso 자세 ────────────────────────────────────────────────────────────────

def torso_gravity_vec(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """torso body frame 의 중력 방향 벡터 → (N, 3).

    rot_mat(9) 대신 gravity_vec(3)으로 torso 기울기를 표현.
    [0,0,-1] 에서 얼마나 기울어졌는지 → pitch/roll 상태를 포함.
    """
    q = env.scene[asset_cfg.name].data.root_quat_w   # (N, 4) [w, x, y, z]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # world gravity = [0, 0, -1] 을 body frame 으로 변환
    # R^T @ [0,0,-1]  (R = rotation from body to world)
    gx = -2.0 * (x*z - w*y)
    gy = -2.0 * (y*z + w*x)
    gz = -(1.0 - 2.0 * (x*x + y*y))
    return torch.stack([gx, gy, gz], dim=-1)


def torso_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """torso 각속도 (world frame) → (N, 3)."""
    return env.scene[asset_cfg.name].data.root_ang_vel_w


# ── 목표 오차 ─────────────────────────────────────────────────────────────────

def hip_pitch_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """[left, right] hip_pitch - 목표(-20°) → (N, 2)."""
    q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    return q - _HIP_PITCH_TARGET


def waist_pitch_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """waist_pitch - 목표(+15°) → (N, 1)."""
    q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    return (q[:, 0] - _WAIST_PITCH_TARGET).unsqueeze(-1)


# ── 팔 관절 상태 ──────────────────────────────────────────────────────────────

def arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """팔 관절 위치 → (N, k). 왼팔/오른팔 공용."""
    return env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]


def arm_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """팔 관절 속도 → (N, k). 왼팔 전용 (오른팔은 pos만 사용)."""
    return env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]


# ── Contact ───────────────────────────────────────────────────────────────────

def foot_contact_flag(
    env: ManagerBasedRLEnv,
    left_foot_cfg: SceneEntityCfg,
    right_foot_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """양발 contact AND → (N, 1).  1=둘 다 접지."""
    lf = env.scene[left_foot_cfg.name].data.net_forces_w_history
    rf = env.scene[right_foot_cfg.name].data.net_forces_w_history
    lf_ok = lf[:, :, 0, :].norm(dim=-1).max(dim=1).values > threshold
    rf_ok = rf[:, :, 0, :].norm(dim=-1).max(dim=1).values > threshold
    return (lf_ok & rf_ok).float().unsqueeze(-1)


def left_hand_contact_flag(
    env: ManagerBasedRLEnv,
    left_hand_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """왼손목 contact → (N, 1).  1=테이블에 닿음."""
    force = env.scene[left_hand_cfg.name].data.net_forces_w_history[:, :, 0, :].norm(dim=-1).max(dim=1).values
    return (force > threshold).float().unsqueeze(-1)


# ── 테이블 방향 힌트 ──────────────────────────────────────────────────────────

def left_hand_to_table_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    left_hand_body_name: str = "left_wrist_roll_link",
    table_top_z: float = TABLE_TOP_Z,
) -> torch.Tensor:
    """왼손목 → 테이블 상면 중심 상대 벡터 (N, 3).

    policy 가 손을 테이블 방향으로 뻗는 데 필요한 거리/방향 힌트.
    [Δx, Δy, Δz] = table_center - hand_pos  (world frame)
    """
    robot      = env.scene[asset_cfg.name]
    hand_idx   = robot.body_names.index(left_hand_body_name)
    hand_pos_w = robot.data.body_pos_w[:, hand_idx, :]             # (N, 3)

    if "table_center_pos_w" not in env.extras:
        # fallback: events.py 가 아직 실행 안 된 경우
        table_x = env.scene.env_origins[:, 0] + 0.3 + 0.2         # TABLE_DISTANCE_X + DEPTH/2
        table_y = env.scene.env_origins[:, 1]
        table_z = torch.full((env.num_envs,), table_top_z, device=env.device)
        env.extras["table_center_pos_w"] = torch.stack([table_x, table_y, table_z], dim=-1)

    return env.extras["table_center_pos_w"] - hand_pos_w           # (N, 3)


# ── 하위 호환 alias (bow_env_cfg.py 에서 참조하는 이름) ───────────────────────
hip_pitch_joint_pos = body_joint_pos
hip_pitch_joint_vel = body_joint_vel
waist_joint_pos     = body_joint_pos
waist_joint_vel     = body_joint_vel