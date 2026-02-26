# Copyright (c) 2022-2025, unitree_rl_lab
"""Reward functions for G1 bow + 3-point contact task.

[부호 규칙]
  penalty 함수 → 양수 반환,  RewTerm weight → 음수
  tracking 함수 → [0,1] 반환, RewTerm weight → 양수

[핵심 설계 의도]
  목표: 로봇이 쓰러지는 과정에서 왼팔로 테이블을 짚어 3-point contact 달성

  NG 전략 차단:
    - 몸통(torso/pelvis/waist)이 테이블에 닿으면 강한 패널티
    - 오른팔이 테이블에 닿아도 패널티
    → 반드시 왼팔(left_wrist/elbow/hand)로만 지지해야 보상

  OK 전략 유도:
    1. 왼손목을 테이블 방향으로 뻗어 가까워질수록 보상 (reaching)
    2. 왼팔이 테이블에 닿으면 즉시 보상 (contact)
    3. 양발 + 왼팔 3-point, 정지 4초 이상 → 큰 보상 (hold)

[수정 사항 - 팔 reaching 버그 수정]
  문제 1: left_wrist_forward_of_torso 가 world x 축 기준으로 비교
          → 로봇이 y 방향을 바라보면 리워드 = 0, 팔이 안 움직임
  수정:   로봇의 heading 방향(yaw)을 기준으로 forward 거리 계산

  문제 2: left_hand_reaching_reward 가 table_center_pos_w 없으면 항상 0
          → env.extras 에 테이블 위치를 등록하는 코드가 없음
  수정:   scene 의 table rigid body 위치를 직접 읽어서 계산
"""
from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_yaw

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_HIP_PITCH_TARGET   = -0.1 + math.radians(-20.0)
_WAIST_PITCH_TARGET = 0.0  + math.radians(15.0)
_FORCE_THRESH       = 1.0
_HOLD_TIME_SEC      = 4.0
_VEL_STILL_THRESH   = 0.05


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _get_table_pos_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """테이블 상면 중심 위치 (N, 3).

    lean_env_cfg 에서 table RigidObject 를 'table' 이름으로 등록했다면
    scene["table"] 로 직접 읽는다.
    없으면 env_origins + 하드코딩 offset 으로 fallback.
    """
    if "table" in env.scene.keys():
        # RigidObject: root_pos_w 는 body 중심 → 상면은 z += half_height
        TABLE_HALF_H = 0.5  # lean_env_cfg TABLE_HEIGHT=1.0 기준
        pos = env.scene["table"].data.root_pos_w.clone()  # (N, 3)
        pos[:, 2] += TABLE_HALF_H
        return pos
    else:
        # fallback: env_origins 기준 고정 offset
        TABLE_X, TABLE_W, TABLE_H = 1.0, 0.5, 1.0
        pos = env.scene.env_origins.clone()
        pos[:, 0] += TABLE_X + TABLE_W * 0.5
        pos[:, 2] += TABLE_H
        return pos


def _heading_vec(quat_w: torch.Tensor) -> torch.Tensor:
    """torso quaternion(wxyz) → yaw 기준 forward 단위 벡터 (N,3)=(x방향,y방향,0).

    torso 의 yaw 만 추출해 (cos_yaw, sin_yaw, 0) 반환.
    """
    # quat: (w, x, y, z)
    w, x, y, z = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3]
    # yaw = atan2(2(wz+xy), 1-2(y²+z²))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    fwd = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)
    return fwd  # (N, 3)


# ── 몸통 tracking ─────────────────────────────────────────────────────────────

def hip_pitch_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    theta_d: float = _HIP_PITCH_TARGET,
) -> torch.Tensor:
    q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    return torch.exp(-4.0 * (q - theta_d).pow(2).mean(dim=-1))


def waist_pitch_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    theta_d: float = _WAIST_PITCH_TARGET,
) -> torch.Tensor:
    q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    return torch.exp(-4.0 * (q[:, 0] - theta_d).pow(2))


# ── 다리 고정 ─────────────────────────────────────────────────────────────────

def leg_symmetry_penalty(
    env: ManagerBasedRLEnv,
    left_cfg: SceneEntityCfg,
    right_cfg: SceneEntityCfg,
) -> torch.Tensor:
    q_l = env.scene[left_cfg.name].data.joint_pos[:, left_cfg.joint_ids]
    q_r = env.scene[right_cfg.name].data.joint_pos[:, right_cfg.joint_ids]
    return (q_l - q_r).pow(2).sum(dim=-1)


def leg_fixed_pose_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    q_cur = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q_def = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return (q_cur - q_def).pow(2).sum(dim=-1)


# ── waist 측면 고정 ───────────────────────────────────────────────────────────

def waist_lateral_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    return q.pow(2).sum(dim=-1)


# ── 몸통/오른팔 테이블 contact 패널티 ────────────────────────────────────────

def body_table_contact_penalty(
    env: ManagerBasedRLEnv,
    body_contact_sensor_cfg: SceneEntityCfg,
    threshold: float = _FORCE_THRESH,
) -> torch.Tensor:
    """몸통 또는 오른팔이 테이블에 닿으면 패널티. r ∈ {0,1}"""
    sensor = env.scene[body_contact_sensor_cfg.name]
    # net_forces_w_history: (N, H, B, 3)
    f    = sensor.data.net_forces_w_history
    peak = f.norm(dim=-1).max(dim=1).values.max(dim=-1).values  # (N,)
    return (peak > threshold).float()


# ── 왼팔 reaching / contact 보상 ─────────────────────────────────────────────

def left_wrist_forward_of_torso(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    left_hand_body_name: str = "left_wrist_roll_link",
) -> torch.Tensor:
    """왼손목이 torso CoM 보다 로봇 진행 방향 앞에 있을수록 보상.

    [수정] world x 가 아닌 torso yaw 기준 forward 성분으로 계산.
    로봇이 어느 방향을 바라보든 정확하게 동작.

    r = clamp(dot(hand_rel, fwd), 0, 1.0)  → [0, 1]
    """
    robot    = env.scene[asset_cfg.name]
    hand_idx = robot.body_names.index(left_hand_body_name)
    hand_pos = robot.data.body_pos_w[:, hand_idx, :]   # (N, 3)
    torso_pos = robot.data.root_pos_w                  # (N, 3)
    rel       = hand_pos - torso_pos                   # (N, 3)  torso 기준 상대 위치

    # torso yaw 기준 forward 단위벡터
    fwd = _heading_vec(robot.data.root_quat_w)         # (N, 3)

    # forward 성분만 추출
    forward_dist = (rel * fwd).sum(dim=-1)             # (N,) dot product
    return forward_dist.clamp(min=0.0, max=1.0)


def left_shoulder_pitch_forward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    init_pitch: float = 0.3,
) -> torch.Tensor:
    """shoulder_pitch 가 init(0.3rad) 보다 작을수록 (음수 방향) 보상.
    G1 left_shoulder_pitch_joint: 음수 = 팔이 앞으로 뻗리는 방향(flexion).
    r = clamp(init_pitch - q, 0, π)  → [0, π]
    """
    q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]  # (N, 1)
    return (init_pitch - q[:, 0]).clamp(min=0.0, max=math.pi)


def left_hand_reaching_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    left_hand_body_name: str = "left_wrist_roll_link",
) -> torch.Tensor:
    """왼손목 → 테이블 상면 중심 3D 거리 역수 보상.

    [수정] env.extras 의존 제거 → scene["table"] 또는 env_origins 로 직접 계산.
    env.extras 가 비어있어도 항상 gradient 있는 신호 제공.

    r = 1 / (1 + dist_3d)  → (0, 1]
    """
    robot    = env.scene[asset_cfg.name]
    hand_idx = robot.body_names.index(left_hand_body_name)
    hand_pos = robot.data.body_pos_w[:, hand_idx, :]   # (N, 3)

    table_pos = _get_table_pos_w(env)                  # (N, 3)
    dist      = (hand_pos - table_pos).norm(dim=-1)    # (N,)
    return 1.0 / (1.0 + dist)


def left_hand_contact_reward(
    env: ManagerBasedRLEnv,
    left_hand_sensor_cfg: SceneEntityCfg,
    threshold: float = _FORCE_THRESH,
) -> torch.Tensor:
    """왼팔이 테이블에 닿으면 보상. r ∈ {0,1}"""
    sensor = env.scene[left_hand_sensor_cfg.name]
    # net_forces_w_history: (N, H, B, 3)
    f    = sensor.data.net_forces_w_history
    peak = f.norm(dim=-1).max(dim=1).values.max(dim=-1).values  # (N,)
    return (peak > threshold).float()


# ── 3-point contact 정지 보상 ─────────────────────────────────────────────────

def three_point_contact_hold_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    left_foot_sensor_cfg: SceneEntityCfg,
    right_foot_sensor_cfg: SceneEntityCfg,
    left_hand_sensor_cfg: SceneEntityCfg,
    body_contact_sensor_cfg: SceneEntityCfg,
    hold_time_sec: float = _HOLD_TIME_SEC,
    vel_thresh: float    = _VEL_STILL_THRESH,
    force_thresh: float  = _FORCE_THRESH,
) -> torch.Tensor:
    """양발 + 왼팔 3-point contact, 4초 이상 정지 → 큰 보상."""
    robot = env.scene[asset_cfg.name]
    dt    = env.step_dt

    def _contact_on(cfg):
        # net_forces_w_history: (N, H, B, 3)
        f = env.scene[cfg.name].data.net_forces_w_history
        return f.norm(dim=-1).max(dim=1).values.max(dim=-1).values > force_thresh

    lf_ok   = _contact_on(left_foot_sensor_cfg)
    rf_ok   = _contact_on(right_foot_sensor_cfg)
    lh_ok   = _contact_on(left_hand_sensor_cfg)
    body_ng = _contact_on(body_contact_sensor_cfg)

    still  = robot.data.root_ang_vel_w.norm(dim=-1) < vel_thresh
    all_ok = lf_ok & rf_ok & lh_ok & ~body_ng & still

    if "hold_timer" not in env.extras:
        env.extras["hold_timer"] = torch.zeros(env.num_envs, device=env.device)

    timer = env.extras["hold_timer"]
    timer[all_ok]  += dt
    timer[~all_ok]  = 0.0

    achieved = timer >= hold_time_sec
    progress = (timer / hold_time_sec).clamp(0.0, 1.0)
    return torch.where(achieved, torch.full_like(timer, 10.0), progress * 0.5)


# ── 안정성 패널티 ─────────────────────────────────────────────────────────────

def torso_ang_vel_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_ang_vel_w.norm(dim=-1).pow(2)


def joint_vel_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.joint_vel.pow(2).sum(dim=-1)


def foot_slide_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    left_foot_body_name:  str = "left_ankle_roll_link",
    right_foot_body_name: str = "right_ankle_roll_link",
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    names = asset.body_names
    l_idx = names.index(left_foot_body_name)
    r_idx = names.index(right_foot_body_name)
    lf_xy = asset.data.body_pos_w[:, l_idx, :2]
    rf_xy = asset.data.body_pos_w[:, r_idx, :2]

    if "init_left_foot_pos_xy" not in env.extras:
        env.extras["init_left_foot_pos_xy"]  = lf_xy.detach().clone()
        env.extras["init_right_foot_pos_xy"] = rf_xy.detach().clone()

    return (
        (lf_xy - env.extras["init_left_foot_pos_xy"]).pow(2).sum(dim=-1) +
        (rf_xy - env.extras["init_right_foot_pos_xy"]).pow(2).sum(dim=-1)
    )