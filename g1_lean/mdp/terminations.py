# Copyright (c) 2022-2025, unitree_rl_lab
"""Termination functions for G1 bow task.

종료 조건:
  torso_fallen_forward : torso pitch > 45° → 앞으로 쓰러짐  ← 핵심 확인 항목
  torso_roll_exceeded  : torso roll  > 40° → 옆으로 넘어짐
  torso_too_low        : torso z     < 0.3m → 완전히 쓰러짐
  time_out             : episode_length_s 경과

env.extras["has_fallen"] 에 쓰러진 환경 bool 기록.
"""
from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length - 1


def torso_roll_exceeded(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    limit_deg: float = 40.0,
) -> torch.Tensor:
    """Torso roll 초과 → 옆으로 넘어짐."""
    roll, _, _ = euler_xyz_from_quat(env.scene[asset_cfg.name].data.root_quat_w)
    fallen = roll.abs() > math.radians(limit_deg)
    _log_fallen(env, fallen)
    return fallen


def torso_too_low(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_height: float = 0.3,
) -> torch.Tensor:
    """Torso CoM 높이 미달 → 바닥에 쓰러짐."""
    fallen = env.scene[asset_cfg.name].data.root_pos_w[:, 2] < min_height
    _log_fallen(env, fallen)
    return fallen


def torso_fallen_forward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    limit_deg: float = 45.0,
) -> torch.Tensor:
    """Torso pitch > 45° → 앞으로 쓰러짐 (복원 불가 시점).

    hip_pitch 15° 숙이기 이후 무게중심이 앞으로 넘어가 쓰러지는지 감지.
    """
    _, pitch, _ = euler_xyz_from_quat(env.scene[asset_cfg.name].data.root_quat_w)
    fallen = pitch > math.radians(limit_deg)
    _log_fallen(env, fallen)
    return fallen


def _log_fallen(env, fallen: torch.Tensor) -> None:
    """쓰러짐 여부를 extras 에 OR 기록 (여러 termination 중복 호출 안전)."""
    if "has_fallen" not in env.extras:
        env.extras["has_fallen"] = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    env.extras["has_fallen"] |= fallen

def body_table_contact(
    env: ManagerBasedRLEnv,
    body_contact_sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """몸통(torso)이 테이블에 닿으면 즉시 종료.

    왼팔이 아닌 몸통으로 테이블을 버티는 전략을 원천 차단.
    r ∈ {True, False}
    """
    sensor = env.scene[body_contact_sensor_cfg.name]
    f      = sensor.data.net_forces_w_history
    peak   = f.norm(dim=-1).max(dim=1).values.max(dim=-1).values  # (N,)
    return peak > threshold