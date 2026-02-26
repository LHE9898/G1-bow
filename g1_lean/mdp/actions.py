# Copyright (c) 2022-2025, unitree_rl_lab
"""Bow + 3-point contact task action term.

Action Space (10-DoF residual):
  [0]    left_hip_pitch_joint     목표: -0.1 + (-20°) = -0.4491 rad
  [1]    right_hip_pitch_joint    목표: -0.1 + (-20°) = -0.4491 rad
  [2]    waist_pitch_joint        목표:  0.0 + (+15°) = +0.2618 rad
  [3-9]  왼팔 7-DoF (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
         nominal = init 자세 (팔을 앞으로 뻗는 방향은 RL이 학습)

비제어 관절:
  - 다리 (hip_roll/yaw, knee, ankle): default_joint_pos 고정
  - 오른팔: action 없음 (obs only, disturbance 역할)
  - waist yaw/roll: reward 패널티로 고정
"""
from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ── 제어 관절 정의 (10-DoF) ───────────────────────────────────────────────────
CTRL_JOINT_NAMES = [
    # 몸통 (3)
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_pitch_joint",
    # 왼팔 (7)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
NUM_CTRL = len(CTRL_JOINT_NAMES)  # 10

# ── nominal 목표 자세 ─────────────────────────────────────────────────────────
# UNITREE_G1_29DOF_CFG init_state.joint_pos 기준
# 몸통 3개만 nominal 목표값 정의 (왼팔은 nominal 없음)
_BODY_NOMINAL = {
    "left_hip_pitch_joint":  -0.1 + math.radians(-20.0),
    "right_hip_pitch_joint": -0.1 + math.radians(-20.0),
    "waist_pitch_joint":      0.0 + math.radians(15.0),
}
BODY_JOINT_NAMES = list(_BODY_NOMINAL.keys())  # 앞 3개
ARM_JOINT_NAMES  = CTRL_JOINT_NAMES[3:]        # 뒤 7개


class BowContactAction(ActionTerm):
    """10-DoF residual: 몸통 3 + 왼팔 7.

    비제어 관절(다리 고정, 오른팔)은 default_joint_pos 를 target 으로 설정.
    → actuator stiffness 가 init 자세를 유지하도록 작동.
    """
    cfg: "BowContactActionCfg"

    def __init__(self, cfg: "BowContactActionCfg", env: ManagerBasedRLEnv):
        self._asset    = env.scene[cfg.asset_name]
        all_names      = self._asset.joint_names
        self._ctrl_ids = torch.tensor(
            [all_names.index(n) for n in CTRL_JOINT_NAMES],
            dtype=torch.long, device=env.device,
        )
        super().__init__(cfg, env)

        # 몸통 nominal: 목표값 고정
        body_nominal = torch.tensor(
            [_BODY_NOMINAL[n] for n in BODY_JOINT_NAMES],
            dtype=torch.float32, device=env.device,
        )
        self._body_nominal = body_nominal.unsqueeze(0).expand(env.num_envs, -1).clone()

        # 왼팔 제어 관절 id (apply_actions 에서 default_joint_pos 슬라이싱용)
        self._arm_ids = torch.tensor(
            [all_names.index(n) for n in ARM_JOINT_NAMES],
            dtype=torch.long, device=env.device,
        )
        self._limit             = cfg.delta_q_limit
        self._raw_actions       = torch.zeros(env.num_envs, NUM_CTRL, device=env.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return NUM_CTRL  # 10

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions = actions.clone()
        # 몸통 3개(hip_pitch L/R, waist_pitch): ±0.15 rad (정밀 제어)
        # 왼팔 7개(shoulder~wrist): ±0.8 rad (넓은 탐색 범위)
        limits = torch.tensor(
            [0.15, 0.15, 0.15,          # hip_pitch L, R / waist_pitch
             0.8, 0.8, 0.8, 0.8,        # shoulder pitch/roll/yaw, elbow
             0.8, 0.8, 0.8],            # wrist roll/pitch/yaw
            device=actions.device,
        )
        self._processed_actions = actions.clamp(-limits, limits)

    def apply_actions(self) -> None:
        """
        몸통 3개 : target = body_nominal + Δq     (목표 자세로 수렴)
        왼팔 7개 : target = default_joint_pos + Δq (init 자세 기준 residual, nominal bias 없음)
        나머지   : target = default_joint_pos      (stiffness로 init 유지)
        """
        full_target = self._asset.data.default_joint_pos.clone()       # (N, n_joints)

        # 몸통: body_nominal + Δq[:3]
        body_delta  = self._processed_actions[:, :3]
        full_target[:, self._ctrl_ids[:3]] = self._body_nominal + body_delta

        # 왼팔: default_joint_pos(init) + Δq[3:]
        arm_delta   = self._processed_actions[:, 3:]
        arm_default = self._asset.data.default_joint_pos[:, self._arm_ids]
        full_target[:, self._arm_ids] = arm_default + arm_delta

        self._asset.set_joint_position_target(full_target)


@configclass
class BowContactActionCfg(ActionTermCfg):
    class_type: type     = BowContactAction
    asset_name: str      = "robot"
    delta_q_limit: float = 0.8   # 왼팔 탐색 범위 확보 (몸통은 0.15면 충분하나 통일)   # ±0.20 rad residual