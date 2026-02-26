# Copyright (c) 2022-2025, unitree_rl_lab
"""G1 Bow + 3-Point Contact Task.

요구사항:
  1. 다리 관절(hip_roll/yaw, knee, ankle) 좌우 동일 유지
  2. hip_pitch(L/R)  -0.1 + (-20°) = -0.4491 rad
  3. 왼팔 7-DoF: 테이블에 contact 유도 (action 포함)
  4. 오른팔: obs only (disturbance 역할)
  5. waist yaw/roll init 유지
  6. waist_pitch +15° 유지
  7. 두 발 + 왼팔 3-point contact 4초 이상 → 보상

Scene:
  - 테이블: 로봇 정면 TABLE_DISTANCE_X=0.3m, 0.4x0.4x0.8m
  - 왼손목 contact sensor (left_wrist_roll_link)

Action: 10-DoF  [hip_pitch L/R, waist_pitch, 왼팔 7]
Obs:    57-dim
"""
from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG
from unitree_rl_lab.tasks.g1_lean import mdp

# ── 테이블 치수 ───────────────────────────────────────────────────────────────
# 테이블: 로봇 바로 앞 (쓰러진 자세에서 팔이 닿을 수 있는 거리)
# torso 41° 전경 기준 어깨 x≈0.33m, arm reach≈0.43m → 테이블 앞면 0.3m가 적절
TABLE_DISTANCE_X = 0.5    # 로봇 정면 → 테이블 앞면 (m)
TABLE_WIDTH      = 0.3    # Y 방향 폭
TABLE_DEPTH      = 0.4    # X 방향 깊이
TABLE_HEIGHT     = 0.75   # 테이블 높이 (= 상면 z, 두께 없음)
# 테이블 center: x = TABLE_DISTANCE_X + TABLE_DEPTH/2, y = 0, z = TABLE_HEIGHT/2
TABLE_CENTER_X   = TABLE_DISTANCE_X + TABLE_DEPTH / 2.0   # 0.5 m
TABLE_CENTER_Z   = TABLE_HEIGHT / 2.0                      # 0.4 m

# ── Joint 그룹 ────────────────────────────────────────────────────────────────
HIP_PITCH_JOINT_NAMES = ["left_hip_pitch_joint", "right_hip_pitch_joint"]
WAIST_JOINT_NAMES     = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
WAIST_LATERAL_NAMES   = ["waist_yaw_joint", "waist_roll_joint"]

LEFT_LEG_FIXED_NAMES  = [
    "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
]
RIGHT_LEG_FIXED_NAMES = [
    "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
LEG_FIXED_NAMES = LEFT_LEG_FIXED_NAMES + RIGHT_LEG_FIXED_NAMES

LEFT_ARM_JOINT_NAMES  = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


##############################################################################
# Scene
##############################################################################

@configclass
class G1BowSceneCfg(InteractiveSceneCfg):

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # ── 로봇 ─────────────────────────────────────────────────────────────────
    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # ── 테이블 ───────────────────────────────────────────────────────────────
    # RigidObjectCfg: 물리적 충돌 + contact 가능한 고정 테이블
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(TABLE_DEPTH, TABLE_WIDTH, TABLE_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,       # 테이블은 고정
                kinematic_enabled=True,     # 위치 고정 (움직이지 않음)
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.4, 0.2),   # 나무 색
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(TABLE_CENTER_X, 0.2, TABLE_CENTER_Z),   # 0.5m, 0, 0.4m
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # ── Contact sensors ──────────────────────────────────────────────────────
    left_foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        history_length=3,
    )
    right_foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        history_length=3,
    )
    # 왼팔 contact: 어깨~손 전체, 테이블과의 contact 감지
    # left_shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw, rubber_hand 포함
    left_hand_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_.*_link",
        history_length=3,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
    )

    # 몸통 + 오른팔 contact: 테이블에 닿으면 패널티
    # torso/pelvis/waist + right arm 전체
    body_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        history_length=3,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )


##############################################################################
# MDP
##############################################################################

@configclass
class ActionsCfg:
    """10-DoF: hip_pitch L/R + waist_pitch + 왼팔 7."""
    bow: mdp.BowContactActionCfg = mdp.BowContactActionCfg(
        asset_name="robot",
        delta_q_limit=0.20,
    )


@configclass
class ObservationsCfg:
    """총 45-dim."""

    @configclass
    class PolicyCfg(ObsGroup):
        # ── 몸통 관절 (5+5 = 10) ─────────────────────────────────────────────
        # hip_pitch L/R + waist yaw/roll/pitch 를 하나의 term 으로 통합
        body_joint_pos = ObsTerm(
            func=mdp.body_joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HIP_PITCH_JOINT_NAMES + WAIST_JOINT_NAMES)},
            noise=GaussianNoise(mean=0.0, std=0.004),
        )
        body_joint_vel = ObsTerm(
            func=mdp.body_joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HIP_PITCH_JOINT_NAMES + WAIST_JOINT_NAMES)},
            noise=GaussianNoise(mean=0.0, std=0.02),
        )

        # ── Torso IMU (3+3 = 6) ──────────────────────────────────────────────
        # rot_mat(9) → gravity_vec(3) 으로 압축 (동일 정보, 기울기만 필요)
        torso_gravity = ObsTerm(
            func=mdp.torso_gravity_vec,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        torso_ang_vel_obs = ObsTerm(
            func=mdp.torso_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=GaussianNoise(mean=0.0, std=0.01),
        )

        # ── 목표 오차 (2+1 = 3) ──────────────────────────────────────────────
        hip_pitch_err = ObsTerm(
            func=mdp.hip_pitch_error,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HIP_PITCH_JOINT_NAMES)},
        )
        waist_pitch_err = ObsTerm(
            func=mdp.waist_pitch_error,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_pitch_joint"])},
        )

        # ── 왼팔 — action 대상 (7+7 = 14) ───────────────────────────────────
        left_arm_pos = ObsTerm(
            func=mdp.arm_joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINT_NAMES)},
            noise=GaussianNoise(mean=0.0, std=0.004),
        )
        left_arm_vel = ObsTerm(
            func=mdp.arm_joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINT_NAMES)},
            noise=GaussianNoise(mean=0.0, std=0.02),
        )

        # ── 왼손 contact + 테이블 방향 (1+3 = 4) ─────────────────────────────
        left_hand_contact_obs = ObsTerm(
            func=mdp.left_hand_contact_flag,
            params={"left_hand_cfg": SceneEntityCfg("left_hand_contact")},
        )
        left_hand_to_table = ObsTerm(
            func=mdp.left_hand_to_table_pos,
            params={
                "asset_cfg":           SceneEntityCfg("robot"),
                "left_hand_body_name": "left_wrist_roll_link",
                "table_top_z":         TABLE_HEIGHT,
            },
        )

        # ── 발 contact (1) ────────────────────────────────────────────────────
        foot_contact = ObsTerm(
            func=mdp.foot_contact_flag,
            params={
                "left_foot_cfg":  SceneEntityCfg("left_foot_contact"),
                "right_foot_cfg": SceneEntityCfg("right_foot_contact"),
            },
        )

        # ── 오른팔 — obs only, disturbance (7) ───────────────────────────────
        # 속도 제외: disturbance 역할에는 위치만으로 충분
        right_arm_pos = ObsTerm(
            func=mdp.arm_joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES)},
            noise=GaussianNoise(mean=0.0, std=0.004),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_robot = EventTerm(
        func=mdp.reset_robot_to_standing,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class RewardsCfg:
    # ── 몸통 목표 추종 (weight=양수) ─────────────────────────────────────────
    hip_pitch = RewTerm(
        func=mdp.hip_pitch_tracking,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=HIP_PITCH_JOINT_NAMES),
            "theta_d":   -0.1 + math.radians(-20.0),
        },
    )
    waist_pitch = RewTerm(
        func=mdp.waist_pitch_tracking,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["waist_pitch_joint"]),
            "theta_d":   math.radians(15.0),
        },
    )

    # ── 왼팔 뻗기 유도 (weight=양수) ────────────────────────────────────────
    # 1단계: shoulder_pitch를 앞으로 올리는 동작 자체에 보상
    left_shoulder_forward = RewTerm(
        func=mdp.left_shoulder_pitch_forward,
        weight=2.0,
        params={
            "asset_cfg":   SceneEntityCfg("robot", joint_names=["left_shoulder_pitch_joint"]),
            "init_pitch":  0.3,
        },
    )
    # 2단계: 손목이 torso CoM보다 앞에 있을수록 보상
    left_wrist_forward = RewTerm(
        func=mdp.left_wrist_forward_of_torso,
        weight=3.0,
        params={
            "asset_cfg":           SceneEntityCfg("robot"),
            "left_hand_body_name": "left_wrist_roll_link",
        },
    )
    # 3단계: 테이블 방향으로 가까워질수록 보상 (gradient 유지형)
    left_hand_reaching = RewTerm(
        func=mdp.left_hand_reaching_reward,
        weight=4.0,
        params={
            "asset_cfg":           SceneEntityCfg("robot"),
            "left_hand_body_name": "left_wrist_roll_link",
        },
    )
    # 4단계: 왼팔이 테이블에 닿으면 보상
    left_hand_contact = RewTerm(
        func=mdp.left_hand_contact_reward,
        weight=5.0,
        params={"left_hand_sensor_cfg": SceneEntityCfg("left_hand_contact")},
    )

    # ── 3-point contact 4초 유지 보상 (weight=양수) ───────────────────────────
    three_point_hold = RewTerm(
        func=mdp.three_point_contact_hold_reward,
        weight=5.0,
        params={
            "asset_cfg":             SceneEntityCfg("robot"),
            "left_foot_sensor_cfg":  SceneEntityCfg("left_foot_contact"),
            "right_foot_sensor_cfg": SceneEntityCfg("right_foot_contact"),
            "left_hand_sensor_cfg":  SceneEntityCfg("left_hand_contact"),
            "body_contact_sensor_cfg": SceneEntityCfg("body_contact"),
            "hold_time_sec":         4.0,
            "vel_thresh":            0.05,
            "force_thresh":          1.0,
        },
    )

    # ── 다리 고정 (weight=음수) ───────────────────────────────────────────────
    leg_symmetry = RewTerm(
        func=mdp.leg_symmetry_penalty,
        weight=-5.0,
        params={
            "left_cfg":  SceneEntityCfg("robot", joint_names=LEFT_LEG_FIXED_NAMES),
            "right_cfg": SceneEntityCfg("robot", joint_names=RIGHT_LEG_FIXED_NAMES),
        },
    )
    leg_fixed = RewTerm(
        func=mdp.leg_fixed_pose_penalty,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_FIXED_NAMES)},
    )

    # ── waist 측면 고정 (weight=음수) ────────────────────────────────────────
    waist_lateral = RewTerm(
        func=mdp.waist_lateral_penalty,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WAIST_LATERAL_NAMES)},
    )

    # ── 안정성 패널티 (weight=음수) ───────────────────────────────────────────
    torso_ang_vel = RewTerm(
        func=mdp.torso_ang_vel_penalty,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # 왼팔 제외: 왼팔 움직임을 억제하지 않도록
    joint_vel = RewTerm(
        func=mdp.joint_vel_penalty,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot",
            joint_names=HIP_PITCH_JOINT_NAMES + WAIST_JOINT_NAMES +
                        LEFT_LEG_FIXED_NAMES + RIGHT_LEG_FIXED_NAMES +
                        RIGHT_ARM_JOINT_NAMES)},
    )
    foot_slide = RewTerm(
        func=mdp.foot_slide_penalty,
        weight=-5.0,
        params={
            "asset_cfg":            SceneEntityCfg("robot"),
            "left_foot_body_name":  "left_ankle_roll_link",
            "right_foot_body_name": "right_ankle_roll_link",
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_table_contact = DoneTerm(
        func=mdp.body_table_contact,
        params={"body_contact_sensor_cfg": SceneEntityCfg("body_contact")},
    )
    fallen_forward = DoneTerm(
        func=mdp.torso_fallen_forward,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_deg": 65.0},
    )
    roll_exceeded = DoneTerm(
        func=mdp.torso_roll_exceeded,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_deg": 40.0},
    )
    too_low = DoneTerm(
        func=mdp.torso_too_low,
        params={"asset_cfg": SceneEntityCfg("robot"), "min_height": 0.3},
    )


##############################################################################
# Main Config
##############################################################################

@configclass
class G1BowEnvCfg(ManagerBasedRLEnvCfg):
    scene:        G1BowSceneCfg   = G1BowSceneCfg(num_envs=2048, env_spacing=5.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions:      ActionsCfg      = ActionsCfg()
    events:       EventCfg        = EventCfg()
    rewards:      RewardsCfg      = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation          = 4
        self.episode_length_s    = 10.0    # 4초 유지 조건 + 여유
        self.sim.dt              = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.viewer.eye    = (3.0, -2.0, 2.0)
        self.viewer.lookat = (1.0,  0.0, 0.8)

        self.scene.left_foot_contact.update_period  = self.sim.dt
        self.scene.right_foot_contact.update_period = self.sim.dt
        self.scene.left_hand_contact.update_period  = self.sim.dt
        self.scene.body_contact.update_period      = self.sim.dt


@configclass
class G1BowEnvCfgPLAY(G1BowEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs                        = 16
        self.observations.policy.enable_corruption = False