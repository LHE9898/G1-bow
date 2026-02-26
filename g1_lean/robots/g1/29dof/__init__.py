# Copyright (c) 2022-2025, unitree_rl_lab
"""G1 Bow Task: hip_pitch 15° 숙이기 → 쓰러짐 확인."""
import gymnasium as gym

gym.register(
    id="Unitree-G1-Bow-v0",
    entry_point=f"{__name__}.bow_env_cfg:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bow_env_cfg:G1BowEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.g1_lean.agents.rsl_rl_ppo_cfg:G1BowPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-Bow-Play-v0",
    entry_point=f"{__name__}.bow_env_cfg:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "play_env_cfg_entry_point": f"{__name__}.bow_env_cfg:G1BowEnvCfgPLAY",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.g1_lean.agents.rsl_rl_ppo_cfg:G1BowPPORunnerCfg",
    },
)