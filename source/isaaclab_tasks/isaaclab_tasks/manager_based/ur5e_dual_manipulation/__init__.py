# Copyright (c) Till Laube 
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dual UR5e manipulation environment for GR00T teleoperation data collection."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Ur5e-Dual-Manipulation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5e_dual_manipulation_env_cfg:Ur5eDualManipulationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# Alias for convenience
gym.register(
    id="Isaac-Ur5e-Dual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5e_dual_manipulation_env_cfg:Ur5eDualManipulationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

