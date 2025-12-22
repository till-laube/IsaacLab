# Copyright (c) Till Laube
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

##
# Single UR5e Arm with Robotiq 2F-140 Gripper - Teleoperation/Manipulation Task
##
gym.register(
    id="Isaac-Ur5e-Single-v0",
    entry_point=f"{__name__}.ur5e_single_manipulation_env:Ur5eSingleManipulationEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5e_single_manipulation_env_cfg:Ur5eSingleManipulationEnvCfg",
    },
    disable_env_checker=True,
)
