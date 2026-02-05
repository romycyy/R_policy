# Copyright (c) 2018-2023, NVIDIA Corporation
# HumanoidGPT: Humanoid task with dof_force / sensor_force_torques for GPT-style reward API.
# Subclasses Humanoid and adds aliases expected by compute_reward(potentials, ..., dof_force, sensor_force_torques, ...).

import torch
from typing import Dict, Tuple

from isaacgymenvs.utils.torch_jit_utils import *
from .humanoid import Humanoid


class HumanoidGPT(Humanoid):
    """Humanoid task with dof_force and sensor_force_torques attributes for reward APIs that expect them."""

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        # Aliases expected by GPT-style compute_reward(potentials, ..., self.dof_force, self.sensor_force_torques, ...)
        self.dof_force = self.dof_force_tensor
        self.sensor_force_torques = self.vec_sensor_tensor
        self.rew_dict = {}

    def compute_reward(self, actions):
        # dof_force / sensor_force_torques already refreshed in compute_observations() before this
        self.rew_buf[:], self.rew_dict = compute_reward_gpt(
            self.potentials,
            self.prev_potentials,
            self.root_states,
            self.actions,
            self.dof_vel,
            self.dof_pos,
            self.dof_force,
            self.sensor_force_torques,
            self.up_vec,
            self.heading_vec,
            self.dt,
        )
        for k, v in self.rew_dict.items():
            self.extras[k] = v.mean()


@torch.jit.script
def compute_reward_gpt(
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    root_states: torch.Tensor,
    actions: torch.Tensor,
    dof_vel: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_force: torch.Tensor,
    sensor_force_torques: torch.Tensor,
    up_vec: torch.Tensor,
    heading_vec: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Dict[str, Tensor]]
    progress_reward = potentials - prev_potentials
    rew_buf = progress_reward
    rew_dict = {"progress": progress_reward}
    return rew_buf, rew_dict
