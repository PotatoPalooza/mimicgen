# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import warp as wp

import os
from copy import deepcopy

import numpy as np
import robosuite.utils.transform_utils as T
import torch
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.binding_utils import MjSimWarp
from robosuite.utils.mjcf_utils import CustomMaterial, add_material, find_elements, string_to_array
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

import mimicgen
from mimicgen.envs.robosuite.single_arm_env_mg import SingleArmEnv_MG
from mimicgen.models.robosuite.objects import (
    BlenderObject,
    CoffeeMachineObject,
    CoffeeMachinePodObject,
    LongDrawerObject,
)


def _lnorm(x, dim=-1):
    """L2 norm that works for both numpy arrays and torch tensors."""
    if isinstance(x, torch.Tensor):
        return torch.linalg.norm(x, dim=dim)
    return np.linalg.norm(x, axis=dim)


class Coffee(SingleArmEnv_MG):
    """
    This class corresponds to the coffee task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        use_warp: bool = False,
        num_envs: int = 1,
        difficulty: float = 0.0,
        placement_bounds_pad: float = 0.05,
        full_rotation_randomization: bool = True,
        tipover_prob: float = 0.0,
        tipover_angle_range: tuple[float, float] = (np.pi / 3.0, np.pi / 2.0),
        tipover_z_bump: float = 0.03,
        fall_off_termination: bool = False,
        fall_off_z_margin: float = 0.1,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # Randomization curriculum: d in [0, 1] scales tipover prob + bound padding
        # and interpolates z_rot toward full (-pi, pi). d=0 matches BC distribution.
        self._difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self.placement_bounds_pad = placement_bounds_pad
        self.full_rotation_randomization = full_rotation_randomization
        self.tipover_prob = tipover_prob
        self.tipover_angle_range = tipover_angle_range
        self.tipover_z_bump = tipover_z_bump

        # Early-termination on tracked objects dropping below
        # ``table_offset[2] - fall_off_z_margin``. Feeds ``_check_early_termination``.
        self.fall_off_termination = fall_off_termination
        self.fall_off_z_margin = fall_off_z_margin

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            use_warp=use_warp,
            num_envs=num_envs,
        )

    def reward(self, action: np.ndarray | wp.array | None = None) -> float:
        """
        Reward function for the task.

        The sparse reward only consists of the threading component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.0 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        success = self._check_success()

        if isinstance(success, torch.Tensor):
            # Warp: return per-env reward tensor
            reward = success.float()
            if self.reward_scale is not None:
                reward = reward * self.reward_scale
            return reward

        reward = 1.0 if success else 0.0
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Add camera with full tabletop perspective
        self._add_agentview_full_camera(mujoco_arena)

        # initialize objects of interest
        self.coffee_pod = CoffeeMachinePodObject(name="coffee_pod")
        self.coffee_machine = CoffeeMachineObject(name="coffee_machine")
        objects = [self.coffee_pod, self.coffee_machine]

        # Create placement initializer
        self._get_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objects,
        )

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects (e.g.
        what happens when env.reset is called). Should return a dictionary with the following
        structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not change)
        """
        return dict(
            coffee_machine=dict(
                x=(0.0, 0.0),
                y=(-0.1, -0.1),
                z_rot=(-np.pi / 6.0, -np.pi / 6.0),
                reference=self.table_offset,
            ),
            coffee_pod=dict(
                x=(-0.13, -0.07),
                y=(0.17, 0.23),
                z_rot=(0.0, 0.0),
                reference=self.table_offset,
            ),
        )

    def _padded_objects(self) -> tuple[str, ...]:
        return ("coffee_pod", "coffee_machine")

    def _full_rotation_objects(self) -> tuple[str, ...]:
        return ("coffee_pod",)

    def _tippable_objects(self) -> tuple[str, ...]:
        return ("coffee_pod",)

    @property
    def difficulty(self) -> float:
        return self._difficulty

    def set_difficulty(self, difficulty: float) -> None:
        """Update the randomization curriculum level.

        Called from the RL training loop to ramp the reset distribution
        from BC-matched (d=0) up to full extra randomization (d=1).
        Takes effect on the next reset -- live rollouts are unaffected.
        The placement initializer is rebuilt so that each sub-sampler's
        x/y/z_rot ranges reflect the new difficulty (tipover uses the
        live value directly and needs no rebuild).
        """
        self._difficulty = float(np.clip(difficulty, 0.0, 1.0))
        if getattr(self, "placement_initializer", None) is not None:
            self._get_placement_initializer()

    def _maybe_apply_extra_randomization(self, bounds: dict[str, dict]) -> dict[str, dict]:
        d = self._difficulty
        if d <= 0.0:
            return bounds
        pad = self.placement_bounds_pad * d
        padded = set(self._padded_objects())
        full_rot = set(self._full_rotation_objects()) if self.full_rotation_randomization else set()
        out: dict[str, dict] = {}
        for name, b in bounds.items():
            nb = dict(b)
            if name in padded:
                nb["x"] = (b["x"][0] - pad, b["x"][1] + pad)
                nb["y"] = (b["y"][0] - pad, b["y"][1] + pad)
            if name in full_rot:
                lo, hi = b["z_rot"]
                nb["z_rot"] = (lo + d * (-np.pi - lo), hi + d * (np.pi - hi))
            out[name] = nb
        return out

    def _effective_tipover_prob(self) -> float:
        return float(self.tipover_prob * self._difficulty)

    def _maybe_tip_placement(
        self, obj, obj_pos: np.ndarray, obj_quat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # quats are (w, x, y, z) to match placement sampler + MuJoCo free-joint qpos order
        prob = self._effective_tipover_prob()
        if prob <= 0.0 or obj.name not in self._tippable_objects():
            return obj_pos, obj_quat
        if np.random.rand() >= prob:
            return obj_pos, obj_quat

        tip_angle = np.random.uniform(*self.tipover_angle_range)
        axis_angle = np.random.uniform(0.0, 2.0 * np.pi)
        axis = np.array([np.cos(axis_angle), np.sin(axis_angle), 0.0])
        tip_xyzw = T.axisangle2quat(axis * tip_angle)

        base_xyzw = T.convert_quat(np.asarray(obj_quat), to="xyzw")
        new_xyzw = T.quat_multiply(tip_xyzw, base_xyzw)
        new_wxyz = T.convert_quat(new_xyzw, to="wxyz")

        new_pos = np.array(obj_pos, dtype=np.float64)
        new_pos[2] += self.tipover_z_bump
        return new_pos, new_wxyz

    def _maybe_tip_placement_batch(
        self, obj, pos: np.ndarray, quat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised variant of :meth:`_maybe_tip_placement` for
        ``(n, 3)`` ``pos`` + ``(n, 4)`` ``quat`` batches (wxyz). Only the rows
        that "win" the per-env Bernoulli coin flip get tipped; others pass
        through unchanged. Returns fresh arrays.
        """
        prob = self._effective_tipover_prob()
        if prob <= 0.0 or obj.name not in self._tippable_objects():
            return pos, quat

        n = pos.shape[0]
        tip_mask = np.random.rand(n) < prob
        k = int(tip_mask.sum())
        if k == 0:
            return pos, quat

        tip_angle = np.random.uniform(
            low=self.tipover_angle_range[0],
            high=self.tipover_angle_range[1],
            size=k,
        )
        axis_angle = np.random.uniform(0.0, 2.0 * np.pi, size=k)
        # axisangle2quat(axis * angle) with axis = (cos(a), sin(a), 0) reduces
        # to tip_xyzw = (cos(a) sin(angle/2), sin(a) sin(angle/2), 0, cos(angle/2)).
        s = np.sin(tip_angle / 2.0)
        c = np.cos(tip_angle / 2.0)
        tip_xyzw = np.stack(
            [np.cos(axis_angle) * s, np.sin(axis_angle) * s, np.zeros(k), c],
            axis=-1,
        )

        base_wxyz = quat[tip_mask]
        # Convert wxyz -> xyzw: cols (1,2,3,0).
        base_xyzw = base_wxyz[:, [1, 2, 3, 0]]
        from robosuite.utils.placement_samplers import _quat_multiply_batch

        new_xyzw = _quat_multiply_batch(tip_xyzw, base_xyzw)
        new_wxyz = new_xyzw[:, [3, 0, 1, 2]]

        new_quat = quat.copy()
        new_pos = pos.copy()
        new_quat[tip_mask] = new_wxyz
        new_pos[tip_mask, 2] += self.tipover_z_bump
        return new_pos, new_quat

    def _get_placement_initializer(self):
        bounds = self._maybe_apply_extra_randomization(self._get_initial_placement_bounds())

        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CoffeeMachineSampler",
                mujoco_objects=self.coffee_machine,
                x_range=bounds["coffee_machine"]["x"],
                y_range=bounds["coffee_machine"]["y"],
                rotation=bounds["coffee_machine"]["z_rot"],
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["coffee_machine"]["reference"],
                z_offset=0.0,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CoffeePodSampler",
                mujoco_objects=self.coffee_pod,
                x_range=bounds["coffee_pod"]["x"],
                y_range=bounds["coffee_pod"]["y"],
                rotation=bounds["coffee_pod"]["z_rot"],
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["coffee_pod"]["reference"],
                z_offset=0.0,
            )
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references for this env
        self.obj_body_id = dict(
            coffee_pod=self.sim.model.body_name2id(self.coffee_pod.root_body),
            coffee_machine=self.sim.model.body_name2id(self.coffee_machine.root_body),
            coffee_pod_holder=self.sim.model.body_name2id("coffee_machine_pod_holder_root"),
            coffee_machine_lid=self.sim.model.body_name2id("coffee_machine_lid_main"),
        )
        self.hinge_qpos_addr = self.sim.model.get_joint_qpos_addr("coffee_machine_lid_main_joint0")

        # for checking contact (used in reward function, and potentially observation space)
        self.pod_geom_id = self.sim.model.geom_name2id("coffee_pod_g0")
        self.lid_geom_id = self.sim.model.geom_name2id("coffee_machine_lid_g0")
        pod_holder_geom_names = ["coffee_machine_pod_holder_cup_body_hc_{}".format(i) for i in range(64)]
        self.pod_holder_geom_ids = [self.sim.model.geom_name2id(x) for x in pod_holder_geom_names]

        # Gripper fingerpad geom ids, cached for warp contact queries. Fail
        # loudly at setup if the gripper doesn't expose the expected groups.
        gripper = self.robots[0].gripper
        self.pod_contact_geom_ids = [self.sim.model.geom_name2id(g) for g in self.coffee_pod.contact_geoms]
        self.left_fingerpad_geom_ids = [
            self.sim.model.geom_name2id(g) for g in gripper.important_geoms["left_fingerpad"]
        ]
        self.right_fingerpad_geom_ids = [
            self.sim.model.geom_name2id(g) for g in gripper.important_geoms["right_fingerpad"]
        ]

        # size of bounding box for pod holder
        self.pod_holder_size = self.coffee_machine.pod_holder_size

        # size of bounding box for pod
        self.pod_size = self.coffee_pod.get_bounding_box_half_size()

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            if self.use_warp:
                import warp as wp
                from robosuite.utils.binding_utils import MjSimWarp

                assert isinstance(self.sim, MjSimWarp)

                # Masked per-env placement: unmasked rows kept untouched;
                # full-width writes via set_joint_qpos would clobber kept envs.
                mask = getattr(self, "_reset_env_mask", None)
                if mask is None:
                    sample_idxs_arr = np.arange(self.num_envs)
                elif isinstance(mask, torch.Tensor):
                    sample_idxs_arr = mask.nonzero().flatten().cpu().numpy()
                else:
                    sample_idxs_arr = np.asarray(mask).nonzero()[0]

                if sample_idxs_arr.size > 0:
                    k = int(sample_idxs_arr.size)
                    placements = self.placement_initializer.sample_batch(k)
                    qpos_t = wp.to_torch(self.sim._warp_data.qpos)
                    row_idx = torch.as_tensor(
                        sample_idxs_arr, device=qpos_t.device, dtype=torch.long
                    )
                    for obj_pos, obj_quat, obj in placements.values():
                        obj_pos, obj_quat = self._maybe_tip_placement_batch(obj, obj_pos, obj_quat)
                        addr = self.sim.model.get_joint_qpos_addr(obj.joints[0])
                        start, end = addr if isinstance(addr, tuple) else (addr, addr + 1)
                        stacked = np.concatenate(
                            [obj_pos.astype(np.float32), obj_quat.astype(np.float32)], axis=-1
                        )  # (k, 7)
                        qpos_t[row_idx, start:end] = torch.as_tensor(
                            stacked, device=qpos_t.device, dtype=torch.float32
                        )
            else:
                # Sample from the placement initializer for all objects
                object_placements = self.placement_initializer.sample()

                # Loop through all objects and reset their positions
                for obj_pos, obj_quat, obj in object_placements.values():
                    from robosuite.utils.binding_utils import MjSimWarp

                    assert self.sim is not None and not isinstance(self.sim, MjSimWarp)
                    obj_pos, obj_quat = self._maybe_tip_placement(obj, obj_pos, obj_quat)
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Masked hinge reset: unmasked rows kept to preserve in-progress lid angles.
        from robosuite.utils.binding_utils import MjSimWarp

        if isinstance(self.sim, MjSimWarp):
            import warp as wp
            qpos_t = wp.to_torch(self.sim._warp_data.qpos)
            mask = getattr(self, "_reset_env_mask", None)
            if mask is None:
                qpos_t[:, self.hinge_qpos_addr] = 2.0 * np.pi / 3.0
            else:
                row_idx = (mask.nonzero().flatten() if isinstance(mask, torch.Tensor)
                           else torch.as_tensor(np.asarray(mask).nonzero()[0],
                                                device=qpos_t.device, dtype=torch.long))
                qpos_t[row_idx, self.hinge_qpos_addr] = 2.0 * np.pi / 3.0
        else:
            self.sim.data.qpos[self.hinge_qpos_addr] = 2.0 * np.pi / 3.0
        self.sim.forward()

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # for conversion to relative gripper frame
            @sensor(modality=modality)
            def world_pose_in_gripper(obs_cache):
                if f"{pf}eef_pos" not in obs_cache or f"{pf}eef_quat" not in obs_cache:
                    return np.eye(4)
                eef_pos = obs_cache[f"{pf}eef_pos"]
                eef_quat = obs_cache[f"{pf}eef_quat"]
                if isinstance(self.sim, MjSimWarp):
                    return T.pose_inv_torch(T.pose2mat_torch(eef_pos, eef_quat))  # (num_envs, 4, 4)
                return T.pose_inv(T.pose2mat((eef_pos, eef_quat)))

            sensors = [world_pose_in_gripper]
            names = ["world_pose_in_gripper"]
            actives = [False]

            @sensor(modality=modality)
            def eef_control_frame_pose(obs_cache):
                if f"{pf}eef_pos" not in obs_cache or f"{pf}eef_quat" not in obs_cache:
                    return np.eye(4)
                eef_name = self.robots[0].controller.eef_name
                if isinstance(self.sim, MjSimWarp):
                    sid = self.sim.model.site_name2id(eef_name)
                    return T.make_pose_torch(
                        self.sim.data.site_xpos[sid], self.sim.data.site_xmat[sid]
                    )  # (num_envs, 4, 4)
                return T.make_pose(
                    np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(eef_name)]),
                    np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(eef_name)].reshape([3, 3])),
                )

            sensors += [eef_control_frame_pose]
            names += ["eef_control_frame_pose"]
            actives += [False]

            # add ground-truth poses (absolute and relative to eef) for all objects
            for obj_name in self.obj_body_id:
                obj_sensors, obj_sensor_names = self._create_obj_sensors(obj_name=obj_name, modality=modality)
                sensors += obj_sensors
                names += obj_sensor_names
                actives += [True] * len(obj_sensors)

            obj_centric_sensors, obj_centric_sensor_names = self._create_obj_centric_sensors(modality="object_centric")
            sensors += obj_centric_sensors
            names += obj_centric_sensor_names
            actives += [True] * len(obj_centric_sensors)

            # add hinge angle of lid
            @sensor(modality=modality)
            def hinge_angle(obs_cache):
                if isinstance(self.sim, MjSimWarp):
                    return self.sim.data.qpos[[self.hinge_qpos_addr]]  # (num_envs, 1) torch.Tensor
                return np.array([self.sim.data.qpos[self.hinge_qpos_addr]])

            sensors += [hinge_angle]
            names += ["hinge_angle"]
            actives += [True]

            # Create observables
            for name, s, active in zip(names, sensors, actives):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    active=active,
                )

        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """

        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            bid = self.obj_body_id[obj_name]
            if isinstance(self.sim, MjSimWarp):
                return self.sim.data.body_xpos[bid]  # (num_envs, 3) torch.Tensor
            return np.array(self.sim.data.body_xpos[bid])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            bid = self.obj_body_id[obj_name]
            if isinstance(self.sim, MjSimWarp):
                q = self.sim.data.body_xquat[bid]  # (num_envs, 4) wxyz torch.Tensor
                return q[:, [1, 2, 3, 0]]  # (num_envs, 4) xyzw torch.Tensor
            return T.convert_quat(self.sim.data.body_xquat[bid], to="xyzw")

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            # Immediately return default value if cache is empty
            if any(
                [name not in obs_cache for name in [f"{obj_name}_pos", f"{obj_name}_quat", "world_pose_in_gripper"]]
            ):
                return np.zeros(3)
            if isinstance(self.sim, MjSimWarp):
                obj_pos = obs_cache[f"{obj_name}_pos"]  # (num_envs, 3) tensor
                obj_quat = obs_cache[f"{obj_name}_quat"]  # (num_envs, 4) tensor xyzw
                world_poses = obs_cache["world_pose_in_gripper"]  # (num_envs, 4, 4) tensor
                obj_pose = T.pose2mat_torch(obj_pos, obj_quat)  # (num_envs, 4, 4)
                rel_pose = world_poses @ obj_pose  # (num_envs, 4, 4)
                obs_cache[f"{obj_name}_to_{pf}eef_quat"] = T.mat2quat_torch(rel_pose[:, :3, :3])
                obs_cache[f"{obj_name}_pose"] = obj_pose
                return rel_pose[:, :3, 3]  # (num_envs, 3)
            obj_pose = T.pose2mat((obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"]))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            obs_cache[f"{obj_name}_pose"] = obj_pose
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            key = f"{obj_name}_to_{pf}eef_quat"
            if key in obs_cache:
                return obs_cache[key]
            if isinstance(self.sim, MjSimWarp):
                return torch.zeros(self.sim.num_envs, 4, device="cuda")
            return np.zeros(4)

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [f"{obj_name}_pos", f"{obj_name}_quat", f"{obj_name}_to_{pf}eef_pos", f"{obj_name}_to_{pf}eef_quat"]

        return sensors, names

    def _create_obj_centric_sensors(self, modality="object_centric"):
        """
        Creates sensors for poses relative to certain objects. This is abstracted in a separate
        function call so that we don't have local function naming collisions during
        the _setup_observables() call.

        Args:
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """
        sensors = []
        names = []
        pf = self.robots[0].robot_model.naming_prefix

        # helper function for relative position sensors, to avoid code duplication
        def _pos_helper(obs_cache, obs_name, ref_name, quat_cache_name):
            # Immediately return default value if cache is empty
            if any([name not in obs_cache for name in [obs_name, ref_name]]):
                if isinstance(self.sim, MjSimWarp):
                    return torch.zeros(self.sim.num_envs, 3, device="cuda")
                return np.zeros(3)
            ref_pose = obs_cache[ref_name]
            obs_pose = obs_cache[obs_name]
            if isinstance(ref_pose, torch.Tensor):
                rel_pose = T.pose_inv_torch(ref_pose) @ obs_pose  # (num_envs, 4, 4)
                obs_cache[quat_cache_name] = T.mat2quat_torch(rel_pose[:, :3, :3])
                return rel_pose[:, :3, 3]
            rel_pose = T.pose_in_A_to_pose_in_B(obs_pose, T.pose_inv(ref_pose))
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[quat_cache_name] = rel_quat
            return rel_pos

        # helper function for relative quaternion sensors, to avoid code duplication
        def _quat_helper(obs_cache, quat_cache_name):
            if quat_cache_name in obs_cache:
                return obs_cache[quat_cache_name]
            if isinstance(self.sim, MjSimWarp):
                return torch.zeros(self.sim.num_envs, 4, device="cuda")
            return np.zeros(4)

        # eef pose relative to ref object frames
        @sensor(modality=modality)
        def eef_pos_rel_pod(obs_cache):
            return _pos_helper(
                obs_cache=obs_cache,
                obs_name="eef_control_frame_pose",
                ref_name="coffee_pod_pose",
                quat_cache_name="eef_quat_rel_pod",
            )

        @sensor(modality=modality)
        def eef_quat_rel_pod(obs_cache):
            return _quat_helper(
                obs_cache=obs_cache,
                quat_cache_name="eef_quat_rel_pod",
            )

        sensors += [eef_pos_rel_pod, eef_quat_rel_pod]
        names += [f"{pf}eef_pos_rel_pod", f"{pf}eef_quat_rel_pod"]

        @sensor(modality=modality)
        def eef_pos_rel_pod_holder(obs_cache):
            return _pos_helper(
                obs_cache=obs_cache,
                obs_name="eef_control_frame_pose",
                ref_name="coffee_pod_holder_pose",
                quat_cache_name="eef_quat_rel_pod_holder",
            )

        @sensor(modality=modality)
        def eef_quat_rel_pod_holder(obs_cache):
            return _quat_helper(
                obs_cache=obs_cache,
                quat_cache_name="eef_quat_rel_pod_holder",
            )

        sensors += [eef_pos_rel_pod_holder, eef_quat_rel_pod_holder]
        names += [f"{pf}eef_pos_rel_pod_holder", f"{pf}eef_quat_rel_pod_holder"]

        # obj pose relative to ref object frame
        @sensor(modality=modality)
        def pod_pos_rel_pod_holder(obs_cache):
            return _pos_helper(
                obs_cache=obs_cache,
                obs_name="coffee_pod_pose",
                ref_name="coffee_pod_holder_pose",
                quat_cache_name="pod_quat_rel_pod_holder",
            )

        @sensor(modality=modality)
        def pod_quat_rel_pod_holder(obs_cache):
            return _quat_helper(
                obs_cache=obs_cache,
                quat_cache_name="pod_quat_rel_pod_holder",
            )

        sensors += [pod_pos_rel_pod_holder, pod_quat_rel_pod_holder]
        names += ["pod_pos_rel_pod_holder", "pod_quat_rel_pod_holder"]

        return sensors, names

    def _get_body_pos(self, body_id: int) -> np.ndarray | torch.Tensor:
        """Body pos: (3,) numpy CPU / (num_envs, 3) tensor warp."""
        return self.sim.data.body_xpos[body_id]

    def _get_hinge_angle(self) -> float | np.ndarray | torch.Tensor:
        """Hinge angle: scalar CPU / (num_envs,) tensor warp."""
        return self.sim.data.qpos[self.hinge_qpos_addr]

    def _check_success(self):
        """
        Check if task is complete.
        """
        metrics = self._get_partial_task_metrics()
        return metrics["task"]

    def _fall_off_tracked_objects(self) -> tuple[str, ...]:
        """Objects whose body COM z is monitored by the fall-off check."""
        return ("coffee_pod", "coffee_machine")

    def _check_early_termination(self) -> dict[str, object]:
        """Adds ``fell_off_<obj>`` cause per tracked object below table threshold; gated by fall_off_termination."""
        extras = super()._check_early_termination()
        if not self.fall_off_termination:
            return extras
        threshold = float(self.table_offset[2]) - float(self.fall_off_z_margin)
        for obj_name in self._fall_off_tracked_objects():
            pos = self._get_body_pos(self.obj_body_id[obj_name])  # (..., 3)
            extras[f"fell_off_{obj_name}"] = pos[..., 2] < threshold
        return extras

    def _check_lid(self):
        hinge_tolerance = 15.0 * np.pi / 180.0
        return self._get_hinge_angle() < hinge_tolerance

    def _check_pod(self):
        pod_holder_pos = self._get_body_pos(self.obj_body_id["coffee_pod_holder"])  # (..., 3)
        pod_pos = self._get_body_pos(self.obj_body_id["coffee_pod"])  # (..., 3)
        lid_pos = self._get_body_pos(self.obj_body_id["coffee_machine_lid"])  # (..., 3)

        r_diff = self.pod_holder_size[0] - self.pod_size[0]
        pod_horz_check = _lnorm(pod_pos[..., :2] - pod_holder_pos[..., :2]) <= r_diff

        z_lim_low = pod_holder_pos[..., 2] - self.pod_holder_size[2]
        z_lim_high = lid_pos[..., 2] - self.coffee_machine.lid_size[2]
        pod_z_check = (pod_pos[..., 2] - self.pod_size[2] >= z_lim_low) & (
            pod_pos[..., 2] + self.pod_size[2] <= z_lim_high
        )
        return pod_horz_check & pod_z_check

    def _get_partial_task_metrics(self):
        metrics = dict()

        lid_check = self._check_lid()
        pod_check = self._check_pod()

        pod_holder_pos = self._get_body_pos(self.obj_body_id["coffee_pod_holder"])  # (..., 3)
        pod_pos = self._get_body_pos(self.obj_body_id["coffee_pod"])  # (..., 3)

        r_diff = self.pod_holder_size[0] - self.pod_size[0]
        pod_horz_check = _lnorm(pod_pos[..., :2] - pod_holder_pos[..., :2]) <= r_diff

        z_lim_low = pod_holder_pos[..., 2] - self.pod_holder_size[2]

        metrics["task"] = lid_check & pod_check

        pod_insertion_z_tolerance = 0.02
        pod_z_check = (pod_pos[..., 2] - self.pod_size[2] > z_lim_low) & (
            pod_pos[..., 2] - self.pod_size[2] < z_lim_low + pod_insertion_z_tolerance
        )
        metrics["insertion"] = pod_horz_check & pod_z_check

        metrics["grasp"] = self._check_pod_is_grasped()

        rim_horz_tolerance = 0.03
        rim_horz_check = _lnorm(pod_pos[..., :2] - pod_holder_pos[..., :2]) < rim_horz_tolerance

        rim_vert_tolerance = 0.026
        rim_vert_length = pod_pos[..., 2] - pod_holder_pos[..., 2] - self.pod_holder_size[2]
        rim_vert_check = (rim_vert_length < rim_vert_tolerance) & (rim_vert_length > 0.0)
        metrics["rim"] = rim_horz_check & rim_vert_check

        return metrics

    def _check_pod_is_grasped(self):
        """
        check if pod is grasped by robot (contact on both fingerpads).
        """
        if isinstance(self.sim, MjSimWarp):
            left_hit = self.sim.check_contact_groups(self.left_fingerpad_geom_ids, self.pod_contact_geom_ids)
            right_hit = self.sim.check_contact_groups(self.right_fingerpad_geom_ids, self.pod_contact_geom_ids)
            return left_hit & right_hit
        return self._check_grasp(
            gripper=self.robots[0].gripper, object_geoms=[g for g in self.coffee_pod.contact_geoms]
        )

    def _check_pod_and_pod_holder_contact(self):
        """
        check if pod is in contact with the container.
        """
        if isinstance(self.sim, MjSimWarp):
            return self.sim.check_contact_groups([self.pod_geom_id], self.pod_holder_geom_ids)
        pod_and_pod_holder_contact = False
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if ((contact.geom1 == self.pod_geom_id) and (contact.geom2 in self.pod_holder_geom_ids)) or (
                (contact.geom2 == self.pod_geom_id) and (contact.geom1 in self.pod_holder_geom_ids)
            ):
                pod_and_pod_holder_contact = True
                break
        return pod_and_pod_holder_contact

    def _check_pod_on_rim(self):
        pod_holder_pos = self._get_body_pos(self.obj_body_id["coffee_pod_holder"])
        pod_pos = self._get_body_pos(self.obj_body_id["coffee_pod"])

        pod_and_pod_holder_contact = self._check_pod_and_pod_holder_contact()

        rim_vert_tolerance_1 = 0.022
        rim_vert_tolerance_2 = 0.026
        rim_vert_length = pod_pos[..., 2] - pod_holder_pos[..., 2] - self.pod_holder_size[2]
        rim_vert_check = (rim_vert_length > rim_vert_tolerance_1) & (rim_vert_length < rim_vert_tolerance_2)

        return pod_and_pod_holder_contact & rim_vert_check

    def _check_pod_being_inserted(self):
        pod_holder_pos = self._get_body_pos(self.obj_body_id["coffee_pod_holder"])
        pod_pos = self._get_body_pos(self.obj_body_id["coffee_pod"])

        rim_horz_tolerance = 0.005
        rim_horz_check = _lnorm(pod_pos[..., :2] - pod_holder_pos[..., :2]) < rim_horz_tolerance

        rim_vert_tolerance_1 = -0.01
        rim_vert_tolerance_2 = 0.023
        rim_vert_length = pod_pos[..., 2] - pod_holder_pos[..., 2] - self.pod_holder_size[2]
        rim_vert_check = (rim_vert_length < rim_vert_tolerance_2) & (rim_vert_length > rim_vert_tolerance_1)

        return rim_horz_check & rim_vert_check

    def _check_pod_inserted(self):
        pod_holder_pos = self._get_body_pos(self.obj_body_id["coffee_pod_holder"])
        pod_pos = self._get_body_pos(self.obj_body_id["coffee_pod"])

        r_diff = self.pod_holder_size[0] - self.pod_size[0]
        pod_horz_check = _lnorm(pod_pos[..., :2] - pod_holder_pos[..., :2]) <= r_diff

        pod_insertion_z_tolerance = 0.02
        z_lim_low = pod_holder_pos[..., 2] - self.pod_holder_size[2]
        pod_z_check = (pod_pos[..., 2] - self.pod_size[2] > z_lim_low) & (
            pod_pos[..., 2] - self.pod_size[2] < z_lim_low + pod_insertion_z_tolerance
        )
        return pod_horz_check & pod_z_check

    def _check_lid_being_closed(self):
        return self._get_hinge_angle() < 2.09

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the coffee machine.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the coffee machine
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.coffee_machine)


class Coffee_D0(Coffee):
    """Rename base class for convenience."""

    pass


class Coffee_D1(Coffee_D0):
    """
    Wider initialization for pod and coffee machine.
    """

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects (e.g.
        what happens when env.reset is called). Should return a dictionary with the following
        structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not change)
        """
        return dict(
            coffee_machine=dict(
                x=(0.05, 0.15),
                y=(-0.2, -0.1),
                z_rot=(-np.pi / 6.0, np.pi / 3.0),
                reference=self.table_offset,
            ),
            coffee_pod=dict(
                # x=(-0.2, -0.2),
                x=(-0.2, 0.05),
                # x=(-0.13, -0.07),
                y=(0.17, 0.3),
                # y=(0.3, 0.3),
                # y=(0.1, 0.1),
                # y=(0.17, 0.23),
                z_rot=(0.0, 0.0),
                reference=self.table_offset,
            ),
        )


class Coffee_D2(Coffee_D1):
    """
    Similar to Coffee_D1, but put pod on the left, and machine on the right. Had to also move
    machine closer to robot (in x) to get kinematics to work out.
    """

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects (e.g.
        what happens when env.reset is called). Should return a dictionary with the following
        structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not change)
        """
        return dict(
            coffee_machine=dict(
                x=(-0.05, 0.05),
                y=(0.1, 0.2),
                z_rot=(2.0 * np.pi / 3.0, 7.0 * np.pi / 6.0),
                reference=self.table_offset,
            ),
            coffee_pod=dict(
                x=(-0.2, 0.05),
                y=(-0.3, -0.17),
                z_rot=(0.0, 0.0),
                reference=self.table_offset,
            ),
        )


class CoffeePreparation(Coffee):
    """
    Harder coffee task where the task starts with materials in drawer and coffee machine closed. The robot
    needs to retrieve the coffee pod and mug from the drawer, open the coffee machine, place the pod and mug
    in the machine, and then close the lid.
    """

    def _get_mug_model(self):
        """
        Allow subclasses to override which mug to use.
        """
        shapenet_id = "3143a4ac"  # beige round mug, works well and matches color scheme of other assets
        shapenet_scale = 1.0
        base_mjcf_path = os.path.join(mimicgen.__path__[0], "models/robosuite/assets/shapenet_core/mugs")
        mjcf_path = os.path.join(base_mjcf_path, "{}/model.xml".format(shapenet_id))

        self.mug = BlenderObject(
            name="mug",
            mjcf_path=mjcf_path,
            scale=shapenet_scale,
            solimp=(0.998, 0.998, 0.001),
            solref=(0.001, 1),
            density=100,
            # friction=(0.95, 0.3, 0.1),
            friction=(1, 1, 1),
            margin=0.001,
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Add camera with full tabletop perspective
        self._add_agentview_full_camera(mujoco_arena)

        # Set default agentview camera to be "agentview_full" (and send old agentview camera to agentview_full)
        old_agentview_camera = find_elements(
            root=mujoco_arena.worldbody, tags="camera", attribs={"name": "agentview"}, return_first=True
        )
        old_agentview_camera_pose = (old_agentview_camera.get("pos"), old_agentview_camera.get("quat"))
        old_agentview_full_camera = find_elements(
            root=mujoco_arena.worldbody, tags="camera", attribs={"name": "agentview_full"}, return_first=True
        )
        old_agentview_full_camera_pose = (old_agentview_full_camera.get("pos"), old_agentview_full_camera.get("quat"))
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=string_to_array(old_agentview_full_camera_pose[0]),
            quat=string_to_array(old_agentview_full_camera_pose[1]),
        )
        mujoco_arena.set_camera(
            camera_name="agentview_full",
            pos=string_to_array(old_agentview_camera_pose[0]),
            quat=string_to_array(old_agentview_camera_pose[1]),
        )

        # Create drawer object
        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        ceramic = CustomMaterial(
            texture="Ceramic",
            tex_name="ceramic",
            mat_name="MatCeramic",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        lightwood = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="MatLightWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"},
        )
        self.cabinet_object = LongDrawerObject(name="CabinetObject")

        # # old: manually set position in xml and add to mujoco arena
        # cabinet_object = self.cabinet_object.get_obj()
        # cabinet_object.set("pos", array_to_string((0.2, 0.30, 0.03)))
        # mujoco_arena.table_body.append(cabinet_object)
        obj_body = self.cabinet_object
        for material in [redwood, ceramic, lightwood]:
            tex_element, mat_element, _, used = add_material(
                root=obj_body.worldbody, naming_prefix=obj_body.naming_prefix, custom_material=deepcopy(material)
            )
            obj_body.asset.append(tex_element)
            obj_body.asset.append(mat_element)

        # Create mug
        self._get_mug_model()

        # Create coffee pod and machine (note that machine no longer has a cup!)
        self.coffee_pod = CoffeeMachinePodObject(name="coffee_pod")
        self.coffee_machine = CoffeeMachineObject(name="coffee_machine", add_cup=False)
        # objects = [self.coffee_pod, self.coffee_machine, self.cabinet_object, self.mug]
        objects = [self.coffee_pod, self.coffee_machine, self.cabinet_object]

        # Create placement initializer
        self._get_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objects,
        )
        # HACK: merge in mug afterwards because its number of geoms may change
        #       and this may break the generate_id_mappings function in task.py
        self.model.merge_objects([self.mug])  # add cleanup object to model

    def _padded_objects(self) -> tuple[str, ...]:
        return ("mug", "coffee_machine")

    def _full_rotation_objects(self) -> tuple[str, ...]:
        return ("mug",)

    def _tippable_objects(self) -> tuple[str, ...]:
        return ("mug",)

    def _fall_off_tracked_objects(self) -> tuple[str, ...]:
        return ("mug",)

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects (e.g.
        what happens when env.reset is called). Should return a dictionary with the following
        structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not change)
        """
        return dict(
            drawer=dict(
                # x=(0.1, 0.1),
                # y=(0.3, 0.3),
                # z_rot=(0.0, 0.0),
                x=(0.15, 0.15),
                y=(-0.35, -0.35),
                z_rot=(np.pi, np.pi),
                reference=self.table_offset,
            ),
            coffee_machine=dict(
                x=(-0.15, -0.15),
                y=(-0.25, -0.25),
                z_rot=(-np.pi / 6.0, -np.pi / 6.0),
                # put vertical
                # z_rot=(-np.pi / 2., -np.pi / 2.),
                reference=self.table_offset,
            ),
            mug=dict(
                # upper right
                # x=(-0.2, -0.2),
                # y=(0.17, 0.23),
                # z_rot=(0.0, 0.0),
                # lower right
                # x=(0.05, 0.20),
                # y=(-0.25, -0.05),
                # z_rot=(0.0, 0.0),
                x=(0.05, 0.20),
                y=(0.05, 0.25),
                z_rot=(0.0, 0.0),
                reference=self.table_offset,
            ),
            coffee_pod=dict(
                x=(-0.03, 0.03),
                y=(-0.05, 0.03),
                z_rot=(0.0, 0.0),
                reference=np.array((0.0, 0.0, 0.0)),
            ),
        )

    def _get_placement_initializer(self):
        bounds = self._maybe_apply_extra_randomization(self._get_initial_placement_bounds())

        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="DrawerSampler",
                mujoco_objects=self.cabinet_object,
                x_range=bounds["drawer"]["x"],
                y_range=bounds["drawer"]["y"],
                rotation=bounds["drawer"]["z_rot"],
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["drawer"]["reference"],
                z_offset=0.03,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CoffeeMachineSampler",
                mujoco_objects=self.coffee_machine,
                x_range=bounds["coffee_machine"]["x"],
                y_range=bounds["coffee_machine"]["y"],
                rotation=bounds["coffee_machine"]["z_rot"],
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["coffee_machine"]["reference"],
                z_offset=0.0,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="MugSampler",
                mujoco_objects=self.mug,
                x_range=bounds["mug"]["x"],
                y_range=bounds["mug"]["y"],
                rotation=bounds["mug"]["z_rot"],
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["mug"]["reference"],
                z_offset=0.02,
            )
        )

        # Note: coffee pod gets its own placement sampler to sample within a box within the drawer.

        # First, got drawer geom size with self.sim.model.geom_size[self.drawer_bottom_geom_id]
        # Value is array([0.08 , 0.09 , 0.008])
        # Then, used this to set reasonable box for pod init within drawer.
        self.pod_placement_initializer = UniformRandomSampler(
            name="CoffeePodInDrawerSampler",
            mujoco_objects=self.coffee_pod,
            x_range=bounds["coffee_pod"]["x"],
            y_range=bounds["coffee_pod"]["y"],
            rotation=bounds["coffee_pod"]["z_rot"],
            rotation_axis="z",
            # ensure_object_boundary_in_range=True, # make sure pod fits within the box
            ensure_object_boundary_in_range=False,  # make sure pod fits within the box
            ensure_valid_placement=True,
            reference_pos=bounds["coffee_pod"]["reference"],
            z_offset=0.0,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        self.cabinet_qpos_addr = self.sim.model.get_joint_qpos_addr(self.cabinet_object.joints[0])
        self.obj_body_id["drawer"] = self.sim.model.body_name2id(self.cabinet_object.root_body)
        self.obj_body_id["mug"] = self.sim.model.body_name2id(self.mug.root_body)
        self.drawer_bottom_geom_id = self.sim.model.geom_name2id("CabinetObject_drawer_bottom")

        # Geom-id caches for warp check_contact_groups (CPU check_contact / _check_grasp_tolerant no-op under warp).
        self.coffee_machine_base_geom_id = self.sim.model.geom_name2id("coffee_machine_base_g0")
        self.mug_contact_geom_ids = [
            self.sim.model.geom_name2id(g) for g in self.mug.contact_geoms
        ]

    def _reset_internal(self):
        """
        Resets simulation internal configurations.

        Warp branch samples placements per-env via ``sample_batch(k)``
        with ``k = |_reset_env_mask|`` and writes only masked qpos rows.
        The drawer is a fixture whose pose is baked into the model at
        XML-load time -- D0/D1/D2 drawer bounds are degenerate single
        points, so the CPU-side ``sim.model.body_pos`` write is a no-op
        under warp (``_warp_model`` is snapshotted once at init and does
        not re-read from the CPU model).
        """
        SingleArmEnv._reset_internal(self)

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            if self.use_warp:
                import warp as wp

                assert isinstance(self.sim, MjSimWarp)

                mask = getattr(self, "_reset_env_mask", None)
                if mask is None:
                    sample_idxs_arr = np.arange(self.num_envs)
                elif isinstance(mask, torch.Tensor):
                    sample_idxs_arr = mask.nonzero().flatten().cpu().numpy()
                else:
                    sample_idxs_arr = np.asarray(mask).nonzero()[0]

                if sample_idxs_arr.size > 0:
                    k = int(sample_idxs_arr.size)
                    placements = self.placement_initializer.sample_batch(k)
                    qpos_t = wp.to_torch(self.sim._warp_data.qpos)
                    row_idx = torch.as_tensor(
                        sample_idxs_arr, device=qpos_t.device, dtype=torch.long
                    )
                    for obj_pos, obj_quat, obj in placements.values():
                        if obj is self.cabinet_object:
                            # Fixture: no free joint; _warp_model snapshotted at init so body_pos writes wouldn't propagate.
                            continue
                        obj_pos, obj_quat = self._maybe_tip_placement_batch(obj, obj_pos, obj_quat)
                        addr = self.sim.model.get_joint_qpos_addr(obj.joints[0])
                        start, end = addr if isinstance(addr, tuple) else (addr, addr + 1)
                        stacked = np.concatenate(
                            [obj_pos.astype(np.float32), obj_quat.astype(np.float32)], axis=-1
                        )  # (k, 7)
                        qpos_t[row_idx, start:end] = torch.as_tensor(
                            stacked, device=qpos_t.device, dtype=torch.float32
                        )
            else:
                # Sample from the placement initializer for all objects
                object_placements = self.placement_initializer.sample()

                # Loop through all objects and reset their positions
                for obj_pos, obj_quat, obj in object_placements.values():
                    if obj is self.cabinet_object:
                        # object is fixture - set pose in model
                        body_id = self.sim.model.body_name2id(obj.root_body)
                        obj_pos_to_set = np.array(obj_pos)
                        # obj_pos_to_set[2] = 0.905 # hardcode z-value to correspond to parent class
                        obj_pos_to_set[2] = 0.805  # hardcode z-value to make sure it lies on table surface
                        self.sim.model.body_pos[body_id] = obj_pos_to_set
                        self.sim.model.body_quat[body_id] = obj_quat
                    else:
                        # object has free joint - use it to set pose
                        obj_pos, obj_quat = self._maybe_tip_placement(obj, obj_pos, obj_quat)
                        self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Masked hinge + drawer-slide reset: unmasked rows kept to preserve in-progress drawer openings.
        if isinstance(self.sim, MjSimWarp):
            import warp as wp
            qpos_t = wp.to_torch(self.sim._warp_data.qpos)
            mask = getattr(self, "_reset_env_mask", None)
            if mask is None:
                qpos_t[:, self.hinge_qpos_addr] = 0.0
                qpos_t[:, self.cabinet_qpos_addr] = 0.0
            else:
                row_idx = (mask.nonzero().flatten() if isinstance(mask, torch.Tensor)
                           else torch.as_tensor(np.asarray(mask).nonzero()[0],
                                                device=qpos_t.device, dtype=torch.long))
                qpos_t[row_idx, self.hinge_qpos_addr] = 0.0
                qpos_t[row_idx, self.cabinet_qpos_addr] = 0.0
        else:
            self.sim.data.qpos[self.hinge_qpos_addr] = 0.0
            self.sim.data.qpos[self.cabinet_qpos_addr] = 0.0
        self.sim.forward()

        if not self.deterministic_reset:
            if self.use_warp:
                import warp as wp
                assert isinstance(self.sim, MjSimWarp)

                mask = getattr(self, "_reset_env_mask", None)
                if mask is None:
                    sample_idxs_arr = np.arange(self.num_envs)
                elif isinstance(mask, torch.Tensor):
                    sample_idxs_arr = mask.nonzero().flatten().cpu().numpy()
                else:
                    sample_idxs_arr = np.asarray(mask).nonzero()[0]

                if sample_idxs_arr.size > 0:
                    k = int(sample_idxs_arr.size)
                    coffee_pod_placement = self.pod_placement_initializer.sample_batch(k, on_top=False)
                    assert len(coffee_pod_placement) == 1
                    rel_pod_pos, rel_pod_quat, pod_obj = list(coffee_pod_placement.values())[0]
                    rel_pod_pos = np.asarray(rel_pod_pos, dtype=np.float64)  # (k, 3)
                    rel_pod_quat = np.asarray(rel_pod_quat, dtype=np.float64)  # (k, 4) wxyz
                    assert pod_obj is self.coffee_pod

                    # Drawer shared across envs; use env-0 geom pos + CPU model quat.
                    drawer_bottom_geom_pos = self.sim.data.geom_xpos[self.drawer_bottom_geom_id][0].cpu().numpy()
                    drawer_rot_mat = T.quat2mat(
                        T.convert_quat(
                            self.sim.model.body_quat[self.sim.model.body_name2id(self.cabinet_object.root_body)],
                            to="xyzw",
                        )
                    )

                    # Rotate the sampled in-drawer offsets into the world frame.
                    rel_pod_pos[:, :2] = rel_pod_pos[:, :2] @ drawer_rot_mat[:2, :2].T

                    # Per-row quat composition (k rows only, once per masked reset).
                    pod_quat_batch = np.empty_like(rel_pod_quat)
                    for i in range(k):
                        rel_pod_mat_i = T.quat2mat(T.convert_quat(rel_pod_quat[i], to="xyzw"))
                        pod_mat_i = drawer_rot_mat.dot(rel_pod_mat_i)
                        pod_quat_batch[i] = T.convert_quat(T.mat2quat(pod_mat_i), to="wxyz")

                    drawer_bottom_geom_z_offset = self.sim.model.geom_size[self.drawer_bottom_geom_id][-1]
                    coffee_pod_bottom_offset = np.abs(self.coffee_pod.bottom_offset[-1])
                    coffee_pod_z = (
                        drawer_bottom_geom_pos[2]
                        + drawer_bottom_geom_z_offset
                        + coffee_pod_bottom_offset
                        + 0.001
                    )

                    pod_pos = rel_pod_pos + drawer_bottom_geom_pos  # (k, 3); drawer pos broadcast
                    pod_pos[:, 2] = coffee_pod_z

                    qpos_t = wp.to_torch(self.sim._warp_data.qpos)
                    row_idx = torch.as_tensor(
                        sample_idxs_arr, device=qpos_t.device, dtype=torch.long
                    )
                    addr = self.sim.model.get_joint_qpos_addr(pod_obj.joints[0])
                    start, end = addr if isinstance(addr, tuple) else (addr, addr + 1)
                    stacked = np.concatenate(
                        [pod_pos.astype(np.float32), pod_quat_batch.astype(np.float32)], axis=-1
                    )  # (k, 7)
                    qpos_t[row_idx, start:end] = torch.as_tensor(
                        stacked, device=qpos_t.device, dtype=torch.float32
                    )
            else:
                # sample pod location relative to center of drawer bottom geom surface
                coffee_pod_placement = self.pod_placement_initializer.sample(on_top=False)
                assert len(coffee_pod_placement) == 1
                rel_pod_pos, rel_pod_quat, pod_obj = list(coffee_pod_placement.values())[0]
                rel_pod_pos, rel_pod_quat = np.array(rel_pod_pos), np.array(rel_pod_quat)
                assert pod_obj is self.coffee_pod

                # center of drawer bottom
                drawer_bottom_geom_pos = np.array(self.sim.data.geom_xpos[self.drawer_bottom_geom_id])

                # our x-y relative position is sampled with respect to drawer geom frame. Here, we use the drawer's rotation
                # matrix to convert this relative position to a world relative position, so we can add it to the drawer world position
                drawer_rot_mat = T.quat2mat(
                    T.convert_quat(
                        self.sim.model.body_quat[self.sim.model.body_name2id(self.cabinet_object.root_body)], to="xyzw"
                    )
                )
                rel_pod_pos[:2] = drawer_rot_mat[:2, :2].dot(rel_pod_pos[:2])

                # also convert the sampled pod rotation to world frame
                rel_pod_mat = T.quat2mat(T.convert_quat(rel_pod_quat, to="xyzw"))
                pod_mat = drawer_rot_mat.dot(rel_pod_mat)
                pod_quat = T.convert_quat(T.mat2quat(pod_mat), to="wxyz")

                # get half-sizes of drawer geom and coffee pod to place coffee pod at correct z-location (on top of drawer bottom geom)
                drawer_bottom_geom_z_offset = self.sim.model.geom_size[self.drawer_bottom_geom_id][
                    -1
                ]  # half-size of geom in z-direction
                coffee_pod_bottom_offset = np.abs(self.coffee_pod.bottom_offset[-1])
                coffee_pod_z = drawer_bottom_geom_pos[2] + drawer_bottom_geom_z_offset + coffee_pod_bottom_offset + 0.001

                # set coffee pod in center of drawer
                pod_pos = np.array(drawer_bottom_geom_pos) + rel_pod_pos
                pod_pos[-1] = coffee_pod_z
                self.sim.data.set_joint_qpos(pod_obj.joints[0], np.concatenate([np.array(pod_pos), np.array(pod_quat)]))

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """

        # super class will populate observables for "mug" and "drawer" poses in addition to coffee machine and coffee pod
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            # add drawer joint angle observable
            @sensor(modality=modality)
            def drawer_joint_angle(obs_cache):
                if isinstance(self.sim, MjSimWarp):
                    return self.sim.data.qpos[[self.cabinet_qpos_addr]]  # (num_envs, 1) torch.Tensor
                return np.array([self.sim.data.qpos[self.cabinet_qpos_addr]])

            sensors = [drawer_joint_angle]
            names = ["drawer_joint_angle"]
            actives = [True]

            # Create observables
            for name, s, active in zip(names, sensors, actives):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    active=active,
                )

        return observables

    def _check_mug_placement(self):
        """
        Returns true if mug has been placed successfully on the coffee machine.

        Scalar bool under CPU sims; ``(num_envs,)`` bool tensor under warp.
        """
        mug_bid = self.obj_body_id["mug"]
        if isinstance(self.sim, MjSimWarp):
            # body_xmat is stored as (num_envs, nbody, 3, 3). The upright
            # check needs xmat[:, 2, 2] (z-column's z-row entry).
            xmat = self.sim.data.body_xmat[mug_bid]  # (N, 3, 3)
            z_axis_z = xmat[..., 2, 2]  # (N,)
            mug_upright = (1.0 - z_axis_z) < 1e-3
            mug_on_machine = self.sim.check_contact_groups(
                [self.coffee_machine_base_geom_id], self.mug_contact_geom_ids
            )
            return mug_upright & mug_on_machine

        # check z-axis alignment by checking z unit-vector of obj pose and dot with (0, 0, 1)
        # then take cosine dist (1 - dot-prod)
        obj_rot = self.sim.data.body_xmat[mug_bid].reshape(3, 3)
        z_axis = obj_rot[:3, 2]
        dist_to_z_axis = 1.0 - z_axis[2]
        mug_upright = dist_to_z_axis < 1e-3

        # to check if mug is placed on the machine successfully, we check that the mug is upright, and that it is
        # making contact with the coffee machine base plate
        coffee_base_plate_geom = "coffee_machine_base_g0"
        mug_on_machine = self.check_contact(coffee_base_plate_geom, self.mug)

        return mug_upright and mug_on_machine

    def _get_partial_task_metrics(self):
        """
        Returns a dictionary of partial task metrics which correspond to different parts of the task being done.

        Under warp all values are ``(num_envs,)`` bool tensors; under CPU
        they are scalar bools. The ``task`` key is AND-combined with
        mug-placement using the type-appropriate operator.
        """

        # populate with superclass metrics that concern the coffee pod and coffee machine
        metrics = super()._get_partial_task_metrics()

        # Mug grasp via both-fingerpads-contact proxy (CPU _check_grasp_tolerant no-ops under warp).
        if isinstance(self.sim, MjSimWarp):
            left_hit = self.sim.check_contact_groups(
                self.left_fingerpad_geom_ids, self.mug_contact_geom_ids
            )
            right_hit = self.sim.check_contact_groups(
                self.right_fingerpad_geom_ids, self.mug_contact_geom_ids
            )
            metrics["mug_grasp"] = left_hit & right_hit
        else:
            metrics["mug_grasp"] = self._check_grasp_tolerant(
                gripper=self.robots[0].gripper, object_geoms=[g for g in self.mug.contact_geoms]
            )

        # whether mug has been placed on coffee machine
        metrics["mug_place"] = self._check_mug_placement()

        # new task success includes mug placement
        if isinstance(self.sim, MjSimWarp):
            metrics["task"] = metrics["task"] & metrics["mug_place"]
        else:
            metrics["task"] = metrics["task"] and metrics["mug_place"]

        # can have a check on drawer being closed here, to make the task even harder
        # print(self.sim.data.qpos[self.cabinet_qpos_addr])

        return metrics


class CoffeePreparation_D0(CoffeePreparation):
    """Rename base class for convenience."""

    pass


class CoffeePreparation_D1(CoffeePreparation_D0):
    """
    Broader initialization for mug (whole right side of table, with rotation) and
    modest movement for coffee machine (some translation and rotation).
    """

    def _get_initial_placement_bounds(self):
        return dict(
            drawer=dict(
                x=(0.15, 0.15),
                y=(-0.35, -0.35),
                z_rot=(np.pi, np.pi),
                reference=self.table_offset,
            ),
            coffee_machine=dict(
                x=(-0.25, -0.15),
                y=(-0.30, -0.25),
                z_rot=(-np.pi / 6.0, np.pi / 6.0),
                reference=self.table_offset,
            ),
            mug=dict(
                x=(-0.15, 0.20),
                y=(0.05, 0.25),
                z_rot=(0.0, 2.0 * np.pi),
                reference=self.table_offset,
            ),
            coffee_pod=dict(
                x=(-0.03, 0.03),
                y=(-0.05, 0.03),
                z_rot=(0.0, 0.0),
                reference=np.array((0.0, 0.0, 0.0)),
            ),
        )
