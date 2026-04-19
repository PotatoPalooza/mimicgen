# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import warp as wp

from collections import OrderedDict
import numpy as np
import torch

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial, find_elements, string_to_array

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.binding_utils import MjSimWarp
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.environments.manipulation.stack import Stack

from mimicgen.envs.robosuite.single_arm_env_mg import SingleArmEnv_MG


class Stack_D0(Stack, SingleArmEnv_MG):
    """
    Augment robosuite stack task for mimicgen.
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"

        # Gated fall-off early termination: any tracked cube dropping below
        # table_offset[2] - fall_off_z_margin triggers a per-env early term.
        self.fall_off_termination = bool(kwargs.pop("fall_off_termination", False))
        self.fall_off_z_margin = float(kwargs.pop("fall_off_z_margin", 0.1))

        bounds = self._get_initial_placement_bounds()

        # ensure cube symmetry
        assert len(bounds) == 2
        for k in ["x", "y", "z_rot", "reference"]:
            assert np.array_equal(np.array(bounds["cubeA"][k]), np.array(bounds["cubeB"][k]))

        placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            x_range=bounds["cubeA"]["x"],
            y_range=bounds["cubeA"]["y"],
            rotation=bounds["cubeA"]["z_rot"],
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=bounds["cubeA"]["reference"],
            z_offset=0.01,
        )

        Stack.__init__(self, placement_initializer=placement_initializer, **kwargs)

    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

    # ------------------------------------------------------------------
    # References / geom-id caches for warp contact-group queries
    # ------------------------------------------------------------------

    def _setup_references(self):
        super()._setup_references()

        gripper = self.robots[0].gripper
        self.cubeA_contact_geom_ids = [
            self.sim.model.geom_name2id(g) for g in self.cubeA.contact_geoms
        ]
        self.cubeB_contact_geom_ids = [
            self.sim.model.geom_name2id(g) for g in self.cubeB.contact_geoms
        ]
        self.left_fingerpad_geom_ids = [
            self.sim.model.geom_name2id(g) for g in gripper.important_geoms["left_fingerpad"]
        ]
        self.right_fingerpad_geom_ids = [
            self.sim.model.geom_name2id(g) for g in gripper.important_geoms["right_fingerpad"]
        ]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_internal(self):
        """Mask-aware per-env placement sampling + qpos writes under warp.

        Bypasses upstream ``Stack._reset_internal`` (full-batch tile write)
        so kept envs' cube qpos flows through untouched during partial
        resets.
        """
        SingleArmEnv._reset_internal(self)

        if self.deterministic_reset:
            return

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

            if sample_idxs_arr.size == 0:
                return

            k = int(sample_idxs_arr.size)
            placements = self.placement_initializer.sample_batch(k)
            qpos_t = wp.to_torch(self.sim._warp_data.qpos)
            row_idx = torch.as_tensor(
                sample_idxs_arr, device=qpos_t.device, dtype=torch.long
            )
            for obj_pos, obj_quat, obj in placements.values():
                addr = self.sim.model.get_joint_qpos_addr(obj.joints[0])
                start, end = addr if isinstance(addr, tuple) else (addr, addr + 1)
                stacked = np.concatenate(
                    [obj_pos.astype(np.float32), obj_quat.astype(np.float32)], axis=-1
                )
                qpos_t[row_idx, start:end] = torch.as_tensor(
                    stacked, device=qpos_t.device, dtype=torch.float32
                )
            return

        object_placements = self.placement_initializer.sample()
        for obj_pos, obj_quat, obj in object_placements.values():
            self.sim.data.set_joint_qpos(
                obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)])
            )

    # ------------------------------------------------------------------
    # Reward / success
    # ------------------------------------------------------------------

    def reward(self, action: np.ndarray | wp.array | None = None) -> float | torch.Tensor:
        """Sparse reward (``2.0 * reward_scale / 2.0 = reward_scale`` on stack).

        Warp path returns ``(num_envs,)`` float tensor; CPU path delegates to
        upstream ``Stack.reward`` (supports shaped rewards).
        """
        if isinstance(self.sim, MjSimWarp):
            r_stack = self._check_cubeA_stacked()
            reward = r_stack.float() * 2.0
            if self.reward_scale is not None:
                reward = reward * (self.reward_scale / 2.0)
            return reward
        return Stack.reward(self, action=action)

    def _check_success(self):
        """Under warp, upstream ``Stack._check_success`` goes through the
        scalar ``staged_rewards`` path which is CPU-only. Short-circuit to
        the same predicate the sparse reward uses.
        """
        stacked = self._check_cubeA_stacked()
        if isinstance(stacked, torch.Tensor):
            return stacked
        return bool(stacked)

    def _check_lifted(self, body_id, margin=0.04):
        """Cube-above-table check. Returns ``(N,)`` bool tensor under warp,
        scalar bool under CPU.
        """
        if isinstance(self.sim, MjSimWarp):
            body_pos = self.sim.data.body_xpos[body_id]  # (N, 3)
            return body_pos[..., 2] > (float(self.table_offset[2]) + float(margin))
        body_pos = self.sim.data.body_xpos[body_id]
        return body_pos[2] > (self.table_offset[2] + margin)

    def _check_cubeA_grasped(self):
        """Robot grasping cubeA: contact on both fingerpads (warp) / CPU
        ``_check_grasp`` (no-warp).
        """
        if isinstance(self.sim, MjSimWarp):
            left_hit = self.sim.check_contact_groups(
                self.left_fingerpad_geom_ids, self.cubeA_contact_geom_ids
            )
            right_hit = self.sim.check_contact_groups(
                self.right_fingerpad_geom_ids, self.cubeA_contact_geom_ids
            )
            return left_hit & right_hit
        return self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)

    def _check_cubeA_lifted(self):
        return self._check_lifted(self.cubeA_body_id, margin=0.04)

    def _check_cubeA_stacked(self):
        if isinstance(self.sim, MjSimWarp):
            grasping = self._check_cubeA_grasped()  # (N,) bool
            lifted = self._check_cubeA_lifted()  # (N,) bool
            touching = self.sim.check_contact_groups(
                self.cubeA_contact_geom_ids, self.cubeB_contact_geom_ids
            )  # (N,) bool
            return (~grasping) & lifted & touching
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        cubeA_lifted = self._check_cubeA_lifted()
        cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
        return (not grasping_cubeA) and cubeA_lifted and cubeA_touching_cubeB

    # ------------------------------------------------------------------
    # Early-termination hook
    # ------------------------------------------------------------------

    def _fall_off_tracked_objects(self) -> tuple[str, ...]:
        return ("cubeA", "cubeB")

    def _fall_off_body_id(self, obj_name: str):
        return {"cubeA": self.cubeA_body_id, "cubeB": self.cubeB_body_id}.get(obj_name)

    def _check_early_termination(self):
        extras = super()._check_early_termination()
        if not self.fall_off_termination:
            return extras
        threshold = float(self.table_offset[2]) - float(self.fall_off_z_margin)
        for obj_name in self._fall_off_tracked_objects():
            bid = self._fall_off_body_id(obj_name)
            if bid is None:
                continue
            pos = self.sim.data.body_xpos[bid]
            extras[f"fell_off_{obj_name}"] = pos[..., 2] < threshold
        return extras

    # ------------------------------------------------------------------
    # Arena / model / observables
    # ------------------------------------------------------------------

    def _load_arena(self):
        """
        Allow subclasses to easily override arena settings.
        """

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

        return mujoco_arena

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = self._load_arena()

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        cubes = [self.cubeA, self.cubeB]
        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.08, 0.08],
                y_range=[-0.08, 0.08],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
        )

    def _setup_observables(self):
        """Rebuild observables with warp-aware sensors. Under warp, pos /
        quat / relative sensors return ``(N, 3)`` / ``(N, 4)`` torch tensors.
        """
        # Skip upstream Stack._setup_observables (CPU-only sensors); rebuild
        # from the SingleArmEnv base and append warp-aware object sensors.
        observables = SingleArmEnv._setup_observables(self)

        if not self.use_object_obs:
            return observables

        pf = self.robots[0].robot_model.naming_prefix
        modality = "object"
        warp = isinstance(self.sim, MjSimWarp)

        def _zeros(shape):
            if warp:
                return torch.zeros(
                    self.num_envs, shape, dtype=torch.float32, device="cuda"
                )
            return np.zeros(shape)

        @sensor(modality=modality)
        def cubeA_pos(obs_cache):
            if warp:
                return self.sim.data.body_xpos[self.cubeA_body_id]  # (N, 3)
            return np.array(self.sim.data.body_xpos[self.cubeA_body_id])

        @sensor(modality=modality)
        def cubeA_quat(obs_cache):
            if warp:
                q = self.sim.data.body_xquat[self.cubeA_body_id]  # (N, 4) wxyz
                return q[..., [1, 2, 3, 0]]
            return convert_quat(np.array(self.sim.data.body_xquat[self.cubeA_body_id]), to="xyzw")

        @sensor(modality=modality)
        def cubeB_pos(obs_cache):
            if warp:
                return self.sim.data.body_xpos[self.cubeB_body_id]
            return np.array(self.sim.data.body_xpos[self.cubeB_body_id])

        @sensor(modality=modality)
        def cubeB_quat(obs_cache):
            if warp:
                q = self.sim.data.body_xquat[self.cubeB_body_id]
                return q[..., [1, 2, 3, 0]]
            return convert_quat(np.array(self.sim.data.body_xquat[self.cubeB_body_id]), to="xyzw")

        @sensor(modality=modality)
        def gripper_to_cubeA(obs_cache):
            if "cubeA_pos" in obs_cache and f"{pf}eef_pos" in obs_cache:
                return obs_cache["cubeA_pos"] - obs_cache[f"{pf}eef_pos"]
            return _zeros(3)

        @sensor(modality=modality)
        def gripper_to_cubeB(obs_cache):
            if "cubeB_pos" in obs_cache and f"{pf}eef_pos" in obs_cache:
                return obs_cache["cubeB_pos"] - obs_cache[f"{pf}eef_pos"]
            return _zeros(3)

        @sensor(modality=modality)
        def cubeA_to_cubeB(obs_cache):
            if "cubeA_pos" in obs_cache and "cubeB_pos" in obs_cache:
                return obs_cache["cubeB_pos"] - obs_cache["cubeA_pos"]
            return _zeros(3)

        sensors = [cubeA_pos, cubeA_quat, cubeB_pos, cubeB_quat, gripper_to_cubeA, gripper_to_cubeB, cubeA_to_cubeB]
        names = [s.__name__ for s in sensors]

        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables

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
        return {
            k : dict(
                x=(-0.08, 0.08),
                y=(-0.08, 0.08),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.8)),
            )
            for k in ["cubeA", "cubeB"]
        }


class Stack_D1(Stack_D0):
    """
    Much wider initialization bounds.
    """
    def _load_arena(self):
        """
        Make default camera have full view of tabletop to account for larger init bounds.
        """
        mujoco_arena = super()._load_arena()

        # Set default agentview camera to be "agentview_full" (and send old agentview camera to agentview_full)
        old_agentview_camera = find_elements(root=mujoco_arena.worldbody, tags="camera", attribs={"name": "agentview"}, return_first=True)
        old_agentview_camera_pose = (old_agentview_camera.get("pos"), old_agentview_camera.get("quat"))
        old_agentview_full_camera = find_elements(root=mujoco_arena.worldbody, tags="camera", attribs={"name": "agentview_full"}, return_first=True)
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

        return mujoco_arena

    def _get_initial_placement_bounds(self):
        max_dim = 0.20
        return {
            k : dict(
                x=(-max_dim, max_dim),
                y=(-max_dim, max_dim),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.8)),
            )
            for k in ["cubeA", "cubeB"]
        }


class StackThree(Stack_D0):
    """
    Stack three cubes instead of two.
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"

        self.fall_off_termination = bool(kwargs.pop("fall_off_termination", False))
        self.fall_off_z_margin = float(kwargs.pop("fall_off_z_margin", 0.1))

        bounds = self._get_initial_placement_bounds()

        # ensure cube symmetry
        assert len(bounds) == 3
        for k in ["x", "y", "z_rot", "reference"]:
            assert np.array_equal(np.array(bounds["cubeA"][k]), np.array(bounds["cubeB"][k]))
            assert np.array_equal(np.array(bounds["cubeB"][k]), np.array(bounds["cubeC"][k]))

        placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            x_range=bounds["cubeA"]["x"],
            y_range=bounds["cubeA"]["y"],
            rotation=bounds["cubeA"]["z_rot"],
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=bounds["cubeA"]["reference"],
            z_offset=0.01,
        )

        Stack.__init__(self, placement_initializer=placement_initializer, **kwargs)

    # ------------------------------------------------------------------
    # Reward / success
    # ------------------------------------------------------------------

    def reward(self, action=None):
        """Sparse reward scaled by ``reward_scale`` (warp: ``(N,)`` float)."""
        success = self._check_success()
        if isinstance(success, torch.Tensor):
            reward = success.float()
            if self.reward_scale is not None:
                reward = reward * self.reward_scale
            return reward
        reward = 1.0 if success else 0.0
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward

    def _check_success(self):
        a_stacked = self._check_cubeA_stacked()
        c_stacked = self._check_cubeC_stacked()
        if isinstance(a_stacked, torch.Tensor) or isinstance(c_stacked, torch.Tensor):
            return a_stacked & c_stacked
        return bool(a_stacked) and bool(c_stacked)

    def _check_cubeC_lifted(self):
        # cube C needs to be higher than A
        return self._check_lifted(self.cubeC_body_id, margin=0.08)

    def _check_cubeC_grasped(self):
        if isinstance(self.sim, MjSimWarp):
            left_hit = self.sim.check_contact_groups(
                self.left_fingerpad_geom_ids, self.cubeC_contact_geom_ids
            )
            right_hit = self.sim.check_contact_groups(
                self.right_fingerpad_geom_ids, self.cubeC_contact_geom_ids
            )
            return left_hit & right_hit
        return self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeC)

    def _check_cubeC_stacked(self):
        if isinstance(self.sim, MjSimWarp):
            grasping = self._check_cubeC_grasped()  # (N,) bool
            lifted = self._check_cubeC_lifted()  # (N,) bool
            touching = self.sim.check_contact_groups(
                self.cubeC_contact_geom_ids, self.cubeA_contact_geom_ids
            )
            return (~grasping) & lifted & touching
        grasping_cubeC = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeC)
        cubeC_lifted = self._check_cubeC_lifted()
        cubeC_touching_cubeA = self.check_contact(self.cubeC, self.cubeA)
        return (not grasping_cubeC) and cubeC_lifted and cubeC_touching_cubeA

    def staged_rewards(self):
        """Placeholder staged rewards. Only the terminal ``r_stack`` component
        is populated — all three cubes stacked correctly. Warp path returns a
        ``(N,)`` float tensor; CPU returns a float.
        """
        stacked = self._check_success()
        if isinstance(stacked, torch.Tensor):
            r_stack = stacked.float()
            zero = torch.zeros_like(r_stack)
            return zero, zero, r_stack
        return 0.0, 0.0, (1.0 if stacked else 0.0)

    def _load_arena(self):
        """
        Allow subclasses to easily override arena settings.
        """

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

        return mujoco_arena

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = self._load_arena()

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="bluewood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        self.cubeC = BoxObject(
            name="cubeC",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=bluewood,
        )
        cubes = [self.cubeA, self.cubeB, self.cubeC]
        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.10, 0.10],
                y_range=[-0.10, 0.10],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
        )

    def _setup_references(self):
        """
        Add reference for cube C
        """
        super()._setup_references()

        # Additional object references from this env
        self.cubeC_body_id = self.sim.model.body_name2id(self.cubeC.root_body)
        self.cubeC_contact_geom_ids = [
            self.sim.model.geom_name2id(g) for g in self.cubeC.contact_geoms
        ]

    def _setup_observables(self):
        """Appends cubeC sensors (warp-aware) to the parent's observables."""
        observables = super()._setup_observables()

        if not self.use_object_obs:
            return observables

        pf = self.robots[0].robot_model.naming_prefix
        modality = "object"
        warp = isinstance(self.sim, MjSimWarp)

        def _zeros(shape):
            if warp:
                return torch.zeros(
                    self.num_envs, shape, dtype=torch.float32, device="cuda"
                )
            return np.zeros(shape)

        @sensor(modality=modality)
        def cubeC_pos(obs_cache):
            if warp:
                return self.sim.data.body_xpos[self.cubeC_body_id]
            return np.array(self.sim.data.body_xpos[self.cubeC_body_id])

        @sensor(modality=modality)
        def cubeC_quat(obs_cache):
            if warp:
                q = self.sim.data.body_xquat[self.cubeC_body_id]
                return q[..., [1, 2, 3, 0]]
            return convert_quat(np.array(self.sim.data.body_xquat[self.cubeC_body_id]), to="xyzw")

        @sensor(modality=modality)
        def gripper_to_cubeC(obs_cache):
            if "cubeC_pos" in obs_cache and f"{pf}eef_pos" in obs_cache:
                return obs_cache["cubeC_pos"] - obs_cache[f"{pf}eef_pos"]
            return _zeros(3)

        @sensor(modality=modality)
        def cubeA_to_cubeC(obs_cache):
            if "cubeA_pos" in obs_cache and "cubeC_pos" in obs_cache:
                return obs_cache["cubeC_pos"] - obs_cache["cubeA_pos"]
            return _zeros(3)

        @sensor(modality=modality)
        def cubeB_to_cubeC(obs_cache):
            if "cubeB_pos" in obs_cache and "cubeC_pos" in obs_cache:
                return obs_cache["cubeB_pos"] - obs_cache["cubeC_pos"]
            return _zeros(3)

        sensors = [cubeC_pos, cubeC_quat, gripper_to_cubeC, cubeA_to_cubeC, cubeB_to_cubeC]
        names = [s.__name__ for s in sensors]

        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables

    # ------------------------------------------------------------------
    # Early-termination hook
    # ------------------------------------------------------------------

    def _fall_off_tracked_objects(self) -> tuple[str, ...]:
        return ("cubeA", "cubeB", "cubeC")

    def _fall_off_body_id(self, obj_name: str):
        return {
            "cubeA": self.cubeA_body_id,
            "cubeB": self.cubeB_body_id,
            "cubeC": self.cubeC_body_id,
        }.get(obj_name)

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
        return {
            k : dict(
                x=(-0.10, 0.10),
                y=(-0.10, 0.10),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.8)),
            )
            for k in ["cubeA", "cubeB", "cubeC"]
        }


class StackThree_D0(StackThree):
    """Rename base class for convenience."""
    pass


class StackThree_D1(StackThree_D0):
    """
    Less z-rotation (for easier datagen) and much wider initialization bounds.
    """
    def _load_arena(self):
        """
        Make default camera have full view of tabletop to account for larger init bounds.
        """
        mujoco_arena = super()._load_arena()

        # Set default agentview camera to be "agentview_full" (and send old agentview camera to agentview_full)
        old_agentview_camera = find_elements(root=mujoco_arena.worldbody, tags="camera", attribs={"name": "agentview"}, return_first=True)
        old_agentview_camera_pose = (old_agentview_camera.get("pos"), old_agentview_camera.get("quat"))
        old_agentview_full_camera = find_elements(root=mujoco_arena.worldbody, tags="camera", attribs={"name": "agentview_full"}, return_first=True)
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

        return mujoco_arena

    def _get_initial_placement_bounds(self):
        max_dim = 0.20
        return {
            k : dict(
                x=(-max_dim, max_dim),
                y=(-max_dim, max_dim),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.8)),
            )
            for k in ["cubeA", "cubeB", "cubeC"]
        }
