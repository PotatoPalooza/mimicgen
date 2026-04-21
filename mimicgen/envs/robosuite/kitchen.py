# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import warp as wp

"""
Contains environments for BUDS kitchen task from robosuite task zoo repo.
((https://github.com/ARISE-Initiative/robosuite-task-zoo))
"""
import os
import numpy as np
import torch
from six import with_metaclass
from copy import deepcopy

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.models.objects import BoxObject, MujocoXMLObject
from robosuite.utils.binding_utils import MjSimWarp
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, string_to_array, find_elements, add_material
from robosuite.utils.buffers import RingBuffer

import robosuite_task_zoo
from robosuite_task_zoo.environments.manipulation.kitchen import KitchenEnv
from robosuite_task_zoo.models.kitchen import PotObject, StoveObject, ButtonObject, ServingRegionObject

import mimicgen
from mimicgen.envs.robosuite.single_arm_env_mg import SingleArmEnv_MG


class StoveObjectNew(StoveObject):
    """
    Override some offsets for placement sampler.
    """
    @property
    def bottom_offset(self):
        # unused since we directly hardcode z
        return np.array([0, 0, -0.02])

    @property
    def top_offset(self):
        # unused since we directly hardcode z
        return np.array([0, 0, 0.02])

    @property
    def horizontal_radius(self):
        return 0.1


class ButtonObjectNew(ButtonObject):
    """
    Override some offsets for placement sampler.
    """
    @property
    def horizontal_radius(self):
        return 0.04


class ServingRegionObjectNew(MujocoXMLObject):
    """
    Override some offsets for placement sampler, and also
    turn the site into a visual-only geom so that it shows up
    in the first env step.
    """
    def __init__(self, name, joints=None):
        # our custom serving region xml - turn site into visual-only geom so that it shows up on env reset (instead
        # of after first env step)
        path_to_serving_region_xml = os.path.join(mimicgen.__path__[0], "models/robosuite/assets/objects/serving_region.xml")
        super().__init__(path_to_serving_region_xml,
                         name=name, joints=None, obj_type="all", duplicate_collision_geoms=True)

    @property
    def horizontal_radius(self):
        return 0.123


class Kitchen_D0(KitchenEnv, SingleArmEnv_MG):
    """
    Augment BUDS kitchen task for mimicgen.
    """
    def __init__(
        self,
        fall_off_termination: bool = False,
        fall_off_z_margin: float = 0.1,
        **kwargs,
    ):
        # Early-termination kwargs. Threshold uses ``table_offset[2] =
        # 0.90`` (KitchenEnv's hardcoded world-frame table height).
        self.fall_off_termination = fall_off_termination
        self.fall_off_z_margin = fall_off_z_margin
        KitchenEnv.__init__(self, **kwargs)

        # some additional variables for better success check -- allocated
        # after the base init so the warp path can read self.sim.
        self._init_stove_state_tracking()

    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

    @property
    def _has_gripper_contact(self):
        """Upstream reads ``robots[0].ee_force`` which is not wired
        through warp -- return per-env zero under warp.
        """
        if isinstance(self.sim, MjSimWarp):
            return torch.zeros(self.num_envs, 1, dtype=torch.float32, device="cuda")
        return KitchenEnv._has_gripper_contact.fget(self)

    def _init_stove_state_tracking(self):
        """Initialise the per-env stove / button latches.

        Under warp these are ``(num_envs,)`` bool tensors on device;
        under CPU they remain scalar bools / ``{i: bool}`` to match
        upstream semantics. ``buttons_on`` is re-initialised below as a
        tensor for warp so ``_post_process`` can update it in-place.
        """
        if isinstance(self.sim, MjSimWarp):
            dev = "cuda"
            self.has_stove_turned_on = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
            self.has_stove_turned_on_with_pot_and_object = torch.zeros(
                self.num_envs, dtype=torch.bool, device=dev
            )
            # Replace scalar-bool dict with per-env tensors.
            for k in list(self.buttons_on.keys()):
                self.buttons_on[k] = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        else:
            self.has_stove_turned_on = False
            self.has_stove_turned_on_with_pot_and_object = False

    def _setup_references(self):
        super()._setup_references()
        # Geom-id caches for warp contact-group queries.
        self.pot_body_geom_ids = [self.sim.model.geom_name2id("PotObject_body_0")]
        self.stove_burner_geom_ids = [self.sim.model.geom_name2id("Stove1_collision_burner")]
        self.bread_contact_geom_ids = [
            self.sim.model.geom_name2id(g) for g in self.bread_ingredient.contact_geoms
        ]
        self.pot_contact_geom_ids = [
            self.sim.model.geom_name2id(g) for g in self.pot_object.contact_geoms
        ]

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Warp-aware override of upstream KitchenEnv obj-sensors.
        """
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            bid = self.obj_body_id[obj_name]
            if isinstance(self.sim, MjSimWarp):
                return self.sim.data.body_xpos[bid]
            return np.array(self.sim.data.body_xpos[bid])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            bid = self.obj_body_id[obj_name]
            if isinstance(self.sim, MjSimWarp):
                q = self.sim.data.body_xquat[bid]
                return q[..., [1, 2, 3, 0]]
            return T.convert_quat(self.sim.data.body_xquat[bid], to="xyzw")

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            if any([name not in obs_cache for name in
                    [f"{obj_name}_pos", f"{obj_name}_quat", "world_pose_in_gripper"]]):
                if isinstance(self.sim, MjSimWarp):
                    return torch.zeros(
                        self.num_envs, 3, dtype=torch.float32,
                        device="cuda",
                    )
                return np.zeros(3)
            if isinstance(self.sim, MjSimWarp):
                obj_pos_t = obs_cache[f"{obj_name}_pos"]
                obj_quat_t = obs_cache[f"{obj_name}_quat"]
                world_poses = obs_cache["world_pose_in_gripper"]
                obj_pose = T.pose2mat_torch(obj_pos_t, obj_quat_t)
                rel_pose = world_poses @ obj_pose
                obs_cache[f"{obj_name}_to_{pf}eef_quat"] = T.mat2quat_torch(rel_pose[:, :3, :3])
                return rel_pose[:, :3, 3]
            obj_pose = T.pose2mat((obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"]))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            key = f"{obj_name}_to_{pf}eef_quat"
            if key in obs_cache:
                return obs_cache[key]
            if isinstance(self.sim, MjSimWarp):
                return torch.zeros(
                    self.num_envs, 4, dtype=torch.float32,
                    device="cuda",
                )
            return np.zeros(4)

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [f"{obj_name}_pos", f"{obj_name}_quat", f"{obj_name}_to_{pf}eef_pos", f"{obj_name}_to_{pf}eef_quat"]
        return sensors, names

    def _reset_internal(self):
        """
        Update from superclass to ensure we reset state variables.

        Under warp, scope the button-qpos reset and the stove-state
        latches to ``_reset_env_mask`` so kept envs' in-progress cooking
        state flows through untouched.
        """
        KitchenEnv._reset_internal(self)

        mask = getattr(self, "_reset_env_mask", None)
        if isinstance(self.sim, MjSimWarp):
            # Ensure latches are allocated (hard-reset rebuilds sim/data).
            if not isinstance(self.has_stove_turned_on, torch.Tensor):
                self._init_stove_state_tracking()
            if mask is None:
                self.has_stove_turned_on.zero_()
                self.has_stove_turned_on_with_pot_and_object.zero_()
                for k in self.buttons_on:
                    if isinstance(self.buttons_on[k], torch.Tensor):
                        self.buttons_on[k].zero_()
            else:
                if isinstance(mask, torch.Tensor):
                    m = mask.to(self.has_stove_turned_on.device)
                else:
                    m = torch.as_tensor(
                        np.asarray(mask, dtype=bool),
                        device=self.has_stove_turned_on.device,
                    )
                self.has_stove_turned_on[m] = False
                self.has_stove_turned_on_with_pot_and_object[m] = False
                for k in self.buttons_on:
                    if isinstance(self.buttons_on[k], torch.Tensor):
                        self.buttons_on[k][m] = False
        else:
            self.has_stove_turned_on = False
            self.has_stove_turned_on_with_pot_and_object = False

    def _check_success(self):
        """
        More stringent success check than upstream: pot must have been
        on the burner while the stove was on and bread was in the pot.

        Returns scalar bool under CPU and ``(num_envs,)`` bool tensor
        under warp. Latches (``has_stove_turned_on``,
        ``has_stove_turned_on_with_pot_and_object``) are updated in place
        before returning so subsequent steps see the same latch state.
        """
        if isinstance(self.sim, MjSimWarp):
            pot_pos = self.sim.data.body_xpos[self.pot_object_id]  # (N, 3)
            serving_region_pos = self.sim.data.body_xpos[self.serving_region_id]  # (N, 3)
            diff = torch.abs(serving_region_pos - pot_pos)
            pot_in_serving_region = (
                (diff[..., 0] < 0.05) & (diff[..., 1] < 0.10) & (diff[..., 2] < 0.05)
            )
            pot_on_burner = self.sim.check_contact_groups(
                self.pot_body_geom_ids, self.stove_burner_geom_ids
            )
            object_in_pot = self.sim.check_contact_groups(
                self.bread_contact_geom_ids, self.pot_contact_geom_ids
            )
            stove_on = self.buttons_on[1]  # (N,) bool tensor
            stove_turned_off = ~stove_on
            self.has_stove_turned_on |= stove_on
            self.has_stove_turned_on_with_pot_and_object |= (
                self.has_stove_turned_on & object_in_pot & pot_on_burner
            )
            return (
                pot_in_serving_region
                & stove_turned_off
                & object_in_pot
                & self.has_stove_turned_on_with_pot_and_object
            )

        pot_pos = self.sim.data.body_xpos[self.pot_object_id]
        serving_region_pos = self.sim.data.body_xpos[self.serving_region_id]
        dist_serving_pot = serving_region_pos - pot_pos
        pot_in_serving_region = np.abs(dist_serving_pot[0]) < 0.05 and np.abs(dist_serving_pot[1]) < 0.10 and np.abs(dist_serving_pot[2]) < 0.05

        # if pot bottom is in contact with stove
        pot_bottom_in_contact_with_stove = self.check_contact("PotObject_body_0", "Stove1_collision_burner")

        # if object is in pot
        object_in_pot = self.check_contact(self.bread_ingredient, self.pot_object)
        stove_turned_off = not self.buttons_on[1]
        if not stove_turned_off:
            self.has_stove_turned_on = True
            if self.has_stove_turned_on and object_in_pot and pot_bottom_in_contact_with_stove:
                self.has_stove_turned_on_with_pot_and_object = True

        return pot_in_serving_region and stove_turned_off and object_in_pot and self.has_stove_turned_on_with_pot_and_object

    def _post_process(self):
        """Per-env button state update.

        Under warp reads the button qpos tensor and sets
        ``buttons_on[i]`` directly -- the scalar upstream latch logic is
        replaced with ``buttons_on = (qpos >= 0)``, which is equivalent
        (see upstream _post_process). Site visibility is not updated
        under warp (visual only; not in the warp model snapshot).
        """
        if isinstance(self.sim, MjSimWarp):
            # visualize() fires _post_process pre-_init_stove_state_tracking; skip if unpopulated.
            if not getattr(self, "button_qpos_addrs", None):
                return
            for i in range(1, self.num_stoves + 1):
                qpos = self.sim.data.qpos[self.button_qpos_addrs[i]]  # (N,)
                prev = self.buttons_on.get(i) if isinstance(self.buttons_on, dict) else None
                if not isinstance(prev, torch.Tensor):
                    self.buttons_on[i] = (qpos >= 0.0)
                else:
                    prev.copy_(qpos >= 0.0)
            return
        return KitchenEnv._post_process(self)

    def _get_observations(self, force_update=False):
        """
        Make sure switch states are up-to-date before observations are returned - this is also
        important for scripts that reset to intermediate demonstration states.
        """
        self._post_process()
        return KitchenEnv._get_observations(self, force_update=force_update)

    def _fall_off_tracked_objects(self) -> tuple[str, ...]:
        return ("pot", "bread")

    def _check_early_termination(self) -> dict[str, object]:
        """Pot/bread fall-off causes (table_offset[2]=0.90 - fall_off_z_margin)."""
        try:
            extras = super()._check_early_termination()
        except AttributeError:
            extras = {}
        if not self.fall_off_termination:
            return extras
        threshold = float(self.table_offset[2]) - float(self.fall_off_z_margin)
        name_to_bid = {
            "pot": self.pot_object_id,
            "bread": self.sim.model.body_name2id(self.bread_ingredient.root_body),
        }
        for obj_name in self._fall_off_tracked_objects():
            bid = name_to_bid.get(obj_name)
            if bid is None:
                continue
            pos = self.sim.data.body_xpos[bid]
            extras[f"fell_off_{obj_name}"] = pos[..., 2] < threshold
        return extras


class Kitchen_D1(Kitchen_D0):
    """
    Specify wider distribution for objects including objects that didn't move before. We also had to make some objects 
    movable that were fixtures before.
    """
    def _reset_internal(self):
        """
        Update to make sure placement initializer can be used to set poses of objects
        that used to be fixtures before.

        Warp branch: mask-aware ``sample_batch(k)`` for free-joint
        objects (bread, pot); fixtures (stove, button, serving region)
        are baked into the model at XML-load time and not re-randomised
        under warp (``_warp_model`` is a snapshot). Button qpos and
        stove-state latches are scoped to ``_reset_env_mask``.
        """
        SingleArmEnv._reset_internal(self)

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
                        if obj.name in self._hardcoded_z_offsets:
                            # Fixture -- no free joint; warp model snapshot.
                            continue
                        addr = self.sim.model.get_joint_qpos_addr(obj.joints[0])
                        start, end = addr if isinstance(addr, tuple) else (addr, addr + 1)
                        stacked = np.concatenate(
                            [obj_pos.astype(np.float32), obj_quat.astype(np.float32)], axis=-1
                        )
                        qpos_t[row_idx, start:end] = torch.as_tensor(
                            stacked, device=qpos_t.device, dtype=torch.float32
                        )
            else:
                object_placements = self.placement_initializer.sample()

                for obj_pos, obj_quat, obj in object_placements.values():
                    if obj.name in self._hardcoded_z_offsets:
                        body_id = self.sim.model.body_name2id(obj.root_body)
                        obj_pos_to_set = np.array(obj_pos)
                        obj_pos_to_set[2] = self._hardcoded_z_offsets[obj.name]
                        self.sim.model.body_pos[body_id] = obj_pos_to_set
                        self.sim.model.body_quat[body_id] = obj_quat
                    else:
                        self.sim.data.set_joint_qpos(
                            obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                        )

        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)
        self._history_force_torque = RingBuffer(dim=6, length=16)
        self._recent_force_torque = []

        # Reset stove/button latches (mask-aware under warp -- see
        # Kitchen_D0._reset_internal for the masked tensor path).
        if isinstance(self.sim, MjSimWarp):
            if not isinstance(self.has_stove_turned_on, torch.Tensor):
                self._init_stove_state_tracking()
            mask = getattr(self, "_reset_env_mask", None)
            if mask is None:
                self.has_stove_turned_on.zero_()
                self.has_stove_turned_on_with_pot_and_object.zero_()
                for k in self.buttons_on:
                    if isinstance(self.buttons_on[k], torch.Tensor):
                        self.buttons_on[k].zero_()
            else:
                m = (mask.to(self.has_stove_turned_on.device) if isinstance(mask, torch.Tensor)
                     else torch.as_tensor(np.asarray(mask, dtype=bool),
                                          device=self.has_stove_turned_on.device))
                self.has_stove_turned_on[m] = False
                self.has_stove_turned_on_with_pot_and_object[m] = False
                for k in self.buttons_on:
                    if isinstance(self.buttons_on[k], torch.Tensor):
                        self.buttons_on[k][m] = False
        else:
            self.buttons_on = {1: False}
            self.has_stove_turned_on = False
            self.has_stove_turned_on_with_pot_and_object = False

        # Switch-off qpos reset (mask-aware under warp).
        if isinstance(self.sim, MjSimWarp):
            import warp as wp
            qpos_t = wp.to_torch(self.sim._warp_data.qpos)
            mask = getattr(self, "_reset_env_mask", None)
            if mask is None:
                qpos_t[:, self.button_qpos_addrs[1]] = -0.3
            else:
                row_idx = (mask.nonzero().flatten() if isinstance(mask, torch.Tensor)
                           else torch.as_tensor(np.asarray(mask).nonzero()[0],
                                                device=qpos_t.device, dtype=torch.long))
                qpos_t[row_idx, self.button_qpos_addrs[1]] = -0.3
        else:
            self.sim.data.qpos[self.button_qpos_addrs[1]] = -0.3
            for stove_num, stove_status in self.buttons_on.items():
                self.stoves[stove_num].set_sites_visibility(sim=self.sim, visible=stove_status)

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
        # Broader bounds for all objects.
        return dict(
            bread=dict(
                x=(-0.2, 0.0),
                y=(-0.25, -0.05),
                # z_rot=(-np.pi / 2., -np.pi / 2.),
                z_rot=(-np.pi / 2., np.pi / 2.),
                reference=self.table_offset,
            ),
            pot=dict(
                x=(0.08, 0.18),
                y=(-0.2, -0.05),
                # z_rot=(-0.1, 0.1),
                z_rot=(-np.pi / 6., np.pi / 6.),
                reference=self.table_offset,
            ),
            stove=dict(
                x=(0.06, 0.23),
                y=(0.095, 0.25),
                z_rot=(0., 0.),
                reference=self.table_offset,
            ),
            button=dict(
                x=(-0.2, 0.06),
                y=(0.05, 0.2),
                z_rot=(np.pi, np.pi), # make z-rotation consistent with base env
                reference=self.table_offset,
            ),
            serving_region=dict(
                x=(0.345, 0.345),
                y=(-0.2, -0.05),
                z_rot=(0., 0.),
                reference=self.table_offset,
            ),
        )

    def _get_placement_initializer(self):
        bounds = self._get_initial_placement_bounds()
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # object
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler-bread",
                mujoco_objects=self.bread_ingredient,
                x_range=bounds["bread"]["x"],
                y_range=bounds["bread"]["y"],
                rotation=bounds["bread"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["bread"]["reference"],
                z_offset=0.01,
            )
        )

        # pot
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler-pot",
                mujoco_objects=self.pot_object,
                x_range=bounds["pot"]["x"],
                y_range=bounds["pot"]["y"],
                rotation=bounds["pot"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["pot"]["reference"],
                # z_offset=0.02,
                z_offset=-0.11, # account for pot vertical sites being wrong
            )
        )

        # stove
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler-stove",
                mujoco_objects=self.stove_object_1,
                x_range=bounds["stove"]["x"],
                y_range=bounds["stove"]["y"],
                rotation=bounds["stove"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["stove"]["reference"],
                z_offset=0.02,
            )
        )

        # button
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler-button",
                mujoco_objects=self.button_object_1,
                x_range=bounds["button"]["x"],
                y_range=bounds["button"]["y"],
                rotation=bounds["button"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["button"]["reference"],
                z_offset=0.02,
            )
        )

        # serving region
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler-serving",
                mujoco_objects=self.serving_region,
                x_range=bounds["serving_region"]["x"],
                y_range=bounds["serving_region"]["y"],
                rotation=bounds["serving_region"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["serving_region"]["reference"],
                z_offset=0.003,
            )
        )

    def _load_model(self):
        """
        Update to include fixtures that didn't move before in placement initializer, so
        they can move on each episode reset. Also updates the list of objects so that
        we get observables for the button.
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
            table_friction=(0.6, 0.005, 0.0001)
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5386131746834771, -4.392035683362857e-09, 1.4903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
        )

        mujoco_arena.set_camera(
            camera_name="sideview",
            pos=[0.5586131746834771, 0.3, 1.2903500240372423],
            quat=[0.4144233167171478, 0.3100920617580414,
            0.49641484022140503, 0.6968992352485657]
        )
        
        
        bread = CustomMaterial(
            texture="Bread",
            tex_name="bread",
            mat_name="MatBread",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )
        darkwood = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="MatDarkWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )

        metal = CustomMaterial(
            texture="Metal",
            tex_name="metal",
            mat_name="MatMetal",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "1", "shininess": "0.3", "rgba": "0.9 0.9 0.9 1"}
        )

        tex_attrib = {
            "type": "cube"
        }

        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1"
        }
        
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="handle1_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )

        self.stove_object_1 = StoveObjectNew(
            name="Stove1",
            joints=None,
        )

        # # old: manually set position in xml and add to mujoco arena
        # stove_body = self.stove_object_1.get_obj()
        # stove_body.set("pos", array_to_string((0.23, 0.095, 0.02)))
        # mujoco_arena.table_body.append(stove_body)

        self.button_object_1 = ButtonObjectNew(
            name="Button1",
        )

        # # old: manually set position in xml and add to mujoco arena
        # button_body = self.button_object_1.get_obj()
        # button_body.set("quat", array_to_string((0., 0., 0., 1.)))
        # button_body.set("pos", array_to_string((0.06, 0.10, 0.02)))
        # mujoco_arena.table_body.append(button_body)

        self.serving_region = ServingRegionObjectNew(
            name="ServingRegionRed"
        )

        # # old: manually set position in xml and add to mujoco arena
        # serving_region_object = self.serving_region.get_obj()
        # serving_region_object.set("pos", array_to_string((0.345, -0.15, 0.003)))
        # mujoco_arena.table_body.append(serving_region_object)
        
        self.pot_object = PotObject(
            name="PotObject",
        )
        
        for obj_body in [
                self.button_object_1,
                self.stove_object_1,
                self.serving_region,
        ]:
            for material in [darkwood, metal, redwood]:
                tex_element, mat_element, _, used = add_material(root=obj_body.worldbody,
                                                                 naming_prefix=obj_body.naming_prefix,
                                                                 custom_material=deepcopy(material))
                obj_body.asset.append(tex_element)
                obj_body.asset.append(mat_element)

        ingredient_size = [0.015, 0.025, 0.02]

        self.bread_ingredient = BoxObject(
            name="cube_bread",
            size_min=ingredient_size,
            size_max=ingredient_size,
            rgba=[1, 0, 0, 1],
            material=bread,
            density=500.,
        )
        
        # make placement initializer
        self._get_placement_initializer()
        
        mujoco_objects = [self.bread_ingredient,
                          self.pot_object,
                          self.stove_object_1,
                          self.button_object_1,
                          self.serving_region,
        ]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=mujoco_objects,
        )
        self.stoves = {1: self.stove_object_1,
                       # 2: self.stove_object_2
        }

        self.num_stoves = len(self.stoves.keys())
        
        self.buttons = {1: self.button_object_1,
                        # 2: self.button_object_2,
        }

        self.objects = [
            self.stove_object_1,
            self.bread_ingredient,
            self.pot_object,
            self.serving_region,
            self.button_object_1,
        ]
        
        self.model.merge_assets(self.button_object_1)
        self.model.merge_assets(self.stove_object_1)
        self.model.merge_assets(self.serving_region)

        # hardcode some z-offsets here
        self._hardcoded_z_offsets = {
            self.stove_object_1.name : 0.895,
            self.button_object_1.name : 0.895,
            self.serving_region.name : 0.878,
        }

    def visualize(self, vis_settings):
        """
        Update site visualization to make sure stove object site visualization is kept up to date.
        """
        super(Kitchen_D1, self).visualize(vis_settings)
        self._post_process()
