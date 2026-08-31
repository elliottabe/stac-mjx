"""Stac class handling high level functionality of stac-mjx."""

import jax
from jax import numpy as jp

import numpy as np

import mujoco
from mujoco import mjx

from stac_mjx import utils, rescale, compute_stac, io, stac_core

from omegaconf import DictConfig
from typing import List, Union
from pathlib import Path
from copy import deepcopy
import imageio
from tqdm import tqdm

# """Stac class handling high level functionality of stac-mjx."""

_ROOT_QPOS_LB = jp.concatenate([-jp.inf * jp.ones(3), -1.0 * jp.ones(4)])
_ROOT_QPOS_UB = jp.concatenate([jp.inf * jp.ones(3), 1.0 * jp.ones(4)])

# mujoco jnt_type enums: https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjtjoint
_MUJOCO_JOINT_TYPE_DIMS = {
    mujoco.mjtJoint.mjJNT_FREE: 7,
    mujoco.mjtJoint.mjJNT_BALL: 4,
    mujoco.mjtJoint.mjJNT_SLIDE: 1,
    mujoco.mjtJoint.mjJNT_HINGE: 1,
}

_MUJOCO_JOINT_TYPE_UNCONSTRAINED = {
    mujoco.mjtJoint.mjJNT_FREE: (
        jp.concatenate([-jp.inf * jp.ones(3), -1.0 * jp.ones(4)]),
        jp.concatenate([jp.inf * jp.ones(3), 1.0 * jp.ones(4)]),
    ),
    mujoco.mjtJoint.mjJNT_BALL: (
        jp.concatenate([-1.0 * jp.ones(4)]),
        jp.concatenate([1.0 * jp.ones(4)]),
    ),
    mujoco.mjtJoint.mjJNT_SLIDE: (
        jp.concatenate([-jp.inf * jp.ones(1)]),
        jp.concatenate([jp.inf * jp.ones(1)]),
    ),
    mujoco.mjtJoint.mjJNT_HINGE: (
        jp.concatenate([-2 * jp.pi * jp.ones(1)]),
        jp.concatenate([2 * jp.pi * jp.ones(1)]),
    ),
}


def _align_joint_dims(types, ranges, names):
    """Creates bounds and joint names aligned with qpos dimensions."""
    lb = []
    ub = []
    part_names = []
    for type, range, name in zip(types, ranges, names):
        dims = _MUJOCO_JOINT_TYPE_DIMS[type]
        # Set inf bounds for freejoint
        if type == mujoco.mjtJoint.mjJNT_FREE:
            lb.append(_MUJOCO_JOINT_TYPE_UNCONSTRAINED[type][0])
            ub.append(_MUJOCO_JOINT_TYPE_UNCONSTRAINED[type][1])
            part_names += [name] * dims
        else:
            l, u = range
            if l == 0 and u == 0:  # default joint lims are 0 0, which is unconstrained
                l = _MUJOCO_JOINT_TYPE_UNCONSTRAINED[type][0]
                u = _MUJOCO_JOINT_TYPE_UNCONSTRAINED[type][1]
            lb.append(l * jp.ones(dims))
            ub.append(u * jp.ones(dims))
            part_names += [name] * dims

    return jp.minimum(jp.concatenate(lb), 0.0), jp.concatenate(ub), part_names


def _resolve_reg_gate(mj_model, spec, sigma_deg=25.0):
    """Gate the rest prior on how far a DRIVING joint is from its reference.

    `spec` maps {gated_joint: driver_joint}, e.g. wing_roll_left ->
    wing_yaw_left. The gate is exp(-0.5*(driver - driver_rest)^2 / sigma^2):
    full strength while the driver sits at rest (wing folded, where pulling the
    blade to rest is correct) and vanishing as it departs (wing extended, where
    roll/pitch legitimately leave rest).

    Without this the prior taxes the wrong fly. Measured at weight 1e-3: the
    folded fly (bout_00001 fly0) pays -0.21% wing residual while the SINGING
    male (bout_00003 fly1) pays +10.3%, because his extended wing is being
    pulled toward a folded-wing reference. The singer's wings are the ones
    carrying the courtship song, so that is the opposite of the trade we want.

    Indices are returned in HINGE space (qpos index - _FREE_JOINT_NDOF), which
    is what the SE3-root solver's JointVar uses.
    """
    if not spec:
        return None
    free_ndof = 7  # qpos entries of the free root joint
    n_hinges = int(mj_model.nq) - free_ndof

    def hinge_idx(name):
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if jid < 0:
            raise ValueError(f"JAXLS_REG_GATE names no such joint: {name!r}")
        return int(mj_model.jnt_qposadr[jid]) - free_ndof

    drivers, driver_ref, gate_of = [], [], np.full(n_hinges, -1, dtype=np.int32)
    for gated, driver in dict(spec).items():
        di = hinge_idx(driver)
        if di not in drivers:
            drivers.append(di)
            driver_ref.append(float(mj_model.qpos_spring[di + free_ndof]))
        gate_of[hinge_idx(gated)] = drivers.index(di)
    return dict(driver_hinge_idx=np.asarray(drivers, np.int32),
                driver_ref=np.asarray(driver_ref, np.float32),
                gate_of=gate_of,
                sigma=float(np.radians(sigma_deg)))


def _resolve_rest_prior(mj_model, spec):
    """Build (q_reg_weights, q_ref) for a pull toward the model's REST pose.

    `spec` is {joint_name: weight}. The reference is `mj_model.qpos_spring`,
    the model's own spring rest -- NOT zero. This matters: measured on
    bout_00001 fly0, the folded wing has yaw within 0.2 deg of its rest value
    while roll sits 15.8 deg and pitch 41.6 deg off rest, and those two
    deviations are what drive the blade through the abdomen (r=0.80 / 0.75
    against penetration depth, which reaches 0.046 -- 33x the 0.0013 grazing
    contact the model shows at its own rest pose). Three near-collinear wing
    keypoints fix where the wing POINTS but barely constrain rotation about
    that axis, so the blade orientation is free to drift into the body.

    Raises on an unknown joint name: a typo must not silently disable the prior.
    """
    if not spec:
        return None, None
    nq = int(mj_model.nq)
    w = np.zeros(nq, dtype=np.float32)
    unknown = []
    for name, val in dict(spec).items():
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if jid < 0:
            unknown.append(str(name))
            continue
        w[int(mj_model.jnt_qposadr[jid])] = float(val)
    if unknown:
        raise ValueError(
            f"JAXLS_Q_REG_TO_REST names no such joint(s): {sorted(unknown)}. "
            "Check the spelling against the body model XML."
        )
    return w, np.asarray(mj_model.qpos_spring, dtype=np.float32).copy()


def _resolve_smooth_q_mult(mj_model, spec):
    """Map a {joint_name: multiplier} spec to a per-qpos multiplier array.

    Used for the wing blade-roll prior: roll is near-unobservable from three
    near-collinear wing keypoints, so it wanders on measurement noise. Damping
    that one DOF's frame-to-frame change leaves the wing-direction DOFs (which
    carry the ~193 Hz song) at the global smoothness weight.

    Raises on an unknown joint name rather than silently ignoring it -- a typo
    would make the prior a no-op that still looks like it ran.
    """
    if not spec:
        return None
    mult = np.ones(int(mj_model.nq), dtype=np.float32)
    unknown = []
    for name, m in dict(spec).items():
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if jid < 0:
            unknown.append(str(name))
            continue
        mult[int(mj_model.jnt_qposadr[jid])] = float(m)
    if unknown:
        raise ValueError(
            f"JAXLS_SMOOTH_Q_MULT names no such joint(s): {sorted(unknown)}. "
            "Check the spelling against the body model XML."
        )
    return mult


class Stac:
    """Main class with key functionality for skeletal registration and rendering."""

    def __init__(self, xml_path: str, cfg: DictConfig, kp_names: List[str]):
        """Init stac class, taking values from configs and creating values needed for stac.

        Args:
            xml_path (str): Path to model MJCF.
            cfg (DictConfig): Configs for this run.
            kp_names (List[str]): Ordered list of mocap keypoint names.
        """
        self.cfg = cfg
        self._kp_names = kp_names
        self._spec = mujoco.MjSpec.from_file(str(xml_path))
        self.stac_core_obj = None

        (
            self._mj_model,
            self._body_site_idxs,
            self._is_regularized,
        ) = self._create_body_sites(self._spec)

        self._body_names = [
            self._mj_model.body(i).name for i in range(self._mj_model.nbody)
        ]

        if "ROOT_OPTIMIZATION_KEYPOINT" in self.cfg.model:
            self._root_kp_idx = self._kp_names.index(
                self.cfg.model.ROOT_OPTIMIZATION_KEYPOINT
            )
        else:
            self._root_kp_idx = -1

        # Set up bounds and part_names based on joint ranges, taking into account the dimensionality of parameters
        joint_names = [self._mj_model.joint(i).name for i in range(self._mj_model.njnt)]
        self._lb, self._ub, self._part_names = _align_joint_dims(
            self._mj_model.jnt_type, self._mj_model.jnt_range, joint_names
        )

        self._indiv_parts = self.part_opt_setup()

        # Generate boolean flags for keypoints included in trunk optimization.
        self._trunk_kps = jp.array(
            [n in self.cfg.model.TRUNK_OPTIMIZATION_KEYPOINTS for n in kp_names],
        )

        # Build per-keypoint weights for IK loss (repeated 3x for x,y,z).
        # Falls back to uniform 1.0 if KEYPOINT_WEIGHTS not in config.
        kp_weight_cfg = self.cfg.model.get("KEYPOINT_WEIGHTS", {})
        self._kp_weights = jp.repeat(
            jp.array([float(kp_weight_cfg.get(n, 1.0)) for n in kp_names]),
            3,
        )

        # Per-qpos L2 regularization weights toward rest (q=0), from
        # JOINT_REG_WEIGHTS (joint name -> coefficient). Maps each named joint to
        # its qpos index; unspecified joints stay 0 (no regularization). Used by
        # pose_optimization to e.g. pin the under-constrained wing roll flat.
        q_reg = np.zeros(self._mj_model.nq)
        for jname, w in (self.cfg.model.get("JOINT_REG_WEIGHTS", {}) or {}).items():
            jid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                q_reg[self._mj_model.jnt_qposadr[jid]] = float(w)
            else:
                print(f"Warning: JOINT_REG_WEIGHTS joint '{jname}' not in model")
        self._q_reg_weights = jp.array(q_reg)

        self._mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[cfg.stac.mujoco.solver.lower()]

        self._mj_model.opt.timestep = cfg.stac.mujoco.dt
        self._mj_model.opt.iterations = cfg.stac.mujoco.iterations
        self._mj_model.opt.ls_iterations = cfg.stac.mujoco.ls_iterations

        # Runs faster on GPU with this
        self._mj_model.opt.jacobian = 0  # dense
        self._freejoint = bool(self._mj_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE)
        self._slidejoint = bool(
            self._mj_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_SLIDE
        )
        self._fixed = not (self._freejoint or self._slidejoint)

        # Create Stac_Core object
        self.stac_core_obj = stac_core.StacCore(
            self.cfg.model.FTOL, self.cfg.model.N_ITER_Q, self.cfg.model.N_ITER_M,
            stepsize_q=getattr(self.cfg.model, "STEPSIZE_Q", 0.0),
            use_jaxls=getattr(self.cfg.model, "USE_JAXLS", False),
            jaxls_lambda_initial=getattr(self.cfg.model, "JAXLS_LAMBDA_INITIAL", 1.0),
            jaxls_robust_delta=getattr(self.cfg.model, "JAXLS_ROBUST_DELTA", None),
            jaxls_smooth_q_mult=_resolve_smooth_q_mult(
                self._mj_model, getattr(self.cfg.model, "JAXLS_SMOOTH_Q_MULT", None)),
            **dict(zip(("q_reg_weights", "jaxls_q_ref"), _resolve_rest_prior(
                self._mj_model, getattr(self.cfg.model, "JAXLS_Q_REG_TO_REST", None)))),
            jaxls_reg_gate=_resolve_reg_gate(
                self._mj_model, getattr(self.cfg.model, "JAXLS_REG_GATE", None),
                float(getattr(self.cfg.model, "JAXLS_REG_GATE_SIGMA_DEG", 25.0))),
            smooth_weight=getattr(self.cfg.model, "JAXLS_SMOOTH_WEIGHT", 0.0),
            jaxls_linear_solver=getattr(self.cfg.model, "JAXLS_LINEAR_SOLVER", "auto"),
            jaxls_chunk_size=getattr(self.cfg.model, "JAXLS_CHUNK_SIZE", 100),
            use_se3_root=getattr(self.cfg.model, "JAXLS_USE_SE3_ROOT", True),
            jaxls_cost_tolerance=getattr(
                self.cfg.model, "JAXLS_COST_TOLERANCE", 1e-5),
            jaxls_gradient_tolerance=getattr(
                self.cfg.model, "JAXLS_GRADIENT_TOLERANCE", 1e-8),
            jaxls_parameter_tolerance=getattr(
                self.cfg.model, "JAXLS_PARAMETER_TOLERANCE", 1e-10),
        )
        # Expose root keypoint index on stac_core_obj for jaxls warm-starting
        self.stac_core_obj._root_kp_idx = self._root_kp_idx
        # Expose joint regularization weights so pose_optimization picks them up
        # without threading through every call site.
        self.stac_core_obj._q_reg_weights = self._q_reg_weights

        # Parse orientation keypoints for per-frame quaternion warm-start
        orient_cfg = self.cfg.model.get("JAXLS_ORIENTATION_KEYPOINTS", {})
        if orient_cfg and len(orient_cfg) >= 3:
            try:
                rear_idx = self._kp_names.index(orient_cfg["rear"])
                left_idx = self._kp_names.index(orient_cfg["left"])
                right_idx = self._kp_names.index(orient_cfg["right"])
                front_idx = self._kp_names.index(orient_cfg["front"]) if "front" in orient_cfg else -1
                self.stac_core_obj._orientation_kp_indices = (rear_idx, left_idx, right_idx, front_idx)
            except (ValueError, KeyError) as exc:
                print(f"Warning: JAXLS_ORIENTATION_KEYPOINTS: {exc} — orientation warm-start disabled")
                self.stac_core_obj._orientation_kp_indices = None
        else:
            self.stac_core_obj._orientation_kp_indices = None

    def part_opt_setup(self):
        """Set up the lists of indices for part optimization."""

        def get_part_ids(parts: List) -> jp.ndarray:
            """Get the part ids for a given list of parts."""
            return jp.array(
                [any(part in name for part in parts) for name in self._part_names]
            )

        if "INDIVIDUAL_PART_OPTIMIZATION" not in self.cfg.model:
            indiv_parts = []
        else:
            indiv_parts = jp.array(
                [
                    get_part_ids(parts)
                    for parts in self.cfg.model.INDIVIDUAL_PART_OPTIMIZATION.values()
                ]
            )

        return indiv_parts

    def _create_body_sites(self, spec: mujoco.MjSpec):
        """Create body site elements using dmcontrol mjcf for each keypoint.

        Args:
            spec (mujoco.MjSpec):

        Returns:
            mujoco.Model, list of marker site indices, boolean mask for offset
            regularization, lists for part names and body names.
        """
        for key, v in self.cfg.model.KEYPOINT_MODEL_PAIRS.items():
            parent = spec.body(v)
            pos = self.cfg.model.KEYPOINT_INITIAL_OFFSETS[key]

            if isinstance(pos, str):
                pos = [float(p) for p in pos.split(" ")]

            parent.add_site(
                name=key,
                size=[0.005, 0.005, 0.005],
                rgba=(0, 0, 0, 0.8),
                pos=pos,
                group=3,
            )

        rescale.dm_scale_spec(spec, self.cfg.model.SCALE_FACTOR)

        # Subject-specific per-segment calibration: morph the model's segments to
        # this fly's proportions (legs/head/abdomen) before compiling, so IK is
        # solved on a model that physically matches the animal. Scales come from
        # preprocessing (h5 info -> cfg.model.SEGMENT_SCALES). No-op if absent.
        seg_scales = self.cfg.model.get("SEGMENT_SCALES", None)
        if seg_scales:
            entries = seg_scales.values() if hasattr(seg_scales, "values") else seg_scales
            seg_list = [{"geom_body": e["geom_body"], "length_body": e["length_body"],
                         "scale": float(e["scale"]),
                         "scale_sites_on_body": e.get("scale_sites_on_body", "")}
                        for e in entries]
            rescale.rescale_per_segment(spec, seg_list)  # in place on self._spec
            print(f"[calibration] morphed {len(seg_list)} body segments to subject proportions")

        model = self._spec.compile()

        site_index_map = {
            site_name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            for site_name in self.cfg.model.KEYPOINT_MODEL_PAIRS.keys()
        }

        # Define which offsets to regularize
        is_regularized = []
        for k in site_index_map.keys():
            if any(n == k for n in self.cfg.model.get("SITES_TO_REGULARIZE", [])):
                is_regularized.append(jp.array([1.0, 1.0, 1.0]))
            else:
                is_regularized.append(jp.array([0.0, 0.0, 0.0]))
        is_regularized = jp.stack(is_regularized).flatten()
        body_site_idxs = jp.array(list(site_index_map.values()))
        return (
            model,
            body_site_idxs,
            is_regularized,
        )

    def _get_error_stats(self, errors: list):
        """Compute error stats, ignoring NaN frames.

        NaN frames arise when the input keypoint data contains NaN
        (e.g. kp-wise masking from the preprocessing filters). We report
        nanmean/nanstd so one bad frame does not poison the summary, plus
        the NaN-frame count for visibility.
        """
        flattened_errors = np.array(errors).reshape(-1)
        n_total = flattened_errors.size
        n_nan = int(np.isnan(flattened_errors).sum())
        if n_nan:
            print(f"  [stac] {n_nan}/{n_total} frames had NaN error "
                  f"(ignored in mean/std)")
        mean = np.nanmean(flattened_errors) if n_nan < n_total else np.nan
        std = np.nanstd(flattened_errors) if n_nan < n_total else np.nan

        return flattened_errors, mean, std

    def fit_offsets(self, kp_data):
        """Alternate between pose and offset optimization for a set number of iterations.

        Args:
            kp_data (jp.ndarray): Mocap keypoints to fit to

        Returns:
            Dict: Output data packaged in a dictionary.
        """
        # Create mjx model and data
        mjx_model, mjx_data = utils.mjx_load(self._mj_model)

        # Get and set the offsets of the markers
        self._offsets = jp.copy(utils.get_site_pos(mjx_model, self._body_site_idxs))

        mjx_model = utils.set_site_pos(mjx_model, self._offsets, self._body_site_idxs)

        # Calculate initial xpos and such
        mjx_data = mjx.kinematics(mjx_model, mjx_data)
        mjx_data = mjx.com_pos(mjx_model, mjx_data)

        # Begin optimization steps
        # Skip root optimization if model is fixed (no free joint at root)
        if self._root_kp_idx == -1:
            print(
                "ROOT_OPTIMIZATION_KEYPOINT not specified, skipping Root Optimization."
            )
        elif not self._fixed:
            mjx_data = compute_stac.root_optimization(
                self.stac_core_obj,
                mjx_model,
                mjx_data,
                kp_data,
                self._root_kp_idx,
                self._lb,
                self._ub,
                self._body_site_idxs,
                self._trunk_kps,
            )
        else:
            print(
                "ROOT_OPTIMIZATION_KEYPOINT specified but model has fixed root, skipping Root Optimization"
            )

        for n_iter in range(self.cfg.model.N_ITERS):
            print(f"Calibration iteration: {n_iter + 1}/{self.cfg.model.N_ITERS}")
            mjx_data, qposes, xposes, xquats, marker_sites, frame_time, frame_error = (
                compute_stac.pose_optimization(
                    self.stac_core_obj,
                    mjx_model,
                    mjx_data,
                    kp_data,
                    self._lb,
                    self._ub,
                    self._body_site_idxs,
                    self._indiv_parts,
                    self._kp_weights,
                )
            )

            flattened_errors, mean, std = self._get_error_stats(frame_error)
            # Print the results
            print(f"Mean: {mean}")
            print(f"Standard deviation: {std}")

            print("starting offset optimization", flush=True)
            mjx_model, mjx_data, self._offsets = compute_stac.offset_optimization(
                self.stac_core_obj,
                mjx_model,
                mjx_data,
                kp_data,
                self._offsets,
                qposes,
                self.cfg.model.N_SAMPLE_FRAMES,
                self._is_regularized,
                self._body_site_idxs,
                self.cfg.model.M_REG_COEF,
            )

        # Optimize the pose for the whole sequence
        print("Final pose optimization", flush=True)
        mjx_data, qposes, xposes, xquats, marker_sites, frame_time, frame_error = (
            compute_stac.pose_optimization(
                self.stac_core_obj,
                mjx_model,
                mjx_data,
                kp_data,
                self._lb,
                self._ub,
                self._body_site_idxs,
                self._indiv_parts,
                self._kp_weights,
            )
        )

        flattened_errors, mean, std = self._get_error_stats(frame_error)
        # Print the results
        print(f"Mean: {mean}")
        print(f"Standard deviation: {std}")
        return self._package_data(
            mjx_model,
            np.array(qposes),
            np.array(xposes),
            np.array(xquats),
            np.array(marker_sites),
            np.array(kp_data),
        )

    def ik_only(self, kp_data, offsets):
        """Do only inverse kinematics (no fitting) on motion capture data.

            ik_only is a stand-alone inverse kinematics step to be used after the marker offsets
            have been determined by fit_offsets(). This is most useful when it is desired or necessary
            to run the fit on a different data set than was used during fit. (Otherwise, the output of fit_offsets()
            will contain identical data.)

        Args:
            mj_model (mujoco.Model): Physics model.
            kp_data (jp.ndarray): Keypoint data in meters (batch_size, n_frames, 3, n_keypoints).
                Keypoint order must match the order in the skeleton file.
            offsets (jp.ndarray): offsets loaded from offset.p after fit()
        """
        # Create batches of kp_data
        batched_kp_data = utils.batch_kp_data(
            kp_data,
            self.cfg.stac.n_frames_per_clip,
            continuous=self.cfg.stac.continuous,
        )

        # Create mjx model and data
        mjx_model, mjx_data = utils.mjx_load(self._mj_model)

        def mjx_setup(kp_data, mj_model):
            """Create mjxmodel and mjxdata and set offet.

            Args:
                kp_data (_type_): _description_

            Returns:
                _type_: _description_
            """
            # Create mjx model and data
            mjx_model, mjx_data = utils.mjx_load(mj_model)

            # Set the offsets.
            mjx_model = utils.set_site_pos(mjx_model, offsets, self._body_site_idxs)

            # forward is used to calculate xpos and such
            mjx_data = mjx.kinematics(mjx_model, mjx_data)
            mjx_data = mjx.com_pos(mjx_model, mjx_data)

            return mjx_model, mjx_data

        mjx_model, mjx_data = jax.vmap(mjx_setup, in_axes=(0, None))(
            batched_kp_data, self._mj_model
        )

        # q_phase - root
        if self._root_kp_idx == -1:
            print(
                "Missing or invalid ROOT_OPTIMIZATION_KEYPOINT, skipping root_optimization()"
            )
        elif self._mj_model.jnt_type[0] in (
            mujoco.mjtJoint.mjJNT_FREE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            vmap_root_opt = jax.vmap(
                compute_stac.root_optimization,
                in_axes=(None, 0, 0, 0, None, None, None, None, None),
            )
            mjx_data = vmap_root_opt(
                self.stac_core_obj,
                mjx_model,
                mjx_data,
                batched_kp_data,
                self._root_kp_idx,
                self._lb,
                self._ub,
                self._body_site_idxs,
                self._trunk_kps,
            )
        else:
            print(
                "ROOT_OPTIMIZATION_KEYPOINT specified but model has fixed root, skipping root_optimization()"
            )

        # q_phase - pose
        if self.stac_core_obj._use_jaxls:
            # jaxls uses Python-loop chunking internally, so process clips
            # sequentially instead of vmapping (which would multiply memory).
            import time as _time
            n_clips = batched_kp_data.shape[0]
            results = []
            _t0_all = _time.time()
            for i in range(n_clips):
                _t0 = _time.time()
                print(f"Clip {i+1}/{n_clips} \n", end="", flush=True)
                result = compute_stac.pose_optimization(
                    self.stac_core_obj,
                    jax.tree.map(lambda x: x[i], mjx_model),
                    jax.tree.map(lambda x: x[i], mjx_data),
                    batched_kp_data[i],
                    self._lb,
                    self._ub,
                    self._body_site_idxs,
                    self._indiv_parts,
                    self._kp_weights,
                )
                results.append(result)
                _elapsed = _time.time() - _t0
                _total = _time.time() - _t0_all
                print(f" ({_elapsed:.1f}s, total {_total:.1f}s)", flush=True)
            # Stack results: each is (mjx_data, qposes, xposes, xquats, marker_sites, frame_time, frame_error)
            mjx_data = jax.tree.map(lambda *xs: jp.stack(xs), *(r[0] for r in results))
            qposes = jp.stack([r[1] for r in results])
            xposes = jp.stack([r[2] for r in results])
            xquats = jp.stack([r[3] for r in results])
            marker_sites = jp.stack([r[4] for r in results])
            frame_time = []
            frame_error = jp.stack([r[6] for r in results])
        else:
            vmap_pose_opt = jax.vmap(
                compute_stac.pose_optimization,
                in_axes=(None, 0, 0, 0, None, None, None, None, None),
            )
            mjx_data, qposes, xposes, xquats, marker_sites, frame_time, frame_error = (
                vmap_pose_opt(
                    self.stac_core_obj,
                    mjx_model,
                    mjx_data,
                    batched_kp_data,
                    self._lb,
                    self._ub,
                    self._body_site_idxs,
                    self._indiv_parts,
                    self._kp_weights,
                )
            )

        flattened_errors, mean, std = self._get_error_stats(frame_error)
        # Print the results
        print(f"Mean: {mean}")
        print(f"Standard deviation: {std}")

        return self._package_data(
            mjx_model,
            np.array(qposes),
            np.array(xposes),
            np.array(xquats),
            np.array(marker_sites),
            np.array(batched_kp_data),
            batched=True,
        )

    def _package_data(
        self, mjx_model, qposes, xposes, xquats, marker_sites, kp_data, batched=False
    ):
        """Extract pose, offsets, data, and all parameters.

        marker_sites is the marker positions for each frame--the rodent model's kp_data equivalent
        """
        if batched:
            # prepare batched data to be packaged
            get_batch_offsets = jax.vmap(utils.get_site_pos, in_axes=(0, None))
            offsets = get_batch_offsets(mjx_model, self._body_site_idxs)[0]
            qposes = qposes.reshape(-1, qposes.shape[-1])
            xposes = xposes.reshape(-1, *xposes.shape[2:])
            xquats = xquats.reshape(-1, *xquats.shape[2:])
            marker_sites = marker_sites.reshape(-1, *marker_sites.shape[2:])
        else:
            offsets = self._offsets.reshape((-1, 3))

        offsets = np.array(offsets)
        kp_data = kp_data.reshape(-1, kp_data.shape[-1])

        return io.StacData(
            qpos=qposes,
            xpos=xposes,
            xquat=xquats,
            marker_sites=marker_sites,
            offsets=offsets,
            names_qpos=self._part_names,
            names_xpos=self._body_names,
            kp_data=kp_data,
            kp_names=self._kp_names,
        )

    def _create_render_sites(self):
        """Create sites for keypoints (used for rendering only).

        Returns:
            (mujoco.Model, List, List): Mj_model for rendering, list of keypoint site indices, and list of body site indices
        """
        keypoint_sites = []
        keypoint_site_names = []
        # set up keypoint rendering by adding the kp sites to the root body
        for id, name in enumerate(self.cfg.model.KEYPOINT_MODEL_PAIRS):
            start = (np.random.rand(3) - 0.5) * 0.001
            rgba = self.cfg.model.KEYPOINT_COLOR_PAIRS[name]

            if isinstance(rgba, str):
                rgba = [float(c) for c in rgba.split(" ")]
            site_name = name + "_kp"
            keypoint_site_names.append(site_name)
            site = self._spec.worldbody.add_site(
                name=site_name,
                size=[0.005, 0.005, 0.005],
                rgba=rgba,
                pos=start,
                group=2,
            )
            keypoint_sites.append(site)

        model = self._spec.compile()

        # Combine the two lists of site names and create the index map
        site_index_map = {
            site.name: i
            for i, site in enumerate(self._spec.sites)
            if site.name
            in list(self.cfg.model.KEYPOINT_MODEL_PAIRS.keys()) + keypoint_site_names
        }
        body_site_idxs = [
            site_index_map[n] for n in self.cfg.model.KEYPOINT_MODEL_PAIRS.keys()
        ]
        keypoint_site_idxs = [site_index_map[n] for n in keypoint_site_names]

        self._body_site_idxs = body_site_idxs
        self._keypoint_site_idxs = keypoint_site_idxs
        return (deepcopy(model), body_site_idxs, keypoint_site_idxs)

    def render(
        self,
        qposes: jp.ndarray,
        kp_data: jp.ndarray,
        offsets: jp.ndarray,
        n_frames: int,
        save_path: Union[str, Path],
        start_frame: int = 0,
        camera: Union[int, str] = 0,
        height: int = 1200,
        width: int = 1920,
        show_marker_error: bool = False,
    ):
        """Creates rendering using the instantiated model, given the user's qposes and kp_data.

        Args:
            qposes (jp.ndarray): Set of model joint angles corresponding to kp_data.
            kp_data (jp.ndarray): Set of motion capture keypoints.
            offsets (jp.ndarray): array of marker offsets.
            n_frames (int): Number of frames to render.
            save_path (str): Path to save.
            start_frame (int, optional): Starting frame of qposes/kp_data to render at. Defaults to 0.
            camera (Union[int, str], optional): Mujoco camera name. Defaults to 0.
            height (int, optional): Height in pixels. Defaults to 1200.
            width (int, optional): Width in pixels. Defaults to 1920.
            show_marker_error (bool, optional): Show distance between marker and keypoint. Defaults to False.

        Raises:
            ValueError: qposes and kp_data must have same length (shape[0])
            ValueError: start_frame must be a non-negative value and within the length of kp_data/qposes
            ValueError: start_frame + n_frames must be within the length of kp_data/qposes

        Returns:
            List: List of rendered frames.
        """
        if qposes.shape[0] != kp_data.shape[0]:
            raise ValueError(
                f"Length of qposes ({qposes.shape[0]}) is not equal to the length of kp_data({kp_data.shape[0]})"
            )
        if start_frame < 0 or start_frame > kp_data.shape[0]:
            raise ValueError(
                f"start_frame ({start_frame}) must be non-negative and less than the length of kp_data ({kp_data.shape[0]})"
            )
        if start_frame + n_frames > kp_data.shape[0]:
            raise ValueError(
                f"start_frame + n_frames ({start_frame} + {n_frames}) must be less than the length of given qposes and kp_data ({kp_data.shape[0]})"
            )

        render_mj_model, body_site_idxs, keypoint_site_idxs = (
            self._create_render_sites()
        )

        # Add body sites for new offsets
        for (key, v), pos in zip(
            self.cfg.model.KEYPOINT_MODEL_PAIRS.items(), offsets.reshape((-1, 3))
        ):
            parent = self._spec.body(v)
            parent.add_site(
                name=key + "_new",
                size=[0.005, 0.005, 0.005],
                rgba=[0, 0, 0, 1],
                pos=pos,
                group=2,
            )

        # Tendons from new marker sites to kp
        if show_marker_error:
            for key, v in self.cfg.model.KEYPOINT_MODEL_PAIRS.items():
                tendon = self._spec.add_tendon(
                    name=key + "-" + v,
                    width=0.001,
                    rgba=[1, 0, 0, 1],  # Red (mujoco rgba is 0-1)
                    limited=0,
                )
                tendon.wrap_site(key + "_kp")
                tendon.wrap_site(key + "_new")

        render_mj_model = deepcopy(self._spec.compile())

        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[1] = 1
        scene_option.geomgroup[2] = 1
        scene_option.sitegroup[:] = [1, 1, 1, 1, 1, 0]

        # scene_option.sitegroup[2] = 1
        # scene_option.sitegroup[3] = 1
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True
        scene_option.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        scene_option.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
        scene_option.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = True
        scene_option.flags[mujoco.mjtRndFlag.mjRND_FOG] = True
        mj_data = mujoco.MjData(render_mj_model)

        mujoco.mj_kinematics(render_mj_model, mj_data)

        renderer = mujoco.Renderer(render_mj_model, height=height, width=width)

        # Slice kp_data to match qposes length
        kp_data = kp_data[: qposes.shape[0]]

        # Slice arrays to be the range that is being rendered
        kp_data = kp_data[start_frame : start_frame + n_frames]
        qposes = qposes[start_frame : start_frame + n_frames]

        frames = []
        # Render while stepping using mujoco
        with imageio.get_writer(save_path, fps=self.cfg.model.RENDER_FPS) as video:
            for qpos, kps in tqdm(zip(qposes, kp_data)):
                # Set keypoints--they're in cartesian space, but since they're attached to the worldbody they're the same as offsets
                render_mj_model.site_pos[keypoint_site_idxs] = np.reshape(kps, (-1, 3))
                mj_data.qpos = qpos

                mujoco.mj_fwdPosition(render_mj_model, mj_data)

                renderer.update_scene(mj_data, camera=camera, scene_option=scene_option)
                pixels = renderer.render()
                video.append_data(pixels)
                frames.append(pixels)

        return frames
