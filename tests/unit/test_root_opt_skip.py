"""The safety property behind `Stac.ik_only`'s root_optimization skip.

`root_optimization` solves a one-frame LM problem for the first `root_dims`
qpos entries and leaves every other entry alone. On the jaxls pose path
`_pose_optimization_jaxls` then builds its own per-frame warm start and
OVERWRITES exactly those entries -- xyz from the root keypoint, the quaternion
from the trunk-orientation keypoints. When it does, root_optimization's entire
output is written over before anything reads it, so skipping it cannot change
the fit. Measured cost of not skipping: 55.6 s per bout-fly (Session0
2025_10_20_13_20_04 bout 28 fly1, 2007 frames), nearly all of it XLA compile for
a T=1 problem that executes in 0.01 s.

These tests pin the property itself -- "the warm start does not depend on the
incoming root qpos" -- not just the boolean that decides the skip, because the
boolean is only correct while the property holds.
"""
import types

import jax.numpy as jp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from stac_mjx import compute_stac
from stac_mjx.stac import root_optimization_is_discarded


_XML = """
<mujoco>
  <worldbody>
    <body name="torso" pos="0 0 0">
      <freejoint name="root"/>
      <geom type="sphere" size="0.05"/>
      <site name="s_root" pos="0 0 0" size="0.005"/>
      <site name="s_rear" pos="-0.05 0 0" size="0.005"/>
      <site name="s_left" pos="0 0.05 0" size="0.005"/>
      <site name="s_right" pos="0 -0.05 0" size="0.005"/>
      <body name="limb" pos="0.05 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom type="capsule" size="0.01" fromto="0 0 0 0.05 0 0"/>
        <site name="s_tip" pos="0.05 0 0" size="0.005"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_N_KP = 5          # s_root, s_rear, s_left, s_right, s_tip
_ROOT_KP = 0
_ORIENT = (1, 2, 3, -1)     # rear, left, right, no front


def _model():
    mj = mujoco.MjModel.from_xml_string(_XML)
    mjx_model = mjx.put_model(mj)
    mjx_data = mjx.make_data(mj)
    site_idxs = jp.array([
        mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, n)
        for n in ("s_root", "s_rear", "s_left", "s_right", "s_tip")])
    return mj, mjx_model, mjx_data, site_idxs


def _core(recorder):
    """A StacCore stand-in whose 'solver' records the warm start it is handed."""
    solver = types.SimpleNamespace(
        solve_trajectory=lambda *, q_init, **kw: (recorder.append(np.asarray(q_init))
                                                  or q_init))
    return types.SimpleNamespace(
        _use_jaxls=True, _jaxls_chunk_size=0, _jaxls_solver=solver,
        _root_kp_idx=_ROOT_KP, _orientation_kp_indices=_ORIENT,
        _q_reg_weights=None)


def _run(mjx_model, mjx_data, kp, site_idxs, recorder):
    T = kp.shape[0]
    return compute_stac._pose_optimization_jaxls(
        _core(recorder), mjx_model, mjx_data, kp,
        lb=jp.full(mjx_model.nq, -10.0), ub=jp.full(mjx_model.nq, 10.0),
        site_idxs=site_idxs, kps_to_opt=jp.ones(_N_KP * 3),
        q_reg_weights=jp.zeros(mjx_model.nq), s=0.0, root_kp_idx=_ROOT_KP)


def _kp(T=4, seed=0):
    rng = np.random.default_rng(seed)
    return jp.asarray(rng.normal(scale=0.1, size=(T, _N_KP * 3)).astype(np.float32))


def test_pose_warm_start_ignores_every_root_dof_the_root_phase_writes():
    """THE PROPERTY. Perturb qpos[:7] -- what root_optimization is allowed to
    change -- and the warm start handed to the LM solver must not move."""
    mj, mjx_model, mjx_data, site_idxs = _model()
    kp = _kp()
    a, b = [], []
    _run(mjx_model, mjx_data, kp, site_idxs, a)
    # Stand in for "root_optimization ran": a different pose in the 7 root DOFs,
    # hinges untouched (root_optimization never writes them).
    perturbed = mjx_data.replace(qpos=mjx_data.qpos.at[:7].set(
        jp.array([0.3, -0.2, 0.7, 0.5, 0.5, 0.5, 0.5])))
    _run(mjx_model, perturbed, kp, site_idxs, b)
    assert len(a) == len(b) == 1
    assert np.array_equal(a[0], b[0]), (
        "the pose warm start moved with qpos[:7]; root_optimization's output is "
        f"NOT discarded. max|d| = {np.abs(a[0] - b[0]).max()}")


def test_a_hinge_change_DOES_reach_the_warm_start():
    """Specificity. The test above would pass vacuously if the warm start
    ignored mjx_data.qpos entirely -- it does not: hinges pass straight through,
    which is exactly why the skip is restricted to the root DOFs."""
    mj, mjx_model, mjx_data, site_idxs = _model()
    kp = _kp()
    a, b = [], []
    _run(mjx_model, mjx_data, kp, site_idxs, a)
    bent = mjx_data.replace(qpos=mjx_data.qpos.at[7].set(0.4))
    _run(mjx_model, bent, kp, site_idxs, b)
    assert not np.array_equal(a[0], b[0]), "a hinge change must reach the warm start"


def test_without_the_orientation_warm_start_the_quaternion_survives():
    """Why the skip condition names JAXLS_ORIENTATION_KEYPOINTS. With no
    orientation warm start, qpos[3:7] comes from mjx_data -- i.e. from
    root_optimization -- and skipping it WOULD change the fit."""
    mj, mjx_model, mjx_data, site_idxs = _model()
    kp = _kp()
    core_a = _core([]); core_a._orientation_kp_indices = None
    core_b = _core([]); core_b._orientation_kp_indices = None
    rec_a, rec_b = [], []
    core_a._jaxls_solver.solve_trajectory = (
        lambda *, q_init, **kw: (rec_a.append(np.asarray(q_init)) or q_init))
    core_b._jaxls_solver.solve_trajectory = (
        lambda *, q_init, **kw: (rec_b.append(np.asarray(q_init)) or q_init))
    turned = mjx_data.replace(qpos=mjx_data.qpos.at[3:7].set(
        jp.array([0.5, 0.5, 0.5, 0.5])))
    kwargs = dict(lb=jp.full(mjx_model.nq, -10.0), ub=jp.full(mjx_model.nq, 10.0),
                  site_idxs=site_idxs, kps_to_opt=jp.ones(_N_KP * 3),
                  q_reg_weights=jp.zeros(mjx_model.nq), s=0.0, root_kp_idx=_ROOT_KP)
    compute_stac._pose_optimization_jaxls(core_a, mjx_model, mjx_data, kp, **kwargs)
    compute_stac._pose_optimization_jaxls(core_b, mjx_model, turned, kp, **kwargs)
    assert not np.array_equal(rec_a[0][:, 3:7], rec_b[0][:, 3:7]), (
        "with no orientation warm start the quaternion must come from mjx_data")


@pytest.mark.parametrize("kwargs,expected", [
    (dict(use_jaxls=True, root_kp_idx=0, orientation_kp_indices=(1, 2, 3, -1),
          jnt_type0=mujoco.mjtJoint.mjJNT_FREE), True),
    # the ProjectedGradient path builds no per-frame warm start at all
    (dict(use_jaxls=False, root_kp_idx=0, orientation_kp_indices=(1, 2, 3, -1),
          jnt_type0=mujoco.mjtJoint.mjJNT_FREE), False),
    # no root keypoint -> root_optimization is not run anyway, and xyz is not
    # overwritten either
    (dict(use_jaxls=True, root_kp_idx=-1, orientation_kp_indices=(1, 2, 3, -1),
          jnt_type0=mujoco.mjtJoint.mjJNT_FREE), False),
    # no orientation warm start -> qpos[3:7] survives from root_optimization
    (dict(use_jaxls=True, root_kp_idx=0, orientation_kp_indices=None,
          jnt_type0=mujoco.mjtJoint.mjJNT_FREE), False),
    # SLIDE root: root_optimization writes 4 DOFs, the warm start overwrites 3
    (dict(use_jaxls=True, root_kp_idx=0, orientation_kp_indices=(1, 2, 3, -1),
          jnt_type0=mujoco.mjtJoint.mjJNT_SLIDE), False),
])
def test_skip_condition_truth_table(kwargs, expected):
    assert root_optimization_is_discarded(**kwargs) is expected
