"""The vmapped independent-frame path of JaxlsBatchSolver (smooth_weight == 0).

Why: with no smoothness term the batch problem is block-diagonal and jaxls'
conjugate-gradient path stalls on it (a 300-frame solve that takes 188 s with
smoothing did not finish in 66 min without it, 2026-09-04), while dense
Cholesky on the whole block-diagonal matrix wastes ~T**2 work. The fix builds
the T=1 problem and vmaps its solve over frames. These tests pin that the
path is taken automatically at smooth_weight == 0, recovers known poses, and
agrees with the batch path on the same frames -- on a tiny synthetic MJCF, on
CPU, in seconds.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from stac_mjx.stac_core_jaxls import JaxlsBatchSolver

_XML = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="root" pos="0 0 0">
      <freejoint/>
      <geom type="capsule" size="0.02 0.05" fromto="0 0 0 0.1 0 0"/>
      <site name="s0" pos="0 0.02 0" size="0.005"/>
      <body name="link1" pos="0.1 0 0">
        <joint name="j1" type="hinge" axis="0 0 1" range="-1.5 1.5"/>
        <geom type="capsule" size="0.015 0.04" fromto="0 0 0 0.08 0 0"/>
        <site name="s1" pos="0.08 0.01 0" size="0.005"/>
        <body name="link2" pos="0.08 0 0">
          <joint name="j2" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
          <geom type="capsule" size="0.01 0.03" fromto="0 0 0 0.06 0 0"/>
          <site name="s2" pos="0.06 0 0.01" size="0.005"/>
          <site name="s3" pos="0.03 -0.01 0" size="0.005"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _setup():
    mj = mujoco.MjModel.from_xml_string(_XML)
    mjx_model = mjx.put_model(mj)
    mjx_data = mjx.put_data(mj, mujoco.MjData(mj))
    site_idxs = jnp.arange(mj.nsite)
    nq = mj.nq
    lb = jnp.concatenate([jnp.full(7, -jnp.inf), jnp.array([-1.5, -1.5])])
    ub = jnp.concatenate([jnp.full(7, jnp.inf), jnp.array([1.5, 1.5])])
    return mj, mjx_model, mjx_data, site_idxs, nq, lb, ub


def _fk_sites(mj, q):
    d = mujoco.MjData(mj)
    d.qpos[:] = np.asarray(q)
    mujoco.mj_kinematics(mj, d)
    return d.site_xpos.copy().reshape(-1)


def _targets(mj, T, seed=0):
    rng = np.random.default_rng(seed)
    qs, kps = [], []
    for _ in range(T):
        q = np.zeros(mj.nq)
        q[:3] = rng.normal(scale=0.05, size=3)
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax); ang = rng.uniform(-0.6, 0.6)
        q[3] = np.cos(ang / 2); q[4:7] = np.sin(ang / 2) * ax
        q[7:] = rng.uniform(-1.0, 1.0, size=mj.nq - 7)
        qs.append(q); kps.append(_fk_sites(mj, q))
    return np.stack(qs), np.stack(kps)


def _solve(solver, q_init, kp, mjx_model, mjx_data, site_idxs, lb, ub):
    nq = q_init.shape[1]
    return np.asarray(solver.solve_trajectory(
        q_init=jnp.asarray(q_init), mjx_model=mjx_model, mjx_data_template=mjx_data,
        kp_data=jnp.asarray(kp), qs_to_opt=jnp.ones(nq, bool),
        kps_to_opt=jnp.ones(kp.shape[1]), lb=lb, ub=ub, site_idxs=site_idxs,
        q_reg_weights=jnp.zeros(nq)))


def test_independent_path_is_automatic_at_zero_smoothing_and_off_otherwise():
    assert JaxlsBatchSolver(smooth_weight=0.0)._use_independent(5) is True
    assert JaxlsBatchSolver(smooth_weight=0.0)._use_independent(1) is False
    assert JaxlsBatchSolver(smooth_weight=0.1)._use_independent(5) is False
    assert JaxlsBatchSolver(smooth_weight=0.1, independent_frames=True)._use_independent(5) is True
    assert JaxlsBatchSolver(smooth_weight=0.0, independent_frames=False)._use_independent(5) is False


@pytest.mark.parametrize("T,batch", [(4, 256), (5, 2)])
def test_independent_solve_recovers_known_poses_including_ragged_batches(T, batch):
    mj, mjx_model, mjx_data, site_idxs, nq, lb, ub = _setup()
    q_true, kp = _targets(mj, T)
    q_init = np.tile(np.array([0, 0, 0, 1, 0, 0, 0] + [0.0] * (nq - 7)), (T, 1))
    solver = JaxlsBatchSolver(n_iter=200, smooth_weight=0.0, independent_batch=batch,
                              cost_tolerance=1e-12, gradient_tolerance=1e-12,
                              parameter_tolerance=1e-14)
    q = _solve(solver, q_init, kp, mjx_model, mjx_data, site_idxs, lb, ub)
    assert q.shape == (T, nq)
    resid = np.array([np.linalg.norm(_fk_sites(mj, q[t]) - kp[t]) for t in range(T)])
    assert resid.max() < 1e-3, resid
    np.testing.assert_allclose(q[:, 7:], q_true[:, 7:], atol=2e-2)
    # exactly one compiled batch function per (problem, batch) key
    assert len(solver._independent_fns) == 1


def test_independent_and_batch_paths_agree_on_the_same_frames():
    mj, mjx_model, mjx_data, site_idxs, nq, lb, ub = _setup()
    T = 3
    q_true, kp = _targets(mj, T, seed=1)
    q_init = q_true + np.random.default_rng(2).normal(scale=0.05, size=q_true.shape)
    q_init[:, 3:7] /= np.linalg.norm(q_init[:, 3:7], axis=1, keepdims=True)
    common = dict(n_iter=200, smooth_weight=0.0, cost_tolerance=1e-12,
                  gradient_tolerance=1e-12, parameter_tolerance=1e-14)
    q_ind = _solve(JaxlsBatchSolver(**common), q_init, kp, mjx_model, mjx_data, site_idxs, lb, ub)
    q_bat = _solve(JaxlsBatchSolver(independent_frames=False, linear_solver="dense_cholesky", **common),
                   q_init, kp, mjx_model, mjx_data, site_idxs, lb, ub)
    np.testing.assert_allclose(q_ind[:, 7:], q_bat[:, 7:], atol=1e-2)
    np.testing.assert_allclose(q_ind[:, :3], q_bat[:, :3], atol=1e-3)


def test_problem_cache_distinguishes_dof_and_keypoint_masks():
    """Regression: a T=1 root-only problem (root_optimization) must not be
    reused for a T=1 full-body solve. Before the key carried qs_to_opt /
    kps_to_opt the independent-frame path inherited root_optimization's frozen
    hinges and fit only the trunk (mean frame error 0.81 vs 0.0036)."""
    mj, mjx_model, mjx_data, site_idxs, nq, lb, ub = _setup()
    q_true, kp = _targets(mj, 1, seed=3)
    q_init = np.array([[0, 0, 0, 1, 0, 0, 0] + [0.0] * (nq - 7)])
    solver = JaxlsBatchSolver(n_iter=200, smooth_weight=0.0, cost_tolerance=1e-12,
                              gradient_tolerance=1e-12, parameter_tolerance=1e-14)
    root_only = jnp.array([True] * 7 + [False] * (nq - 7))
    common = dict(mjx_model=mjx_model, mjx_data_template=mjx_data, kp_data=jnp.asarray(kp),
                  lb=lb, ub=ub, site_idxs=site_idxs, q_reg_weights=jnp.zeros(nq))
    q_root = np.asarray(solver.solve_trajectory(q_init=jnp.asarray(q_init), qs_to_opt=root_only,
                                                kps_to_opt=jnp.ones(kp.shape[1]), **common))
    assert np.allclose(q_root[0, 7:], 0.0)                 # hinges frozen, as asked
    q_full = np.asarray(solver.solve_trajectory(q_init=jnp.asarray(q_init), qs_to_opt=jnp.ones(nq, bool),
                                                kps_to_opt=jnp.ones(kp.shape[1]), **common))
    assert len(solver._cache) == 2                          # two distinct problems
    np.testing.assert_allclose(q_full[0, 7:], q_true[0, 7:], atol=2e-2)   # hinges DID move


def test_multistart_matches_single_solves_per_start():
    """solve_trajectory_multistart vmaps the whole-clip solve over S starts;
    each slice must equal the corresponding single solve (same problem, same
    settings), including with the smoothness term on."""
    mj, mjx_model, mjx_data, site_idxs, nq, lb, ub = _setup()
    T = 4
    q_true, kp = _targets(mj, T, seed=5)
    q0 = np.tile(np.array([0, 0, 0, 1, 0, 0, 0] + [0.0] * (nq - 7)), (T, 1))
    q1 = q0.copy(); q1[:, 7] = 0.8                                 # a different hinge start
    solver = JaxlsBatchSolver(n_iter=100, smooth_weight=0.05, cost_tolerance=1e-10,
                              gradient_tolerance=1e-10, parameter_tolerance=1e-12)
    common = dict(mjx_model=mjx_model, mjx_data_template=mjx_data, kp_data=jnp.asarray(kp),
                  qs_to_opt=jnp.ones(nq, bool), kps_to_opt=jnp.ones(kp.shape[1]), lb=lb, ub=ub,
                  site_idxs=site_idxs, q_reg_weights=jnp.zeros(nq))
    multi = np.asarray(solver.solve_trajectory_multistart(q_inits=jnp.asarray(np.stack([q0, q1])), **common))
    assert multi.shape == (2, T, nq)
    for i, qi in enumerate((q0, q1)):
        single = np.asarray(solver.solve_trajectory(q_init=jnp.asarray(qi), **common))
        np.testing.assert_allclose(multi[i], single, atol=1e-4)
