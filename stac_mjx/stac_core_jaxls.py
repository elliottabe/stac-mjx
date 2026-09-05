"""Jaxls (Levenberg-Marquardt) batch trajectory IK solver for stac-mjx.

Solves ALL frames of a clip simultaneously in a single jaxls LeastSquaresProblem,
with optional smoothness coupling between adjacent timesteps — the same approach
pyroki uses for URDF robots, but using MJX forward kinematics so MJCF models
work natively without any format conversion.

Design (use_se3_root=True, default)
------------------------------------
Variables per timestep:
  SE3Var(jnp.arange(T))    — jaxlie.SE3 root pose (tangent_dim=6, no quat drift)
  JointVar(jnp.arange(T))  — shape (nq-7,) hinge angles
  KpVar(jnp.arange(T))     — shape (n_kp*3,) keypoint observations (held fixed)

Costs (vmapped internally by jaxls):
  marker_tracking_cost  : ||FK(root⊕joints) - kp||² * kp_weights   (T instances)
  regularization_cost   : ||sqrt(reg_w) * joints||²                  (T instances, if any reg>0)
  limit_constraint      : lb[7:] ≤ joints ≤ ub[7:]  (aug. Lagrangian)(T instances)
  smoothness_cost       : ||[SE3_log_diff; joint_diff]||² * smooth_w (T-1 instances, optional)

The SE3Var uses jaxlie's right-plus retraction (R_new = R_old @ exp(δ)), keeping
the quaternion on SO(3) without any post-solve normalization hack.

Fallback (use_se3_root=False)
-------------------------------
Uses a single flat QVar(nq) + Euclidean updates. Needed when some root DOFs are
frozen (qs_to_opt[:7] has False entries). Adds a quat normalization post-solve.

Independent frames (smooth_weight == 0)
----------------------------------------
Without the smoothness term the frames do not interact, and neither jaxls
linear solver handles that shape well (dense_cholesky wastes ~T**2, CG stalls;
measured 2026-09-04). `solve_trajectory` then builds the T=1 problem and vmaps
its solve over frames in batches (`independent_batch`), each frame with a
dense per-frame factorisation and its own LM termination. Force with
`independent_frames=True/False`.

Linear solver auto-selection
------------------------------
  T * nq < 5000  → dense_cholesky (faster for small problems)
  T * nq ≥ 5000  → conjugate_gradient (exploits block-diagonal Jacobian sparsity)
Override via linear_solver="dense_cholesky" or "conjugate_gradient".

The problem graph is analyzed once per unique key and cached. Subsequent clips of
the same shape reuse the cached analysis and only call the JIT-compiled solve.

Usage
-----
    from stac_mjx.stac_core_jaxls import JaxlsBatchSolver

    solver = JaxlsBatchSolver(n_iter=50, smooth_weight=0.1)

    # Solve entire clip at once
    qposes = solver.solve_trajectory(
        q_init=jnp.tile(q0, (T, 1)),   # (T, nq) warm start
        mjx_model=mjx_model,
        mjx_data_template=mjx_data,
        kp_data=kp_data,               # (T, n_kp*3)
        qs_to_opt=qs_to_opt,
        kps_to_opt=kps_to_opt,
        lb=lb, ub=ub,
        site_idxs=site_idxs,
        q_reg_weights=q_reg_weights,
    )
    # qposes.shape == (T, nq)

    # Also works per-frame (for compatibility with root_optimization path)
    q_opt = solver.run_frame(q0, mjx_model, mjx_data, kp_data_frame, ...)
"""

import jax
import numpy as np
import jax.numpy as jnp
import jaxlie
import jaxls

from stac_mjx import utils

# Number of free-joint DOFs in MuJoCo (3 translation + 4 quaternion).
_FREE_JOINT_NDOF = 7


# ---------------------------------------------------------------------------
# Internal state cache key
# ---------------------------------------------------------------------------

class _AnalyzedProblem:
    """Holds one analyzed jaxls problem and its variable classes.

    Supports two modes:
      SE3 mode (use_se3_root=True):   SE3Var + JointVar + KpVar
      Flat mode (use_se3_root=False): QVar  + KpVar
    """
    def __init__(self, analyzed, KpVar, *,
                 QVar=None, SE3Var=None, JointVar=None):
        self.analyzed = analyzed
        self.KpVar = KpVar
        # Flat mode
        self.QVar = QVar
        # SE3 mode
        self.SE3Var = SE3Var
        self.JointVar = JointVar

    @property
    def use_se3_root(self) -> bool:
        return self.SE3Var is not None


# ---------------------------------------------------------------------------
# Public solver class
# ---------------------------------------------------------------------------

def _robust_reweight(resid, delta):
    """IRLS reweighting that turns jaxls' L2 sum into a HUBER loss.

    jaxls minimises sum(residual^2), and a single badly-placed keypoint therefore
    pulls on the whole pose in proportion to its error SQUARED. Measured on
    Session0/2025_10_20_13_20_04 bout_00028 fly1, the femur-tibia keypoints sit
    ~1.5x too far from the trochanter in the data (a keypoint-localisation error,
    not a real leg), and the resulting bias is 68% of that bout's total fit
    residual -- it drags the body pose rather than staying local.

    Scaling each residual by sqrt(w) with w = min(1, delta/|r|) makes the
    effective loss quadratic within `delta` and LINEAR beyond it, so an outlier
    contributes a bounded pull. `stop_gradient` on the weight is what makes this
    IRLS rather than a different (and non-convex) objective: LM still sees a
    plain reweighted least-squares problem, with the weights held fixed within
    a step and refreshed at the next one.

    Weighting is per KEYPOINT (the 3-vector norm), not per coordinate, so a
    keypoint is down-weighted as a whole and its error direction is preserved.

    `delta` is in model length units. None/<=0 returns the residual unchanged,
    so the default behaviour is bit-identical to before this function existed.
    """
    if delta is None or delta <= 0:
        return resid
    r3 = resid.reshape(-1, 3)
    n = jnp.linalg.norm(r3, axis=1, keepdims=True)
    w = jnp.minimum(1.0, delta / jnp.maximum(n, 1e-12))
    return (r3 * jnp.sqrt(jax.lax.stop_gradient(w))).reshape(-1)


class JaxlsBatchSolver:
    """Batch trajectory IK solver using jaxls Levenberg-Marquardt.

    Solves all T frames of a clip as a single least-squares problem,
    optionally coupling adjacent frames via a smoothness cost.

    Args:
        n_iter: Maximum LM iterations. Default 50.
        linear_solver: "auto" (default) picks dense_cholesky for T*nq < 5000 and
            conjugate_gradient otherwise. Explicit "dense_cholesky" or
            "conjugate_gradient" override the auto rule.
            dense_cholesky is O(n³) in tangent_dim — fast for small problems.
            conjugate_gradient exploits the block-diagonal Jacobian sparsity that
            arises when smooth_weight=0 (each frame is independent).
        lambda_initial: Initial LM damping. Default 1.0.
        smooth_weight: Weight for the smoothness cost. 0.0 = per-frame equivalent.
        use_se3_root: If True (default), represent the free-joint root pose as a
            jaxls.SE3Var so LM updates stay on the SO(3) manifold — no quaternion
            drift, no post-solve normalization needed. Requires qs_to_opt[:7] all
            True. Set False only when some root DOFs are frozen.
    """

    # Threshold below which dense_cholesky is faster than conjugate_gradient.
    _DENSE_THRESHOLD = 5000  # T * tangent_dim

    def __init__(
        self,
        n_iter: int = 50,
        linear_solver: str = "auto",
        lambda_initial: float = 1.0,
        smooth_weight: float = 0.0,
        use_se3_root: bool = True,
        cost_tolerance: float = 1e-5,
        gradient_tolerance: float = 1e-8,
        parameter_tolerance: float = 1e-10,
        robust_delta: float | None = None,
        smooth_q_mult: jnp.ndarray | None = None,
        q_ref: jnp.ndarray | None = None,
        reg_gate: dict | None = None,
        independent_frames: bool | None = None,
        independent_batch: int = 256,
    ):
        self.n_iter = n_iter
        self.linear_solver = linear_solver
        # Independent-frame path (see _solve_independent). None = automatic:
        # used whenever smooth_weight == 0 and T > 1, because then the batch
        # problem is block-diagonal and jaxls' conjugate-gradient path stalls
        # on it (measured 2026-09-04: a 300-frame solve that takes 188 s with
        # smoothing did not finish in 66 min without it). True/False force it.
        self.independent_frames = independent_frames
        self.independent_batch = int(independent_batch)
        self._independent_fns: dict[tuple, object] = {}
        self.lambda_initial = lambda_initial
        self.smooth_weight = smooth_weight
        self.use_se3_root = use_se3_root
        # Huber threshold for the marker cost, in model length units.
        # None (default) = plain squared error, unchanged behaviour.
        self.robust_delta = robust_delta
        # Per-DOF multiplier on the temporal smoothness cost, in qpos layout
        # (length nq). None (default) = uniform 1.0, unchanged behaviour.
        # Only the hinge block [_FREE_JOINT_NDOF:] is used by the SE3 path;
        # the 6D root tangent is always smoothed at the base weight.
        # Motivating case: wing blade-roll is near-unobservable from three
        # near-collinear wing keypoints, so it wanders on measurement noise.
        # Smoothing that one DOF harder damps the wander without touching the
        # wing-direction DOFs that carry the ~193 Hz song.
        self.smooth_q_mult = (
            None if smooth_q_mult is None else jnp.asarray(smooth_q_mult, dtype=jnp.float32)
        )
        # Reference pose for the joint regularizer, in qpos layout (length nq).
        # None (default) = regularize toward q=0, unchanged behaviour.
        # For the wing blade this MUST be the model's springref, not zero: the
        # folded-wing rest pose has wing_yaw at +85.94 deg and wing_pitch at
        # -57.30 deg, and pulling those toward 0 would unfold every wing.
        self.q_ref = None if q_ref is None else jnp.asarray(q_ref, dtype=jnp.float32)
        # Optional gate that switches the joint regularizer OFF as a driving DOF
        # leaves its reference. Built for the wing: pulling blade roll/pitch to
        # the FOLDED rest pose is right while the wing is folded and wrong while
        # it is extended, and an ungated prior costs the singing male ~10% wing
        # residual (measured, bout_00003 fly1) to fix a folded-wing defect.
        # The gate is stop_gradient'd, so it acts as a per-iteration constant.
        self.reg_gate = reg_gate
        # jaxls termination tolerances, explicit rather than left on jaxls'
        # library defaults. cost stays AT jaxls' default (1e-5); gradient and
        # parameter are TIGHTENED (jaxls: 1e-4 / 1e-6) because weakly
        # conditioned DOFs (e.g. wing blade-roll, Jacobian column ~14x weaker
        # than the strong wing DOFs) terminate far short of their optimum on
        # the loose defaults. See the parent repo's
        # docs/benchmark/2026-08-13-stac-weak-dof-convergence/notes.md.
        # NOTE: FTOL is deliberately NOT mapped here — it is 5e-3, 500x looser
        # than cost_tolerance's 1e-5, and would fire before the gradient
        # criterion ever engaged. FTOL governs the ProjectedGradient path only.
        self.cost_tolerance = cost_tolerance
        self.gradient_tolerance = gradient_tolerance
        self.parameter_tolerance = parameter_tolerance
        # Cache analyzed problems keyed by (T, nq, n_kp_dim, has_smooth, has_reg, se3)
        self._cache: dict[tuple, _AnalyzedProblem] = {}

    def _pick_linear_solver(self, T: int, tangent_dim: int) -> str:
        """Auto-select linear solver based on problem size."""
        if self.linear_solver != "auto":
            return self.linear_solver
        return (
            "dense_cholesky"
            if T * tangent_dim < self._DENSE_THRESHOLD
            else "conjugate_gradient"
        )

    # ------------------------------------------------------------------
    # Problem construction
    # ------------------------------------------------------------------

    def _build_se3(
        self,
        T: int,
        nq: int,
        n_kp_dim: int,
        mjx_model,
        mjx_data_template,
        qs_to_opt: jnp.ndarray,
        kps_to_opt: jnp.ndarray,
        lb: jnp.ndarray,
        ub: jnp.ndarray,
        site_idxs: jnp.ndarray,
        q_reg_weights: jnp.ndarray,
        smooth_weight: float,
        smooth_q_mult: jnp.ndarray | None = None,
        q_ref: jnp.ndarray | None = None,
        reg_gate: dict | None = None,
    ) -> _AnalyzedProblem:
        """Build the jaxls problem using SE3Var (root) + JointVar (hinges).

        The SE3Var uses jaxlie's right-plus retraction, keeping the quaternion
        on SO(3) without any post-solve normalization. The JointVar covers the
        (nq - 7) hinge DOFs with box constraints.

        Assumes qs_to_opt[:7] are all True (root fully optimized).
        """
        n_hinges = nq - _FREE_JOINT_NDOF
        dummy_joints = jnp.zeros((n_hinges,))
        dummy_kp = jnp.zeros((n_kp_dim,))

        # ---- Variable classes ----
        class SE3Var(
            jaxls.Var[jaxlie.SE3],
            default_factory=jaxlie.SE3.identity,
            retract_fn=jaxlie.manifold.rplus,
            tangent_dim=6,
        ): ...
        class JointVar(jaxls.Var[jnp.ndarray], default_factory=lambda: dummy_joints): ...
        class KpVar(jaxls.Var[jnp.ndarray], default_factory=lambda: dummy_kp): ...

        # Batched instances: one per timestep
        root_all  = SE3Var(jnp.arange(T))
        joint_all = JointVar(jnp.arange(T))
        kp_all    = KpVar(jnp.arange(T))

        costs: list[jaxls.Cost] = []
        robust_delta = self.robust_delta      # closed over by marker_cost below

        # ---- Marker tracking cost ----
        @jaxls.Cost.factory
        def marker_cost(
            var_values: jaxls.VarValues,
            root_var: SE3Var,
            joint_var: JointVar,
            kp_var: KpVar,
        ) -> jnp.ndarray:
            T_root  = var_values[root_var]   # jaxlie.SE3
            joints  = var_values[joint_var]  # (n_hinges,)
            kp      = jax.lax.stop_gradient(var_values[kp_var])  # (n_kp_dim,)

            # Reconstruct MuJoCo qpos: [x,y,z, qw,qx,qy,qz, hinges...]
            xyz  = T_root.translation()      # (3,)
            wxyz = T_root.rotation().wxyz    # (4,)  w,x,y,z
            q    = jnp.concatenate([xyz, wxyz, joints])  # (nq,)

            full_q = jnp.where(qs_to_opt, q, mjx_data_template.qpos)
            data = mjx_data_template.replace(qpos=full_q)
            data = utils.kinematics(mjx_model, data)
            data = utils.com_pos(mjx_model, data)
            markers = utils.get_site_xpos(data, site_idxs).flatten()
            # NaN-safe residual: a masked/occluded keypoint coordinate is NaN.
            # Sanitize to a finite value BEFORE subtracting (NaN*0 == NaN), then
            # zero its residual via the finite mask so it neither contributes nor
            # leaks a gradient. Without this, one NaN frame makes the whole-clip
            # cost non-finite and LM rejects every step (clip frozen at init).
            finite = jnp.isfinite(kp)
            kp_clean = jnp.where(finite, kp, 0.0)
            resid = (kp_clean - markers) * kps_to_opt * finite
            return _robust_reweight(resid, robust_delta)

        costs.append(marker_cost(root_all, joint_all, kp_all))

        # ---- Joint regularization cost (hinges only; root typically unreg.) ----
        if jnp.any(q_reg_weights[_FREE_JOINT_NDOF:] > 0):
            hinge_regs = q_reg_weights[_FREE_JOINT_NDOF:]
            hinge_opt  = qs_to_opt[_FREE_JOINT_NDOF:]
            hinge_ref  = (jnp.zeros_like(hinge_regs) if q_ref is None
                          else q_ref[_FREE_JOINT_NDOF:])
            if reg_gate is None:
                gate_src = gate_ref = gate_of = None
                gate_sigma = 0.0
            else:
                gate_src   = jnp.asarray(reg_gate["driver_hinge_idx"], jnp.int32)
                gate_ref   = jnp.asarray(reg_gate["driver_ref"], jnp.float32)
                gate_of    = jnp.asarray(reg_gate["gate_of"], jnp.int32)
                gate_sigma = float(reg_gate["sigma"])

            @jaxls.Cost.factory
            def reg_cost(
                var_values: jaxls.VarValues,
                joint_var: JointVar,
            ) -> jnp.ndarray:
                j = var_values[joint_var]
                w = hinge_regs
                if gate_of is not None:
                    # Gaussian in the driver DOF's deviation from its reference:
                    # 1.0 at the reference (wing folded), falling to 0 as it
                    # departs (wing extended). stop_gradient keeps this a
                    # weight, not a term the solver can game by moving the driver.
                    dev = jax.lax.stop_gradient(j[gate_src] - gate_ref)
                    g = jnp.exp(-0.5 * (dev / gate_sigma) ** 2)
                    g_full = jnp.where(gate_of >= 0, g[jnp.clip(gate_of, 0)], 1.0)
                    w = w * g_full
                return jnp.sqrt(w * hinge_opt) * (j - hinge_ref)

            costs.append(reg_cost(joint_all))

        # ---- Hinge limit constraint (SE3 root is unconstrained by design) ----
        hinge_lb = lb[_FREE_JOINT_NDOF:]
        hinge_ub = ub[_FREE_JOINT_NDOF:]

        @jaxls.Cost.factory(kind="constraint_leq_zero")
        def limit_cost(
            var_values: jaxls.VarValues,
            joint_var: JointVar,
        ) -> jnp.ndarray:
            j = var_values[joint_var]
            return jnp.concatenate([hinge_lb - j, j - hinge_ub])

        costs.append(limit_cost(joint_all))

        # ---- Smoothness: SE3 log-diff + joint diff ----
        if smooth_weight > 0.0 and T > 1:
            # Per-hinge smoothness multiplier (1.0 everywhere unless a prior
            # asks for a specific DOF to be damped harder -- see smooth_q_mult).
            hinge_smooth_mult = (
                1.0 if smooth_q_mult is None else smooth_q_mult[_FREE_JOINT_NDOF:]
            )

            @jaxls.Cost.factory
            def smoothness_cost(
                var_values: jaxls.VarValues,
                root_curr: SE3Var,
                root_prev: SE3Var,
                joint_curr: JointVar,
                joint_prev: JointVar,
            ) -> jnp.ndarray:
                # SE3 geodesic difference in tangent space (6D)
                root_diff  = (var_values[root_prev].inverse() @ var_values[root_curr]).log()
                joint_diff = (var_values[joint_curr] - var_values[joint_prev]) * hinge_smooth_mult
                return jnp.concatenate([root_diff, joint_diff]) * smooth_weight

            costs.append(smoothness_cost(
                SE3Var(jnp.arange(1, T)),      # root_curr
                SE3Var(jnp.arange(0, T-1)),    # root_prev
                JointVar(jnp.arange(1, T)),    # joint_curr
                JointVar(jnp.arange(0, T-1)), # joint_prev
            ))

        variables = [root_all, joint_all, kp_all]

        analyzed = (
            jaxls.LeastSquaresProblem(costs=costs, variables=variables)
            .analyze()
        )

        return _AnalyzedProblem(
            analyzed=analyzed, KpVar=KpVar,
            SE3Var=SE3Var, JointVar=JointVar,
        )

    def _build(
        self,
        T: int,
        nq: int,
        n_kp_dim: int,
        mjx_model,
        mjx_data_template,
        qs_to_opt: jnp.ndarray,
        kps_to_opt: jnp.ndarray,
        lb: jnp.ndarray,
        ub: jnp.ndarray,
        site_idxs: jnp.ndarray,
        q_reg_weights: jnp.ndarray,
        smooth_weight: float,
        smooth_q_mult: jnp.ndarray | None = None,
        q_ref: jnp.ndarray | None = None,
        reg_gate: dict | None = None,
    ) -> _AnalyzedProblem:
        """Build and analyze the jaxls problem for a given (T, nq, n_kp_dim) shape.

        This is called once per unique combination and cached.
        """
        dummy_q = jnp.zeros((nq,))
        dummy_kp = jnp.zeros((n_kp_dim,))

        # ---- Variable classes ----
        class QVar(jaxls.Var[jnp.ndarray], default_factory=lambda: dummy_q): ...
        class KpVar(jaxls.Var[jnp.ndarray], default_factory=lambda: dummy_kp): ...

        # Batched instances: one per timestep
        q_all = QVar(jnp.arange(T))
        kp_all = KpVar(jnp.arange(T))

        costs: list[jaxls.Cost] = []
        robust_delta = self.robust_delta      # closed over by marker_cost below

        # ---- Marker tracking cost ----
        # jaxls vmaps this over T via the batch dimension of q_all and kp_all.
        # mjx_model and site_idxs are closed over (constant across frames).
        @jaxls.Cost.factory
        def marker_cost(
            var_values: jaxls.VarValues,
            q_var: QVar,
            kp_var: KpVar,
        ) -> jnp.ndarray:
            q = var_values[q_var]    # (nq,) — one timestep (jaxls vmaps over T)
            # stop_gradient: kp is an observation, not a parameter to optimize.
            # Zero Jacobian w.r.t. kp_var → LM never updates it, only q is moved.
            kp = jax.lax.stop_gradient(var_values[kp_var])  # (n_kp_dim,)
            # qs_to_opt selects which joints are optimized; rest stay at q
            # (for the batch solver we typically pass qs_to_opt=ones so full_q==q)
            full_q = jnp.where(qs_to_opt, q, mjx_data_template.qpos)
            data = mjx_data_template.replace(qpos=full_q)
            data = utils.kinematics(mjx_model, data)
            data = utils.com_pos(mjx_model, data)
            markers = utils.get_site_xpos(data, site_idxs).flatten()
            # NaN-safe residual (see _build_se3.marker_cost): zero out masked/NaN
            # keypoint coordinates so one NaN frame can't freeze the whole clip.
            finite = jnp.isfinite(kp)
            kp_clean = jnp.where(finite, kp, 0.0)
            resid = (kp_clean - markers) * kps_to_opt * finite
            return _robust_reweight(resid, robust_delta)

        costs.append(marker_cost(q_all, kp_all))

        # ---- Joint regularization cost ----
        if jnp.any(q_reg_weights > 0):
            q_ref_full = jnp.zeros_like(q_reg_weights) if q_ref is None else q_ref

            @jaxls.Cost.factory
            def reg_cost(
                var_values: jaxls.VarValues,
                q_var: QVar,
            ) -> jnp.ndarray:
                q = var_values[q_var]
                return jnp.sqrt(q_reg_weights * qs_to_opt) * (q - q_ref_full)

            costs.append(reg_cost(q_all))

        # ---- Joint limit constraint (augmented Lagrangian) ----
        @jaxls.Cost.factory(kind="constraint_leq_zero")
        def limit_cost(
            var_values: jaxls.VarValues,
            q_var: QVar,
        ) -> jnp.ndarray:
            q = var_values[q_var]
            return jnp.concatenate([lb - q, q - ub])

        costs.append(limit_cost(q_all))

        # ---- Smoothness cost: ||q[t] - q[t-1]||² * weight ----
        if smooth_weight > 0.0 and T > 1:
            q_smooth_mult = 1.0 if smooth_q_mult is None else smooth_q_mult

            @jaxls.Cost.factory
            def smoothness_cost(
                var_values: jaxls.VarValues,
                q_curr: QVar,
                q_prev: QVar,
            ) -> jnp.ndarray:
                return (var_values[q_curr] - var_values[q_prev]) * q_smooth_mult * smooth_weight

            costs.append(smoothness_cost(
                QVar(jnp.arange(1, T)),      # q[1..T-1]
                QVar(jnp.arange(0, T - 1)),  # q[0..T-2]
            ))

        # KpVar must be in variables so jaxls can build the sparsity pattern.
        # stop_gradient in marker_cost gives it a zero Jacobian — LM sees no
        # gradient direction for kp and leaves it fixed. kp_data is supplied
        # per-clip via initial_vals in solve_trajectory().
        variables = [q_all, kp_all]

        analyzed = (
            jaxls.LeastSquaresProblem(costs=costs, variables=variables)
            .analyze()
        )

        return _AnalyzedProblem(analyzed=analyzed, KpVar=KpVar, QVar=QVar)

    def _get_analyzed(
        self,
        T: int,
        mjx_model,
        mjx_data_template,
        kp_data: jnp.ndarray,
        qs_to_opt: jnp.ndarray,
        kps_to_opt: jnp.ndarray,
        lb: jnp.ndarray,
        ub: jnp.ndarray,
        site_idxs: jnp.ndarray,
        q_reg_weights: jnp.ndarray,
    ) -> _AnalyzedProblem:
        nq = int(mjx_model.nq)
        n_kp_dim = int(kp_data.shape[-1]) if kp_data.ndim > 1 else int(kp_data.shape[0])
        has_reg = bool(jnp.any(q_reg_weights > 0))
        has_smooth = self.smooth_weight > 0.0
        # The multiplier is baked into the cost closure, so it is part of the key.
        mult_key = (
            None if self.smooth_q_mult is None
            else tuple(round(float(v), 6) for v in self.smooth_q_mult)
        )
        ref_key = (None if self.q_ref is None
                   else tuple(round(float(v), 6) for v in self.q_ref))
        def _hashable(v):
            try:
                return tuple(round(float(x), 6) for x in v)
            except TypeError:
                return round(float(v), 6)

        gate_key = (None if self.reg_gate is None
                    else tuple(sorted((k, _hashable(v)) for k, v in self.reg_gate.items())))
        # qs_to_opt / kps_to_opt / lb / ub / site_idxs are baked into the cost
        # closures, so they MUST be part of the key. Before 2026-09-05 they were
        # not: root_optimization's T=1 problem (root DOFs only, trunk keypoints
        # only) was then reused by the first full-body T=1 solve of the
        # independent-frame path, which froze every hinge and fit the root to
        # the trunk alone (mean frame error 0.81 vs 0.0036; body 180 deg off).
        def _mask_key(v):
            return tuple(bool(x) for x in np.asarray(v).ravel())
        key = (T, nq, n_kp_dim, has_reg, has_smooth, self.use_se3_root,
               mult_key, ref_key, gate_key,
               _mask_key(qs_to_opt), _hashable(kps_to_opt),
               _hashable(lb), _hashable(ub),
               tuple(int(x) for x in np.asarray(site_idxs).ravel()))

        if key not in self._cache:
            builder = self._build_se3 if self.use_se3_root else self._build
            self._cache[key] = builder(
                T=T,
                nq=nq,
                n_kp_dim=n_kp_dim,
                mjx_model=mjx_model,
                mjx_data_template=mjx_data_template,
                qs_to_opt=qs_to_opt,
                kps_to_opt=kps_to_opt,
                lb=lb,
                ub=ub,
                site_idxs=site_idxs,
                q_reg_weights=q_reg_weights,
                smooth_weight=self.smooth_weight,
                smooth_q_mult=self.smooth_q_mult,
                q_ref=self.q_ref,
                reg_gate=self.reg_gate,
            )
        return self._cache[key]

    # ------------------------------------------------------------------
    # Public API: batch trajectory solve
    # ------------------------------------------------------------------

    def solve_trajectory(
        self,
        q_init: jnp.ndarray,
        mjx_model,
        mjx_data_template,
        kp_data: jnp.ndarray,
        qs_to_opt: jnp.ndarray,
        kps_to_opt: jnp.ndarray,
        lb: jnp.ndarray,
        ub: jnp.ndarray,
        site_idxs: jnp.ndarray,
        q_reg_weights: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve IK for an entire clip simultaneously.

        Solves all T frames as one jaxls LeastSquaresProblem. Adjacent frames
        are coupled via the smoothness cost when smooth_weight > 0.

        Args:
            q_init: Initial joint config for all frames, shape (T, nq).
                Warm-start from previous solution or tile a single q0.
            mjx_model: MJX model (constant).
            mjx_data_template: MJX data used as FK template (qpos overwritten).
            kp_data: Observed keypoints, shape (T, n_kp*3) or (T, n_kp, 3).
            qs_to_opt: Boolean mask (nq,) selecting joints to optimize.
            kps_to_opt: Per-coordinate weight mask (n_kp*3,).
            lb: Joint lower bounds (nq,).
            ub: Joint upper bounds (nq,).
            site_idxs: Site indices for marker positions.
            q_reg_weights: Per-joint L2 regularization weights (nq,).

        Returns:
            Optimized joint angles, shape (T, nq).
        """
        # Flatten kp_data to (T, n_kp*3)
        if kp_data.ndim == 3:
            kp_data = kp_data.reshape(kp_data.shape[0], -1)
        T = q_init.shape[0]

        if self._use_independent(T):
            prob1 = self._get_analyzed(
                1, mjx_model, mjx_data_template,
                kp_data, qs_to_opt, kps_to_opt, lb, ub, site_idxs, q_reg_weights,
            )
            return self._solve_independent(prob1, q_init, kp_data)

        prob = self._get_analyzed(
            T, mjx_model, mjx_data_template,
            kp_data, qs_to_opt, kps_to_opt, lb, ub, site_idxs, q_reg_weights,
        )
        KpVar = prob.KpVar

        if prob.use_se3_root:
            return self._solve_se3(prob, T, q_init, kp_data)
        else:
            return self._solve_flat(prob, T, q_init, kp_data)

    # ------------------------------------------------------------------
    # Independent frames: vmapped single-frame LM (smooth_weight == 0)
    # ------------------------------------------------------------------

    def _use_independent(self, T: int) -> bool:
        if T <= 1:
            return False
        if self.independent_frames is None:
            return self.smooth_weight <= 0.0
        return bool(self.independent_frames)

    def _termination(self) -> "jaxls.TerminationConfig":
        return jaxls.TerminationConfig(
            max_iterations=self.n_iter,
            cost_tolerance=self.cost_tolerance,
            gradient_tolerance=self.gradient_tolerance,
            parameter_tolerance=self.parameter_tolerance,
        )

    def _solve_independent(
        self,
        prob: _AnalyzedProblem,
        q_init: jnp.ndarray,
        kp_data: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve T frames as T independent single-frame problems, vmapped.

        With no smoothness term the batch Hessian is block-diagonal: one
        (6 + n_hinges)-dim block per frame. jaxls has no independent-problem
        mode -- dense_cholesky factorises the whole block-diagonal matrix
        (cost ~ T**3 wasted) and conjugate_gradient stalls on it -- so, per
        jaxls' own guidance for many identical small problems, this builds the
        T=1 problem once and `jax.vmap`s its `solve` over frames. Every frame
        gets a dense per-frame factorisation and its OWN LM termination, so a
        hard frame cannot hold the others hostage. Frames are processed in
        batches of `independent_batch` (one compile per batch shape; the
        ragged tail is padded by repeating its last frame and trimmed).
        """
        T = int(q_init.shape[0])
        B = max(1, min(self.independent_batch, T))
        key = (id(prob), B, bool(prob.use_se3_root))
        if key not in self._independent_fns:
            self._independent_fns[key] = jax.jit(jax.vmap(
                self._single_frame_solver(prob)))
        solve_batch = self._independent_fns[key]

        out = []
        for s in range(0, T, B):
            qi, ki = q_init[s:s + B], kp_data[s:s + B]
            n = int(qi.shape[0])
            if n < B:                       # pad the ragged tail to the batch shape
                qi = jnp.concatenate([qi, jnp.repeat(qi[-1:], B - n, axis=0)])
                ki = jnp.concatenate([ki, jnp.repeat(ki[-1:], B - n, axis=0)])
            out.append(solve_batch(qi, ki)[:n])
        return jnp.concatenate(out, axis=0)

    def _single_frame_solver(self, prob: _AnalyzedProblem):
        """(q_row (nq,), kp_row (n_kp_dim,)) -> q_opt (nq,) for the T=1 problem.

        Same costs, constraints, trust region and termination as the batch
        path; only the linear solver is fixed to dense_cholesky (the per-frame
        system is tiny)."""
        KpVar = prob.KpVar
        trust = jaxls.TrustRegionConfig(lambda_initial=self.lambda_initial)
        term = self._termination()
        one = jnp.arange(1)

        if prob.use_se3_root:
            SE3Var, JointVar = prob.SE3Var, prob.JointVar

            def solve_one(q_row, kp_row):
                wxyz = q_row[3:7]
                qn = jnp.linalg.norm(wxyz)
                wxyz = wxyz / jnp.where(qn > 0, qn, 1.0)
                root = jaxlie.SE3.from_rotation_and_translation(
                    jaxlie.SO3(wxyz=wxyz[None]), q_row[None, :3])
                sol = prob.analyzed.solve(
                    verbose=False, linear_solver="dense_cholesky",
                    trust_region=trust, termination=term,
                    initial_vals=jaxls.VarValues.make([
                        SE3Var(one).with_value(root),
                        JointVar(one).with_value(q_row[None, _FREE_JOINT_NDOF:]),
                        KpVar(one).with_value(kp_row[None]),
                    ]))
                r = sol[SE3Var(one)]
                j = sol[JointVar(one)]
                return jnp.concatenate([r.translation()[0], r.rotation().wxyz[0], j[0]])
        else:
            QVar = prob.QVar

            def solve_one(q_row, kp_row):
                sol = prob.analyzed.solve(
                    verbose=False, linear_solver="dense_cholesky",
                    trust_region=trust, termination=term,
                    initial_vals=jaxls.VarValues.make([
                        QVar(one).with_value(q_row[None]),
                        KpVar(one).with_value(kp_row[None]),
                    ]))
                q = sol[QVar(one)][0]
                quat = q[3:7]
                qn = jnp.linalg.norm(quat)
                return q.at[3:7].set(quat / jnp.where(qn > 0, qn, 1.0))
        return solve_one

    def _solve_se3(
        self,
        prob: _AnalyzedProblem,
        T: int,
        q_init: jnp.ndarray,
        kp_data: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve trajectory using the SE3Var + JointVar representation."""
        SE3Var  = prob.SE3Var
        JointVar = prob.JointVar
        KpVar   = prob.KpVar

        # Split q_init into root pose (SE3) + hinge angles.
        # MuJoCo free-joint layout: [x,y,z, qw,qx,qy,qz, hinges...]
        xyz_init    = q_init[:, :3]                      # (T, 3)
        wxyz_init   = q_init[:, 3:7]                     # (T, 4)
        hinges_init = q_init[:, _FREE_JOINT_NDOF:]       # (T, n_hinges)

        # Normalize quaternion before building SE3 (invalid quat → NaN gradients)
        qn = jnp.linalg.norm(wxyz_init, axis=-1, keepdims=True)
        wxyz_init = wxyz_init / jnp.where(qn > 0, qn, 1.0)

        # Build batched jaxlie.SE3 initial values
        roots_init = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(wxyz=wxyz_init),
            xyz_init,
        )  # batch shape (T,)

        # Pick linear solver based on problem size.
        # SE3 path: tangent_dim = 6 (root) + n_hinges per frame
        n_hinges   = q_init.shape[1] - _FREE_JOINT_NDOF
        tangent_dim = 6 + n_hinges
        linear_solver = self._pick_linear_solver(T, tangent_dim)

        sol = prob.analyzed.solve(
            verbose=False,
            linear_solver=linear_solver,
            trust_region=jaxls.TrustRegionConfig(lambda_initial=self.lambda_initial),
            termination=jaxls.TerminationConfig(
                max_iterations=self.n_iter,
                cost_tolerance=self.cost_tolerance,
                gradient_tolerance=self.gradient_tolerance,
                parameter_tolerance=self.parameter_tolerance,
            ),
            initial_vals=jaxls.VarValues.make([
                SE3Var(jnp.arange(T)).with_value(roots_init),
                JointVar(jnp.arange(T)).with_value(hinges_init),
                KpVar(jnp.arange(T)).with_value(kp_data),
            ]),
        )

        # Recombine SE3 + joints back into (T, nq) qpos array.
        sol_roots  = sol[SE3Var(jnp.arange(T))]        # SE3 batch (T,)
        sol_joints = sol[JointVar(jnp.arange(T))]      # (T, n_hinges)

        xyz_sol  = sol_roots.translation()              # (T, 3)
        wxyz_sol = sol_roots.rotation().wxyz            # (T, 4) — already on SO(3)

        return jnp.concatenate([xyz_sol, wxyz_sol, sol_joints], axis=-1)  # (T, nq)

    def _solve_flat(
        self,
        prob: _AnalyzedProblem,
        T: int,
        q_init: jnp.ndarray,
        kp_data: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve trajectory using the flat QVar representation."""
        QVar  = prob.QVar
        KpVar = prob.KpVar

        linear_solver = self._pick_linear_solver(T, q_init.shape[1])

        sol = prob.analyzed.solve(
            verbose=False,
            linear_solver=linear_solver,
            trust_region=jaxls.TrustRegionConfig(lambda_initial=self.lambda_initial),
            termination=jaxls.TerminationConfig(
                max_iterations=self.n_iter,
                cost_tolerance=self.cost_tolerance,
                gradient_tolerance=self.gradient_tolerance,
                parameter_tolerance=self.parameter_tolerance,
            ),
            initial_vals=jaxls.VarValues.make([
                QVar(jnp.arange(T)).with_value(q_init),
                KpVar(jnp.arange(T)).with_value(kp_data),
            ]),
        )

        qposes = sol[QVar(jnp.arange(T))]  # (T, nq)

        # Post-normalize quaternion: Euclidean LM updates can drift off SO(3).
        quat      = qposes[:, 3:7]
        quat_norm = jnp.linalg.norm(quat, axis=-1, keepdims=True)
        return qposes.at[:, 3:7].set(quat / jnp.where(quat_norm > 0, quat_norm, 1.0))

    # ------------------------------------------------------------------
    # Public API: single-frame solve (for root_optimization compatibility)
    # ------------------------------------------------------------------

    def run_frame(
        self,
        q0: jnp.ndarray,
        mjx_model,
        mjx_data,
        kp_data: jnp.ndarray,
        qs_to_opt: jnp.ndarray,
        kps_to_opt: jnp.ndarray,
        lb: jnp.ndarray,
        ub: jnp.ndarray,
        site_idxs: jnp.ndarray,
        q_reg_weights: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve IK for a single frame (T=1 batch).

        Used by root_optimization() and any other per-frame path.

        Returns:
            Optimized joint angles (nq,).
        """
        kp_flat = kp_data.flatten() if kp_data.ndim > 1 else kp_data
        q_init = q0[None]          # (1, nq)
        kp_batch = kp_flat[None]   # (1, n_kp*3)

        result = self.solve_trajectory(
            q_init=q_init,
            mjx_model=mjx_model,
            mjx_data_template=mjx_data,
            kp_data=kp_batch,
            qs_to_opt=qs_to_opt,
            kps_to_opt=kps_to_opt,
            lb=lb,
            ub=ub,
            site_idxs=site_idxs,
            q_reg_weights=q_reg_weights,
        )
        return result[0]  # (nq,)
