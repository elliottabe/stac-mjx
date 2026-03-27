"""Jaxls (Levenberg-Marquardt) batch trajectory IK solver for stac-mjx.

Solves ALL frames of a clip simultaneously in a single jaxls LeastSquaresProblem,
with optional smoothness coupling between adjacent timesteps — the same approach
pyroki uses for URDF robots, but using MJX forward kinematics so MJCF models
work natively without any format conversion.

Design
------
Variables:
  QVar(jnp.arange(T))  — shape (T, nq), all joint configs for the clip
  KpVar(jnp.arange(T)) — shape (T, n_kp*3), keypoint observations (held fixed)

Costs (all vmapped internally by jaxls):
  marker_tracking_cost   : ||FK(q[t]) - kp[t]||² * kp_weights        (T instances)
  regularization_cost    : ||sqrt(reg_w) * q[t]||²                    (T instances)
  limit_constraint       : lb ≤ q[t] ≤ ub  (augmented Lagrangian)     (T instances)
  smoothness_cost        : ||q[t] - q[t-1]||² * smooth_w              (T-1 instances, optional)

The problem graph is analyzed once per unique (T, nq, n_kp) combination and
cached. Subsequent clips of the same shape reuse the cached analyzed problem,
only calling the JIT-compiled solve kernel.

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
import jax.numpy as jnp
import jaxls

from stac_mjx import utils


# ---------------------------------------------------------------------------
# Internal state cache key
# ---------------------------------------------------------------------------

class _AnalyzedProblem:
    """Holds one analyzed jaxls problem and its variable classes."""
    def __init__(self, analyzed, QVar, KpVar):
        self.analyzed = analyzed
        self.QVar = QVar
        self.KpVar = KpVar


# ---------------------------------------------------------------------------
# Public solver class
# ---------------------------------------------------------------------------

class JaxlsBatchSolver:
    """Batch trajectory IK solver using jaxls Levenberg-Marquardt.

    Solves all T frames of a clip as a single least-squares problem,
    optionally coupling adjacent frames via a smoothness cost.

    Args:
        n_iter: Maximum LM iterations. Default 50.
        linear_solver: "dense_cholesky" (fast, O(n³) in T×nq) for short clips,
            "conjugate_gradient" for longer clips where dense is too expensive.
            Rule of thumb: use dense_cholesky for T×nq < ~5000 (e.g. T=50, nq=93).
        lambda_initial: Initial LM damping. Default 1.0.
        smooth_weight: Weight for the smoothness cost ||q[t]-q[t-1]||².
            0.0 disables smoothness (per-frame equivalent). Start with 0.01–0.1.
    """

    def __init__(
        self,
        n_iter: int = 50,
        linear_solver: str = "dense_cholesky",
        lambda_initial: float = 1.0,
        smooth_weight: float = 0.0,
    ):
        self.n_iter = n_iter
        self.linear_solver = linear_solver
        self.lambda_initial = lambda_initial
        self.smooth_weight = smooth_weight
        # Cache analyzed problems keyed by (T, nq, n_kp_dim, has_smoothness, has_reg)
        self._cache: dict[tuple, _AnalyzedProblem] = {}

    # ------------------------------------------------------------------
    # Problem construction
    # ------------------------------------------------------------------

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
            kp = var_values[kp_var]  # (n_kp_dim,)
            # qs_to_opt selects which joints are optimized; rest stay at q
            # (for the batch solver we typically pass qs_to_opt=ones so full_q==q)
            full_q = jnp.where(qs_to_opt, q, jnp.zeros_like(q))
            data = mjx_data_template.replace(qpos=full_q)
            data = utils.kinematics(mjx_model, data)
            data = utils.com_pos(mjx_model, data)
            markers = utils.get_site_xpos(data, site_idxs).flatten()
            return (kp - markers) * kps_to_opt

        costs.append(marker_cost(q_all, kp_all))

        # ---- Joint regularization cost ----
        if jnp.any(q_reg_weights > 0):
            @jaxls.Cost.factory
            def reg_cost(
                var_values: jaxls.VarValues,
                q_var: QVar,
            ) -> jnp.ndarray:
                q = var_values[q_var]
                return jnp.sqrt(q_reg_weights * qs_to_opt) * q

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
            @jaxls.Cost.factory
            def smoothness_cost(
                var_values: jaxls.VarValues,
                q_curr: QVar,
                q_prev: QVar,
            ) -> jnp.ndarray:
                return (var_values[q_curr] - var_values[q_prev]) * smooth_weight

            costs.append(smoothness_cost(
                QVar(jnp.arange(1, T)),      # q[1..T-1]
                QVar(jnp.arange(0, T - 1)),  # q[0..T-2]
            ))

        # All variables: KpVar is "optimized" too but held fixed via initial_vals.
        # Including it as a variable lets jaxls trace through it dynamically.
        variables = [q_all, kp_all]

        analyzed = (
            jaxls.LeastSquaresProblem(costs=costs, variables=variables)
            .analyze()
        )

        return _AnalyzedProblem(analyzed=analyzed, QVar=QVar, KpVar=KpVar)

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
        # kp_data is (T, n_kp*3) after flatten — use last dim
        n_kp_dim = int(kp_data.shape[-1]) if kp_data.ndim > 1 else int(kp_data.shape[0])
        has_reg = bool(jnp.any(q_reg_weights > 0))
        has_smooth = self.smooth_weight > 0.0
        key = (T, mjx_model.nq, n_kp_dim, has_reg, has_smooth)

        if key not in self._cache:
            self._cache[key] = self._build(
                T=T,
                nq=mjx_model.nq,
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

        prob = self._get_analyzed(
            T, mjx_model, mjx_data_template,
            kp_data, qs_to_opt, kps_to_opt, lb, ub, site_idxs, q_reg_weights,
        )
        QVar = prob.QVar
        KpVar = prob.KpVar

        sol = prob.analyzed.solve(
            verbose=False,
            linear_solver=self.linear_solver,
            trust_region=jaxls.TrustRegionConfig(lambda_initial=self.lambda_initial),
            termination=jaxls.TerminationConfig(max_iterations=self.n_iter),
            initial_vals={
                QVar(jnp.arange(T)): q_init,   # (T, nq)
                KpVar(jnp.arange(T)): kp_data,  # (T, n_kp*3) — treated as fixed
            },
        )

        return sol[QVar(jnp.arange(T))]  # (T, nq)

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
