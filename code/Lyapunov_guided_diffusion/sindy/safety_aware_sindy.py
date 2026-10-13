"""
Item #3: Safety-aware SINDy --- volume-penalized identification + alternating loop.

Plain SINDy fits ``f_hat`` to minimise prediction error uniformly. But the
downstream guarantee (Theorem 8) does not care about uniform error ``delta``; it
cares about the *volume of the robust violation set*

    Omega_rob = { x : inf_u [ grad V . f_hat(x,u) + lambda V(x) + ||grad V|| r ] > 0 },

the region where the learned model cannot certify certificate decrease. This
module fits ``f_hat`` to *also* shrink ``Vol(Omega_rob)`` --- i.e. to spend its
accuracy budget where safety needs it (the certificate boundary).

The circular dependency William flagged (``V`` comes from ``f_hat`` but choosing
``f_hat`` needs ``V``) is broken by an *alternating* scheme:

    plain SINDy  ->  train CLBF (V_0)  ->  refine f_hat against V_0
                 ->  retrain CLBF (V_1)  ->  refine again  ->  ...

with a data-consistency leash so the estimator cannot "cheat" by inventing a
model that looks safe but does not fit the recordings.

Design notes (grounded in the existing pipeline):
  * ``pysindy`` STLSQ is *not* differentiable, so the volume penalty needs a
    torch re-implementation of the coefficient fit. That lives here.
  * We restrict to a *control-affine* library ``f_hat(x,u) = a(x) + B u`` --- the
    same structure the paper's ``ControlAffineSystem`` assumes --- so the
    ``inf_u`` inside ``Omega_rob`` is closed form.
  * The refined model exposes a numpy ``predict(X, u=U)`` so it drops straight
    into ``SINDyEstimatedSystem`` with no change to the rest of the pipeline.

The certificate interface used is ``controller.V_with_jacobian(x) -> (V, JV)``.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Differentiable control-affine feature library (inverted pendulum).
# --------------------------------------------------------------------------- #
# Drift features of the STATE only. The control enters linearly (control-affine),
# so f_hat(x, u) = drift_features(x) @ Xi_a + B * u. This matches the true
# pendulum, whose dynamics are degree-1 polynomial + sin(theta) + linear u.
DRIFT_FEATURE_NAMES: List[str] = [
    "1", "theta", "theta_dot",
    "sin(theta)", "cos(theta)", "sin(theta_dot)", "cos(theta_dot)",
]
N_DRIFT = len(DRIFT_FEATURE_NAMES)


def drift_features(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the differentiable drift library on a batch of states.

    Args:
        x: ``(N, 2)`` tensor of states ``(theta, theta_dot)``.
    Returns:
        ``(N, N_DRIFT)`` tensor of features.
    """
    theta = x[:, 0:1]
    theta_dot = x[:, 1:2]
    return torch.cat(
        [
            torch.ones_like(theta),
            theta,
            theta_dot,
            torch.sin(theta),
            torch.cos(theta),
            torch.sin(theta_dot),
            torch.cos(theta_dot),
        ],
        dim=1,
    )


class TorchControlAffineSINDy:
    """A differentiable control-affine SINDy model with a numpy ``predict``.

    ``f_hat(x, u) = drift_features(x) @ Xi_a + u * B`` with ``Xi_a`` of shape
    ``(N_DRIFT, n_dims)`` and ``B`` of shape ``(n_dims,)``. The ``predict``
    method mirrors ``pysindy``'s signature so an instance drops straight into
    :class:`SINDyEstimatedSystem`.
    """

    def __init__(self, Xi_a: torch.Tensor, B: torch.Tensor):
        self.Xi_a = Xi_a.detach().clone()
        self.B = B.detach().clone()
        self.n_dims = self.Xi_a.shape[1]

    # -- differentiable core (torch) -------------------------------------- #
    def f_hat(self, x: torch.Tensor, u: torch.Tensor,
              Xi_a: Optional[torch.Tensor] = None,
              B: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Control-affine field, differentiable in (Xi_a, B). ``u`` is ``(N,1)``."""
        Xi_a = self.Xi_a if Xi_a is None else Xi_a
        B = self.B if B is None else B
        return drift_features(x) @ Xi_a + u * B  # (N, n_dims)

    # -- numpy interface for SINDyEstimatedSystem ------------------------- #
    def predict(self, X_np: np.ndarray, u=None) -> np.ndarray:
        x = torch.as_tensor(np.asarray(X_np), dtype=torch.float32)
        if u is None:
            u_t = torch.zeros(x.shape[0], 1)
        else:
            u_t = torch.as_tensor(np.asarray(u), dtype=torch.float32).reshape(x.shape[0], -1)[:, :1]
        with torch.no_grad():
            return self.f_hat(x, u_t).cpu().numpy()

    def coefficients(self) -> np.ndarray:
        """Stacked ``[Xi_a ; B]`` as ``(n_dims, N_DRIFT + 1)`` for inspection."""
        with torch.no_grad():
            W = torch.cat([self.Xi_a, self.B.unsqueeze(0)], dim=0)  # (N_DRIFT+1, n_dims)
            return W.t().cpu().numpy()

    def get_feature_names(self) -> List[str]:
        return DRIFT_FEATURE_NAMES + ["u"]


# --------------------------------------------------------------------------- #
# Plain-SINDy warm start (torch STLSQ over the control-affine library).
# --------------------------------------------------------------------------- #
def torch_stlsq_warmstart(
    X: np.ndarray,
    U: np.ndarray,
    Xdot: np.ndarray,
    threshold: float = 0.05,
    n_iter: int = 10,
) -> TorchControlAffineSINDy:
    """Ordinary sequentially-thresholded least squares in the affine library.

    This is *plain* SINDy (no safety term) --- the warm start / baseline the
    alternating loop refines from.
    """
    x = torch.as_tensor(X, dtype=torch.float32)
    u = torch.as_tensor(U, dtype=torch.float32).reshape(x.shape[0], -1)[:, :1]
    y = torch.as_tensor(Xdot, dtype=torch.float32)  # (N, n_dims)

    Phi = torch.cat([drift_features(x), u], dim=1)  # (N, N_DRIFT+1)
    n_feat = Phi.shape[1]
    n_dims = y.shape[1]

    W = torch.linalg.lstsq(Phi, y).solution  # (n_feat, n_dims)
    active = torch.ones_like(W, dtype=torch.bool)
    for _ in range(n_iter):
        small = W.abs() < threshold
        W = W.masked_fill(small, 0.0)
        new_active = ~small
        if torch.equal(new_active, active) and _ > 0:
            break
        active = new_active
        # refit each output column on its active features
        for k in range(n_dims):
            cols = active[:, k]
            if cols.any():
                W[cols, k] = torch.linalg.lstsq(Phi[:, cols], y[:, k:k + 1]).solution.squeeze(1)

    Xi_a = W[:N_DRIFT].clone()
    B = W[N_DRIFT].clone()
    return TorchControlAffineSINDy(Xi_a, B)


# --------------------------------------------------------------------------- #
# Robust violation residual and its smooth-volume surrogate.
# --------------------------------------------------------------------------- #
def robust_residual_inf(
    x: torch.Tensor,
    V: torch.Tensor,
    gradV: torch.Tensor,
    Xi_a: torch.Tensor,
    B: torch.Tensor,
    lam: float,
    r: float,
    u_lo: float,
    u_hi: float,
) -> torch.Tensor:
    """``h_rob(x) = inf_u [ grad V . f_hat(x,u) + lambda V + ||grad V|| r ]``.

    Closed form for control-affine ``f_hat = a(x) + B u`` over the control box
    ``[u_lo, u_hi]``. Differentiable in ``(Xi_a, B)``; ``V, gradV`` are treated
    as constants (the certificate is frozen during a refinement round).
    """
    a = drift_features(x) @ Xi_a            # (N, n_dims)
    gVa = (gradV * a).sum(dim=1)            # (N,)
    gVb = (gradV * B).sum(dim=1)            # (N,)  = grad V . B
    # min over u in [u_lo, u_hi] of (grad V . B) * u
    min_ctrl = torch.minimum(gVb * u_lo, gVb * u_hi)
    gradV_norm = gradV.norm(dim=1)
    return gVa + min_ctrl + lam * V + gradV_norm * r


def volume_surrogate(h_rob: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    """Smooth fraction of sampled states inside ``Omega_rob`` (``h_rob > 0``)."""
    return torch.sigmoid(h_rob / tau).mean()


# --------------------------------------------------------------------------- #
# Volume-penalized refinement (one alternating step; V frozen).
# --------------------------------------------------------------------------- #
def refine_coefficients(
    model0: TorchControlAffineSINDy,
    X: np.ndarray,
    U: np.ndarray,
    Xdot: np.ndarray,
    V_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    lam: float = 1.0,
    r: Optional[float] = None,
    rho: float = 1.0,
    kappa: float = 0.0,
    u_range: Optional[Tuple[float, float]] = None,
    sample_box: Tuple[Tuple[float, float], ...] = ((-1.0, 1.0), (-1.0, 1.0)),
    n_sample: int = 2000,
    steps: int = 400,
    lr: float = 1e-2,
    data_tol: float = 0.25,
    data_tol_abs: float = 1e-3,
    tau: float = 0.05,
    objective: str = "volume",
    c_safe: Optional[float] = None,
    seed: int = 0,
    verbose: bool = False,
) -> Tuple[TorchControlAffineSINDy, Dict[str, float]]:
    """Refine ``f_hat`` to shrink ``Vol(Omega_rob)`` while staying data-consistent.

    Objective (minimised over the coefficients):

        data_mse  +  kappa * ||coeffs||_1  +  rho * vol_surrogate(V, f_hat)
                  +  leash * relu(data_mse - (1 + data_tol) * baseline_mse)

    The final term is the *data-consistency leash*: it forbids trading prediction
    accuracy for a smaller danger zone beyond a tolerance, which is what stops the
    estimator from cheating.

    Args:
        model0: warm-start model (plain SINDy) to refine.
        X, U, Xdot: transition data (numpy).
        V_fn: certificate accessor ``x -> (V (N,), gradV (N, n_dims))``, frozen.
        lam: certificate rate ``lambda`` (``clf_lambda``).
        r: constant error-envelope value; ``None`` uses the residual-RMS envelope.
        rho: weight on the volume surrogate.
        kappa: L1 sparsity weight.
        u_range: control box for the ``inf_u`` (defaults to data range).
        sample_box: operating box to sample states for the volume penalty.
        n_sample: number of states to sample for the volume penalty.
        steps, lr: optimizer budget.
        data_tol: allowed relative increase in data MSE (the leash).
        data_tol_abs: absolute slack added to the leash ceiling, so a near-zero
            baseline MSE (clean data) does not create a razor-thin band.
    Returns:
        (refined_model, diagnostics).
    """
    torch.manual_seed(seed)
    x_data = torch.as_tensor(X, dtype=torch.float32)
    u_data = torch.as_tensor(U, dtype=torch.float32).reshape(x_data.shape[0], -1)[:, :1]
    y_data = torch.as_tensor(Xdot, dtype=torch.float32)

    if u_range is None:
        u_lo, u_hi = float(u_data.min()), float(u_data.max())
    else:
        u_lo, u_hi = float(u_range[0]), float(u_range[1])

    # Fixed sample points for the volume penalty, with frozen (V, gradV).
    g = torch.Generator().manual_seed(seed)
    lows = torch.tensor([b[0] for b in sample_box])
    highs = torch.tensor([b[1] for b in sample_box])
    x_vol = lows + (highs - lows) * torch.rand(n_sample, len(sample_box), generator=g)
    with torch.no_grad():
        V_vol, gradV_vol = V_fn(x_vol)
        V_vol = V_vol.detach()
        gradV_vol = gradV_vol.detach()

    Xi_a = model0.Xi_a.clone().requires_grad_(True)
    B = model0.B.clone().requires_grad_(True)

    def data_mse(Xi_a_, B_):
        pred = drift_features(x_data) @ Xi_a_ + u_data * B_
        return ((pred - y_data) ** 2).mean()

    baseline_mse = float(data_mse(Xi_a, B).detach())
    r_val = (baseline_mse ** 0.5) if r is None else float(r)  # residual-RMS envelope
    ceiling = (1.0 + data_tol) * baseline_mse + data_tol_abs
    leash_w = 100.0

    def vol_term(Xi_a_, B_):
        h = robust_residual_inf(x_vol, V_vol, gradV_vol, Xi_a_, B_, lam, r_val, u_lo, u_hi)
        return volume_surrogate(h, tau=tau)

    # Direction 2 (Barrier-Regularized STLSQ): a boundary-weighted, gradV-projected
    # accuracy penalty. Unlike the volume surrogate, it keeps f_hat an honest data
    # fit -- it just spends extra accuracy on (a) the certificate-relevant direction
    # grad V and (b) the data points closest to the safety boundary (largest V),
    # where a small error can flip the Lie derivative's sign. The measured
    # derivative y_data stands in for the (unknown) true field f.
    with torch.no_grad():
        V_dat, gradV_dat = V_fn(x_data)
        V_dat = V_dat.detach()
        gradV_dat = gradV_dat.detach()
        # c is the certificate level of the safety boundary. The recorded data sits
        # in the safe interior, so we self-calibrate c just above the largest V we
        # actually observe -- this focuses the weight on the closest-to-boundary
        # points we have (an explicit c_safe overrides this).
        c_eff = (1.05 * float(V_dat.max()) + 1e-6) if c_safe is None else float(c_safe)
        w_bnd = 1.0 / (c_eff - V_dat).clamp(min=1e-3)          # 1 / (c - V)
        w_bnd = w_bnd.clamp(max=100.0 * float(w_bnd.median()))  # no single point dominates
        w_bnd = w_bnd / w_bnd.mean()                            # stable rho scale

    def bnd_term(Xi_a_, B_):
        pred = drift_features(x_data) @ Xi_a_ + u_data * B_
        resid = y_data - pred                       # (N, n_dims) = f - f_hat on data
        proj = (gradV_dat * resid).sum(dim=1)        # grad V . (f - f_hat), (N,)
        return (w_bnd * proj ** 2).mean()

    pen_term = vol_term if objective == "volume" else bnd_term

    pen0 = float(pen_term(Xi_a, B).detach())

    opt = torch.optim.Adam([Xi_a, B], lr=lr)
    for step in range(steps):
        opt.zero_grad()
        dm = data_mse(Xi_a, B)
        pt = pen_term(Xi_a, B)
        l1 = Xi_a.abs().sum() + B.abs().sum()
        leash = torch.relu(dm - ceiling)
        loss = dm + kappa * l1 + rho * pt + leash_w * leash
        loss.backward()
        opt.step()
        if verbose and (step % max(1, steps // 5) == 0):
            print(f"    step {step:4d}  data_mse={float(dm):.5f}  {objective}={float(pt):.4f}")

    with torch.no_grad():
        final_mse = float(data_mse(Xi_a, B))
        final_pen = float(pen_term(Xi_a, B))

    refined = TorchControlAffineSINDy(Xi_a, B)
    diag = {
        "objective": objective,
        "baseline_data_mse": baseline_mse,
        "final_data_mse": final_mse,
        "data_mse_ratio": final_mse / max(baseline_mse, 1e-12),
        "vol_surrogate_before": pen0,
        "vol_surrogate_after": final_pen,
        "vol_reduction": pen0 - final_pen,
        "cheated": bool(final_mse > ceiling + 1e-9),
    }
    return refined, diag


# --------------------------------------------------------------------------- #
# Certificate accessor + alternating-loop driver (uses the real pipeline).
# --------------------------------------------------------------------------- #
def make_V_fn(controller) -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Wrap ``controller.V_with_jacobian`` as ``x -> (V (N,), gradV (N, n_dims))``.

    Device-aware: samples are moved to the certificate's device and the results
    returned detached on CPU, so the (CPU) refinement is decoupled from a
    possibly-GPU controller.
    """
    try:
        device = next(controller.V_nn.parameters()).device
    except (StopIteration, AttributeError):
        device = torch.device("cpu")

    def V_fn(x: torch.Tensor):
        V, JV = controller.V_with_jacobian(x.to(device))
        gradV = JV.squeeze(1)  # (N, n_dims)
        return V.detach().cpu(), gradV.detach().cpu()
    return V_fn


def alternating_refine(
    X: np.ndarray,
    U: np.ndarray,
    Xdot: np.ndarray,
    reference_system,
    train_clbf_fn: Callable,
    scenarios,
    n_rounds: int = 2,
    threshold: float = 0.05,
    lam: float = 1.0,
    r: Optional[float] = None,
    rho: float = 1.0,
    objective: str = "volume",
    c_safe: Optional[float] = None,
    u_range: Optional[Tuple[float, float]] = None,
    dt: float = 0.01,
    controller_dt: float = 0.05,
    refine_kwargs: Optional[dict] = None,
    verbose: bool = True,
):
    """Run the ``f_hat -> V -> f_hat`` alternating loop.

    Round 0 is plain SINDy + a certificate; each subsequent round refines the
    model against the *current* certificate (shrinking ``Omega_rob``), then
    retrains the certificate against the refined model.

    Args:
        X, U, Xdot: transition data.
        reference_system: the true system (geometry / masks source).
        train_clbf_fn: callable ``(sindy_system, scenarios) -> controller``
            (e.g. a partial of ``build_and_train``).
        n_rounds: number of *refinement* rounds after the warm start.
    Returns:
        dict with the final model, controller, and per-round diagnostics.
    """
    from neural_clbf.systems.sindy_estimated_system import SINDyEstimatedSystem

    refine_kwargs = dict(refine_kwargs or {})

    def build_system(model):
        return SINDyEstimatedSystem(
            model, reference_system=reference_system,
            dt=dt, controller_dt=controller_dt, scenarios=scenarios,
        )

    # Round 0: plain SINDy warm start + certificate.
    model = torch_stlsq_warmstart(X, U, Xdot, threshold=threshold)
    if verbose:
        print("[alt] round 0: plain SINDy warm start")
    controller = train_clbf_fn(build_system(model), scenarios)

    history = []
    for rnd in range(1, n_rounds + 1):
        V_fn = make_V_fn(controller)
        model, diag = refine_coefficients(
            model, X, U, Xdot, V_fn,
            lam=lam, r=r, rho=rho, objective=objective, c_safe=c_safe,
            u_range=u_range, **refine_kwargs,
        )
        if verbose:
            print(f"[alt] round {rnd}: vol {diag['vol_surrogate_before']:.4f} -> "
                  f"{diag['vol_surrogate_after']:.4f}  "
                  f"(data mse x{diag['data_mse_ratio']:.2f}, "
                  f"cheated={diag['cheated']})")
        controller = train_clbf_fn(build_system(model), scenarios)
        history.append(diag)

    return {"model": model, "controller": controller, "history": history}


# --------------------------------------------------------------------------- #
# Local smoke test (no CLBF training, no cluster): validates the differentiable
# pieces --- warm start recovers the pendulum, and the volume penalty back-props
# and shrinks the surrogate while the data leash holds.
# --------------------------------------------------------------------------- #
def _pendulum_xdot(x: np.ndarray, u: np.ndarray,
                   g: float = 9.81, m: float = 1.0, L: float = 1.0, b: float = 0.01) -> np.ndarray:
    theta, theta_dot = x[:, 0], x[:, 1]
    ddtheta = (g / L) * np.sin(theta) - (b / (m * L ** 2)) * theta_dot + u[:, 0] / (m * L ** 2)
    return np.stack([theta_dot, ddtheta], axis=1)


def _smoke_test() -> None:
    rng = np.random.default_rng(0)
    N = 4000
    X = rng.uniform([-1.0, -1.0], [1.0, 1.0], size=(N, 2)).astype(np.float32)
    U = rng.uniform(-2.0, 2.0, size=(N, 1)).astype(np.float32)
    Xdot = _pendulum_xdot(X, U).astype(np.float32)
    # Measurement noise on the derivatives: the method's real regime is noisy,
    # finite data where the fit does NOT pin f_hat down exactly, leaving room to
    # trade a little accuracy for a smaller danger zone.
    Xdot = (Xdot + rng.normal(0.0, 0.15, size=Xdot.shape)).astype(np.float32)

    print("== 1. warm-start (plain torch STLSQ) ==")
    model = torch_stlsq_warmstart(X, U, Xdot, threshold=0.05)
    coeffs = model.coefficients()  # (n_dims, N_DRIFT+1)
    names = model.get_feature_names()
    for k, row in enumerate(coeffs):
        terms = [f"{c:+.3f}*{n}" for c, n in zip(row, names) if abs(c) > 1e-3]
        print(f"   xdot[{k}] = " + " ".join(terms))
    pred = model.predict(X, u=U)
    warm_mse = float(np.mean((pred - Xdot) ** 2))
    print(f"   warm-start data MSE = {warm_mse:.3e}")
    # sin(theta) coefficient of the theta_dot_dot equation should be ~g
    sin_idx = DRIFT_FEATURE_NAMES.index("sin(theta)")
    assert abs(coeffs[1, sin_idx] - 9.81) < 0.5, "should recover gravity term ~9.81"
    assert abs(coeffs[1, -1] - 1.0) < 0.2, "should recover unit control gain"
    print("   OK: recovers gravity + unit control gain despite noise")

    print("== 2. volume-penalized refinement against a mock quadratic V ==")

    def mock_V_fn(x: torch.Tensor):
        # V = 0.5 (theta^2 + theta_dot^2);  grad V = (theta, theta_dot)
        V = 0.5 * (x ** 2).sum(dim=1)
        JV = x.unsqueeze(1)  # (N, 1, 2)
        return V, JV.squeeze(1)

    refined, diag = refine_coefficients(
        model, X, U, Xdot, mock_V_fn,
        lam=1.0, r=0.05, rho=20.0, steps=400, lr=1e-2,
        data_tol=0.25, data_tol_abs=2e-2, seed=0, verbose=True,
    )
    print("   diagnostics:")
    for kk, vv in diag.items():
        print(f"     {kk}: {vv}")
    assert diag["vol_reduction"] > 1e-3, "refinement should reduce the danger-zone surrogate"
    assert not diag["cheated"], "data-consistency leash should hold"
    print("   OK: danger-zone surrogate shrank and the data leash held")

    print("== 3. boundary-weighted Direction 2 refinement against a mock quadratic V ==")
    refined_b, diag_b = refine_coefficients(
        model, X, U, Xdot, mock_V_fn,
        lam=1.0, rho=2.0, objective="boundary", steps=400, lr=1e-2,
        data_tol=0.25, data_tol_abs=2e-2, seed=0, verbose=True,
    )
    print("   diagnostics:")
    for kk, vv in diag_b.items():
        print(f"     {kk}: {vv}")
    assert diag_b["vol_reduction"] > 0.0, "boundary penalty should decrease"
    assert not diag_b["cheated"], "data-consistency leash should hold"
    print("   OK: boundary-weighted penalty shrank and the data leash held")

    print("\nAll smoke-test checks passed.")


if __name__ == "__main__":
    _smoke_test()
