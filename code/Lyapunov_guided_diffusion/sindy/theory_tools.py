"""
Computable quantities from the paper (paperv2.tex), for the pendulum experiments.

Everything here mirrors a numbered object in the paper:

  * fit_ridge / LinearFeatureModel  -> ridge least squares on a fixed feature map,
                                       the estimator of Proposition 5 (prop:envelope)
  * Envelope                         -> r_N(x,u) of eq. (envelope-data)
  * RobustCertificate.d_rob          -> d_rob(x,u) of eq. (robust-residual)
  * RobustCertificate.h_rob          -> h_rob(x) = inf_u d_rob of eq. (hrob)
  * analyze_geometry                 -> safe level c, K_c, Omega_rob, height tau_hat (eq. tauhat),
                                       Theorem 2 constants beta_V, lambda_bar, tau_delta, delta_star
  * filter_controls                  -> the safety filter (eq. filter), minimal change to the
                                       proposed control so that d_rob <= 0 (Assumption 1, s = 0)
  * rollout                          -> sampled-data closed loop of Section 2 (execution model)

The control is scalar for the pendulum (m = 1) and the feature map is
phi(x,u) = [psi(x), u], so f_hat is control-affine by construction.
"""
import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch

from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_clbf.experiments import ExperimentSuite
from neural_clbf.systems import InvertedPendulum

GRAV = 9.80665

# State box of the pendulum (state_limits) and the unsafe radius (unsafe_mask).
STATE_BOX = ((-2.0, 2.0), (-2.0, 2.0))
UNSAFE_RADIUS = 1.5


# ---------------------------------------------------------------------------
# Feature libraries psi(x); phi(x,u) = [psi(x), u]
# ---------------------------------------------------------------------------
LIBRARIES = {
    # Exact and well conditioned: theta' = theta_dot, theta'' = g sin(theta) - b theta_dot + u.
    "exact": ["sin_theta", "theta_dot"],
    # Exact but redundant (near-collinear near the goal): shows the conditioning effect.
    "rich": ["one", "theta", "theta_dot", "sin_theta", "cos_theta"],
    # Misspecified: no sin(theta), so a bias floor m_bar > 0 remains (Prop. 5).
    "linear": ["theta", "theta_dot"],
    # Exact plus a constant: still well specified, but the fitted model can carry an
    # offset at the goal, so Omega_rob is non-empty and the O(delta^2) rate is visible.
    "bias": ["one", "sin_theta", "theta_dot"],
}


def psi(x: torch.Tensor, library: str) -> torch.Tensor:
    th, thd = x[..., 0], x[..., 1]
    table = {
        "one": torch.ones_like(th),
        "theta": th,
        "theta_dot": thd,
        "sin_theta": torch.sin(th),
        "cos_theta": torch.cos(th),
    }
    return torch.stack([table[name] for name in LIBRARIES[library]], dim=-1)


def phi(x: torch.Tensor, u: torch.Tensor, library: str) -> torch.Tensor:
    return torch.cat([psi(x, library), u], dim=-1)


# ---------------------------------------------------------------------------
# True pendulum field (evaluation only; never seen by the controller)
# ---------------------------------------------------------------------------
def true_f(x: torch.Tensor, u: torch.Tensor, m=1.0, L=1.0, b=0.01) -> torch.Tensor:
    th, thd = x[..., 0], x[..., 1]
    acc = GRAV / L * torch.sin(th) - b / (m * L ** 2) * thd + u[..., 0] / (m * L ** 2)
    return torch.stack([thd, acc], dim=-1)


# ---------------------------------------------------------------------------
# Ridge least squares (Prop. 5) and a pysindy-compatible model wrapper
# ---------------------------------------------------------------------------
class LinearFeatureModel:
    """f_hat(x,u) = Xi^T phi(x,u). Exposes predict(x, u=u) like a pysindy model,
    so SINDyEstimatedSystem can wrap it unchanged."""

    def __init__(self, Xi: np.ndarray, library: str):
        self.Xi = np.asarray(Xi, dtype=np.float64)  # (p, n)
        self.library = library

    def predict(self, x, u=None):
        x_t = torch.as_tensor(np.atleast_2d(x), dtype=torch.float64)
        u_t = torch.as_tensor(np.atleast_2d(u), dtype=torch.float64)
        feats = phi(x_t, u_t, self.library).numpy()
        return feats @ self.Xi

    def f_torch(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        Xi = torch.as_tensor(self.Xi, dtype=x.dtype, device=x.device)
        return phi(x, u, self.library) @ Xi

    def drift_and_gain(self, x: torch.Tensor):
        """f_hat(x,u) = f0(x) + g u with g constant (control-affine library)."""
        Xi = torch.as_tensor(self.Xi, dtype=x.dtype, device=x.device)
        p_x = Xi.shape[0] - 1
        f0 = psi(x, self.library) @ Xi[:p_x]  # (B, n)
        g = Xi[p_x]  # (n,)
        return f0, g


@dataclass
class Envelope:
    """r_N(x,u) = c_N * ||phi(x,u)||_{G^{-1}} + m_bar  (Prop. 5)."""

    Ginv: np.ndarray
    c_N: float
    beta_N: float
    m_bar: float
    library: str
    kappa: float  # lambda_min(Phi^T Phi) / N, persistent excitation level

    def leverage(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """||phi(x,u)||_{G^{-1}} for x (B,n), u (B,J): returns (B,J)."""
        M = torch.as_tensor(self.Ginv, dtype=x.dtype, device=x.device)
        p_x = M.shape[0] - 1
        ps = psi(x, self.library)  # (B, p_x)
        q_xx = ((ps @ M[:p_x, :p_x]) * ps).sum(-1)  # (B,)
        q_xu = ps @ M[:p_x, p_x]  # (B,)
        q_uu = M[p_x, p_x]
        q = q_xx[:, None] + 2.0 * q_xu[:, None] * u + q_uu * u ** 2
        return torch.sqrt(torch.clamp(q, min=0.0))

    def r(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.c_N * self.leverage(x, u) + self.m_bar


def fit_ridge(X, U, Y, library, sigma, S, alpha, rho, m_bar=0.0, b_fd=0.0):
    """Ridge estimate and the uniform envelope of Prop. 5.

    beta_N = sigma sqrt(2 log(n/alpha) + log det(G)/rho^p) + sqrt(rho) S
    c_N    = sqrt(n) (beta_N + sqrt(N) b_bar),  b_bar = b_fd + m_bar
    """
    x_t = torch.as_tensor(X, dtype=torch.float64)
    u_t = torch.as_tensor(U, dtype=torch.float64)
    Phi = phi(x_t, u_t, library).numpy()
    N, p = Phi.shape
    n = Y.shape[1]
    G = Phi.T @ Phi + rho * np.eye(p)
    Xi = np.linalg.solve(G, Phi.T @ Y)
    _, logdet = np.linalg.slogdet(G)
    beta_N = sigma * math.sqrt(max(2.0 * math.log(n / alpha) + logdet - p * math.log(rho), 0.0)) \
        + math.sqrt(rho) * S
    b_bar = b_fd + m_bar
    c_N = math.sqrt(n) * (beta_N + math.sqrt(N) * b_bar)
    kappa = float(np.linalg.eigvalsh(Phi.T @ Phi).min() / N)
    env = Envelope(np.linalg.inv(G), c_N, beta_N, m_bar, library, kappa)
    return LinearFeatureModel(Xi, library), env


def oracle_misspecification(library, u_max, n_grid=121, n_u=41):
    """Best-approximation coefficients Xi* of the true field in the library over
    X x U, and m_bar = sup ||f - Xi*^T phi||. Uses the true f, so it is an ORACLE
    quantity: it is what Prop. 5 assumes is known when the library is misspecified."""
    xs = _state_grid(n_grid)
    us = torch.linspace(-u_max, u_max, n_u, dtype=torch.float64)
    X = xs.repeat_interleave(n_u, 0)
    U = us.repeat(xs.shape[0])[:, None]
    F = true_f(X, U)
    Phi = phi(X, U, library)
    Xi_star = torch.linalg.lstsq(Phi, F).solution
    m_bar = float(torch.norm(F - Phi @ Xi_star, dim=-1).max())
    return Xi_star.numpy(), m_bar


# ---------------------------------------------------------------------------
# Certificate quantities
# ---------------------------------------------------------------------------
def V_and_grad(controller, x: torch.Tensor):
    with torch.no_grad():
        V, JV = controller.V_with_jacobian(x.float())
    return V.double().reshape(-1), JV[:, 0, :].double()


class RobustCertificate:
    """d_rob(x,u) = grad V . f_hat(x,u) + lam V + ||grad V|| r(x,u)."""

    def __init__(self, controller, model: LinearFeatureModel, env: Envelope, lam: float,
                 u_max: float, n_u: int = 401):
        self.controller = controller
        self.model = model
        self.env = env
        self.lam = lam
        self.u_max = u_max
        self.u_grid = torch.linspace(-u_max, u_max, n_u, dtype=torch.float64)

    def d_rob(self, x: torch.Tensor, u: torch.Tensor, V=None, gV=None) -> torch.Tensor:
        """x (B,n), u (B,J) -> (B,J)."""
        if V is None:
            V, gV = V_and_grad(self.controller, x)
        f0, g = self.model.drift_and_gain(x.double())
        LfV = (gV * f0).sum(-1)
        LgV = gV @ g
        r = self.env.r(x.double(), u)
        return LfV[:, None] + LgV[:, None] * u + self.lam * V[:, None] + gV.norm(dim=-1)[:, None] * r

    def h_rob(self, x: torch.Tensor, batch: int = 4096) -> torch.Tensor:
        """inf over the control grid (an upper bound on the exact inf, so the
        resulting Omega_rob is conservative)."""
        out = []
        for i in range(0, x.shape[0], batch):
            xb = x[i:i + batch].double()
            ub = self.u_grid.expand(xb.shape[0], -1)
            out.append(self.d_rob(xb, ub).min(dim=-1).values)
        return torch.cat(out)


class InvalidCertificate(Exception):
    """V has no sublevel set around the goal that avoids the unsafe ring."""


QUAD_P = torch.tensor([[3.0, 1.0], [1.0, 1.0]], dtype=torch.float64)


class CertView:
    """The certificate the theorems are checked on, separate from the controller that
    proposes controls: the trained V (optionally shifted so V(goal) = 0, which the decay
    condition V' <= -lambda V requires) or a fixed quadratic 0.5 x^T P x."""

    def __init__(self, controller, kind: str = "nn", shift_goal: bool = False):
        self.controller, self.kind = controller, kind
        self.shift = 0.0
        if kind == "nn" and shift_goal:
            with torch.no_grad():
                self.shift = float(controller.V_with_jacobian(torch.zeros(1, 2))[0])

    def V_with_jacobian(self, x):
        if self.kind == "quad":
            xd = x.double()
            V = 0.5 * (xd @ QUAD_P * xd).sum(-1)
            return V.to(x.dtype), (xd @ QUAD_P)[:, None, :].to(x.dtype)
        V, J = self.controller.V_with_jacobian(x)
        return V - self.shift, J


def _minimax_level_from_goal(V: torch.Tensor, n_grid: int) -> torch.Tensor:
    """For each grid point, the lowest level L such that it lies in the goal's
    connected component of {V <= L} (minimax path value, 4-neighbour grid)."""
    import heapq
    Vg = V.reshape(n_grid, n_grid).numpy()
    best = np.full((n_grid, n_grid), np.inf)
    i0 = j0 = n_grid // 2
    best[i0, j0] = Vg[i0, j0]
    heap = [(Vg[i0, j0], i0, j0)]
    while heap:
        lv, i, j = heapq.heappop(heap)
        if lv > best[i, j]:
            continue
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < n_grid and 0 <= b < n_grid:
                nl = max(lv, Vg[a, b])
                if nl < best[a, b]:
                    best[a, b] = nl
                    heapq.heappush(heap, (nl, a, b))
    return torch.as_tensor(best.reshape(-1), dtype=V.dtype)


def _state_grid(n_grid: int) -> torch.Tensor:
    axes = [torch.linspace(lo, hi, n_grid, dtype=torch.float64) for (lo, hi) in STATE_BOX]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([m.reshape(-1) for m in mesh], dim=-1)


def analyze_geometry(controller, cert: RobustCertificate, env: Envelope, lam: float,
                     u_max: float, n_grid: int = 161, n_u_true: int = 201,
                     v_floor: float = 1e-6, band_frac: float = 0.5) -> Dict:
    """Everything the theorems need, computed on a grid of the state box.

    safe level c   : just below the lowest level at which the goal's connected component
                     of {V <= level} reaches {||x|| >= 1.5}; K_c is that component
                     (Assumption reg). V outside the training region can dip low near
                     the box edge, so a global min over the ring would be wrong.
    tau_hat        : sup V over Omega_rob cap K_c (eq. tauhat)
    delta_c (env)  : sup r_N over K_c x U (eq. deltac), computable
    delta_c (true) : sup ||f - f_hat|| over K_c x U, uses the true f (evaluation only)
    beta_V         : sup ||grad V|| / V^{1/2} on K_c (Thm 2(b), q = 2)
    lambda_bar     : inf over K_c of -inf_u L_f V / V (Thm 2(a)), uses the true f
    """
    xs = _state_grid(n_grid)
    V, gV = V_and_grad(controller, xs)
    norms = xs.norm(dim=-1)
    outside = norms >= UNSAFE_RADIUS
    level = _minimax_level_from_goal(V, n_grid)
    c = float(level[outside].min()) * (1.0 - 1e-3)
    inK = level <= c
    V_goal, _ = V_and_grad(controller, torch.zeros(1, 2, dtype=torch.float64))
    if int(inK.sum()) < 10 or float(V_goal) > c:
        raise InvalidCertificate(
            f"V(goal)={float(V_goal):.4g} vs c={c:.4g}, |K_c| grid points={int(inK.sum())}")
    xK, VK, gK = xs[inK], V[inK], gV[inK]

    h = cert.h_rob(xK)
    in_omega = h > 0
    tau_hat = float(VK[in_omega].max()) if bool(in_omega.any()) else 0.0
    tau_hat_coarse = tau_hat
    # Omega_rob shrinks onto the goal as delta -> 0 and can fall between grid points;
    # zoom in around the goal so the O(delta^2) rate is resolved.
    dx0 = (STATE_BOX[0][1] - STATE_BOX[0][0]) / (n_grid - 1)
    R = (float(xK[in_omega].norm(dim=-1).max()) if bool(in_omega.any()) else 0.0) + 2 * dx0
    for _ in range(4):
        if R >= 1.0:
            break
        ax_z = torch.linspace(-R, R, n_grid, dtype=torch.float64)
        mz = torch.meshgrid(ax_z, ax_z, indexing="ij")
        xz = torch.stack([m.reshape(-1) for m in mz], dim=-1)
        Vz, _ = V_and_grad(controller, xz)
        keep = (Vz <= c) & (xz.norm(dim=-1) < UNSAFE_RADIUS)
        xz, Vz = xz[keep], Vz[keep]
        om_z = cert.h_rob(xz) > 0
        if bool(om_z.any()):
            tau_hat = max(tau_hat, float(Vz[om_z].max()))
            R_new = float(xz[om_z].norm(dim=-1).max()) + 4 * R / (n_grid - 1)
        else:
            R_new = 0.25 * R
        if R_new >= 0.9 * R:
            break
        R = R_new

    # Envelope and true error over K_c x U (true f only for evaluation).
    u_true = torch.linspace(-u_max, u_max, 41, dtype=torch.float64)
    r_vals, err_vals = [], []
    for i in range(0, xK.shape[0], 2048):
        xb = xK[i:i + 2048]
        ub = u_true.expand(xb.shape[0], -1)
        r_vals.append(env.r(xb, ub))
        Xr = xb.repeat_interleave(u_true.numel(), 0)
        Ur = u_true.repeat(xb.shape[0])[:, None]
        e = torch.norm(true_f(Xr, Ur) - cert.model.f_torch(Xr, Ur), dim=-1)
        err_vals.append(e.reshape(xb.shape[0], -1))
    r_all = torch.cat(r_vals)
    e_all = torch.cat(err_vals)
    band = VK >= band_frac * c
    env_ratio = float((e_all / torch.clamp(r_all, min=1e-12)).max())

    # Theorem 2 constants (q = 2).
    pos = VK > v_floor
    beta_V = float((gK[pos].norm(dim=-1) / VK[pos].sqrt()).max())
    u_tr = torch.linspace(-u_max, u_max, n_u_true, dtype=torch.float64)
    best_LfV = []
    for i in range(0, xK.shape[0], 1024):
        xb, gb = xK[i:i + 1024], gK[i:i + 1024]
        Xr = xb.repeat_interleave(u_tr.numel(), 0)
        Ur = u_tr.repeat(xb.shape[0])[:, None]
        fv = true_f(Xr, Ur).reshape(xb.shape[0], u_tr.numel(), -1)
        best_LfV.append((fv * gb[:, None, :]).sum(-1).min(dim=-1).values)
    best_LfV = torch.cat(best_LfV)
    lambda_bar = float((-best_LfV[pos] / VK[pos]).min())
    delta_c_env = float(r_all.max())
    if lambda_bar > lam:
        tau_delta = (2.0 * beta_V * delta_c_env / (lambda_bar - lam)) ** 2
        delta_star = (lambda_bar - lam) * math.sqrt(c) / (2.0 * beta_V)
    else:
        tau_delta, delta_star = float("nan"), float("nan")

    # Local surrogate of the Assumption-reg constant B on K_c x U, for the sampled-data
    # terms of Theorem 1 (grid estimate; Hessian via finite differences of grad V).
    f_abs = torch.norm(true_f(xK.repeat_interleave(2, 0),
                              torch.tensor([[-u_max], [u_max]], dtype=torch.float64)
                              .repeat(xK.shape[0], 1)), dim=-1).max()
    dx = (STATE_BOX[0][1] - STATE_BOX[0][0]) / (n_grid - 1)
    gV_grid = gV.reshape(n_grid, n_grid, 2)
    hess = torch.cat([(gV_grid[1:, :, :] - gV_grid[:-1, :, :]).abs().reshape(-1),
                      (gV_grid[:, 1:, :] - gV_grid[:, :-1, :]).abs().reshape(-1)]).max() / dx
    B_loc = float(max(gK.norm(dim=-1).max(), f_abs, hess, GRAV / 1.0 + 0.01))

    return {
        "c": c,
        "Kc_frac": float(inK.double().mean()),
        "omega_frac_Kc": float(in_omega.double().mean()),
        "tau_hat": tau_hat,
        "tau_hat_coarse": tau_hat_coarse,
        "delta_c_env": delta_c_env,
        "delta_c_true": float(e_all.max()),
        "env_valid": int(bool((e_all <= r_all + 1e-9).all())),
        "env_ratio_max": env_ratio,
        "r_band": float(r_all[band].max()) if bool(band.any()) else float("nan"),
        "r_inner": float(r_all[~band].max()) if bool((~band).any()) else float("nan"),
        "beta_V": beta_V,
        "lambda_bar": lambda_bar,
        "tau_delta": tau_delta,
        "delta_star": delta_star,
        "thm2_applicable": int(lambda_bar > lam),
        "thm2_holds": int(lambda_bar <= lam or tau_hat <= tau_delta * (1 + 1e-6) + 1e-12),
        "B_loc": B_loc,
        "grid_spacing": dx,
        "_xK": xK,
    }


# ---------------------------------------------------------------------------
# Safety filter (eq. filter) and the sampled-data rollout
# ---------------------------------------------------------------------------
def filter_controls(cert: RobustCertificate, x: torch.Tensor, u_prop: torch.Tensor,
                    n_u: int = 2001):
    """Closest control in U to u_prop with d_rob <= 0; if none exists (x in Omega_rob)
    return argmin_u d_rob (anything is allowed there). Returns (u, d_rob(x,u), h_rob(x))."""
    xb = x.double()
    V, gV = V_and_grad(cert.controller, xb)
    ug = torch.linspace(-cert.u_max, cert.u_max, n_u, dtype=torch.float64).expand(xb.shape[0], -1)
    up = torch.clamp(u_prop.double(), -cert.u_max, cert.u_max)
    cand = torch.cat([up, ug], dim=-1)  # first column = the proposal itself
    d = cert.d_rob(xb, cand, V, gV)
    feasible = d <= 0
    dist = (cand - up).abs().masked_fill(~feasible, float("inf"))
    j_feas = dist.argmin(dim=-1)
    j_best = d.argmin(dim=-1)
    has = feasible.any(dim=-1)
    j = torch.where(has, j_feas, j_best)
    u = cand.gather(1, j[:, None])
    return u, d.gather(1, j[:, None])[:, 0], d.min(dim=-1).values


def make_policy(kind: str, controller, cert: RobustCertificate):
    """kind in {qp, qp_filter, diffusion, diffusion_filter}. Returns pi(x) -> (B,1)."""

    def proposal(x):
        if kind.startswith("qp"):
            with torch.no_grad():
                return controller.u(x.float()).double()
        with torch.no_grad():
            _, best = controller.sample_trajectories(
                x.float(), horizon=3, num_samples=32, temperature=0.5, use_mean=True)
        return best[:, 0, :].double()

    def pi(x):
        u0 = torch.clamp(proposal(x), -cert.u_max, cert.u_max)
        if kind.endswith("filter"):
            u, _, _ = filter_controls(cert, x, u0)
            return u
        return u0

    return pi


def rollout(policy: Callable, cert: RobustCertificate, x0: torch.Tensor, c: float,
            tau_hat: float, lam: float, t_sim: float, h: float, dt: float) -> Dict:
    """Sampled-data closed loop on the TRUE pendulum (Section 2, execution model).

    Records V at every simulation step, and at every sampling instant whether the
    state was certifiable (x_k in K_c minus Omega_rob) and the d_rob of the applied
    control, which is what Assumption 1 constrains.
    """
    n_ctrl = int(round(t_sim / h))
    sub = int(round(h / dt))
    x = x0.double().clone()
    B = x.shape[0]
    V0, _ = V_and_grad(cert.controller, x)
    Vs = [V0]
    ts = [0.0]
    max_norm = x.norm(dim=-1)
    diverged = torch.zeros(B, dtype=torch.bool)
    cert_states, cert_ok, d_applied = 0, 0, []
    t = 0.0
    for _ in range(n_ctrl):
        u = policy(x)
        with torch.no_grad():
            V_k, gV_k = V_and_grad(cert.controller, x)
            d_k = cert.d_rob(x, u, V_k, gV_k)[:, 0]
            h_k = cert.h_rob(x)
        certifiable = (h_k <= 0) & (V_k <= c) & (~diverged)
        cert_states += int(certifiable.sum())
        cert_ok += int((certifiable & (d_k <= 1e-9)).sum())
        if bool(certifiable.any()):
            d_applied.append(d_k[certifiable])
        for _ in range(sub):
            x = x + dt * true_f(x, u)
            t += dt
            bad = (~torch.isfinite(x).all(-1)) | (x.abs().amax(-1) > 1e3)
            diverged |= bad
            x = torch.where(diverged[:, None], torch.full_like(x, 2.0), x)
            Vt, _ = V_and_grad(cert.controller, x)
            Vs.append(Vt)
            ts.append(t)
            max_norm = torch.maximum(max_norm, x.norm(dim=-1))
    V_traj = torch.stack(Vs, dim=1)  # (B, T)
    t_arr = torch.tensor(ts, dtype=torch.float64)
    ideal = torch.maximum(torch.exp(-lam * t_arr)[None, :] * V0[:, None],
                          torch.full_like(V_traj, tau_hat))
    excess = torch.clamp(V_traj - ideal, min=0.0)
    tail = V_traj[:, int(0.8 * V_traj.shape[1]):]
    safe_box = (max_norm < UNSAFE_RADIUS) & (~diverged)
    safe_V = (V_traj.max(dim=1).values < c) & safe_box
    d_cat = torch.cat(d_applied) if d_applied else torch.zeros(0, dtype=torch.float64)
    return {
        "V_traj": V_traj,
        "t": t_arr,
        "V0": V0,
        "safe_box_rate": float(safe_box.double().mean()),
        "safe_V_rate": float(safe_V.double().mean()),
        "diverged_rate": float(diverged.double().mean()),
        "max_V": float(V_traj.max()),
        "tail_max_V": float(tail.max()),
        "limsup_V_mean": float(tail.mean()),
        "max_norm": float(max_norm.max()),
        "cert_frac": cert_ok / cert_states if cert_states else float("nan"),
        "s_emp": float(d_cat.max()) if d_cat.numel() else float("nan"),
        "bound_excess_ideal": float(excess.max()),
    }


# ---------------------------------------------------------------------------
# Controller construction without training (used to load a fixed reference V)
# ---------------------------------------------------------------------------
def make_untrained_controller(dynamics_model, scenarios, controller_period=0.05, barrier=True):
    """Same hyper-parameters as training/train_inverted_pendulum_sindy.build_and_train,
    except barrier: Theorem 1 needs {V <= c} inside the safe region, which only the
    barrier terms (V <= 1 on the safe ball, V >= 1 on the unsafe ring) enforce."""
    data_module = EpisodicDataModule(
        dynamics_model, [(-np.pi / 2, np.pi / 2), (-1.0, 1.0)],
        trajectories_per_episode=0, trajectory_length=1, fixed_samples=10000,
        max_points=100000, val_split=0.1, batch_size=64,
    )
    return NeuralCLBFController(
        dynamics_model, scenarios, data_module, experiment_suite=ExperimentSuite([]),
        clbf_hidden_layers=2, clbf_hidden_size=64, clf_lambda=1.0, safe_level=1.0,
        controller_period=controller_period, clf_relaxation_penalty=1e2,
        num_init_epochs=5, epochs_per_episode=100, barrier=barrier, disable_gurobi=True,
        # V = 0.5||NN||^2 alone vanishes on a curve, not a point; the LQR bowl makes it a bowl.
        add_nominal=True,
    )


def train_certificate(dynamics_model, scenarios, max_epochs, accelerator="cpu",
                      barrier=True, controller_period=0.05):
    import tempfile
    import pytorch_lightning as pl
    from pytorch_lightning import loggers as pl_loggers

    controller = make_untrained_controller(dynamics_model, scenarios, controller_period, barrier)
    trainer = pl.Trainer(
        logger=pl_loggers.TensorBoardLogger(tempfile.mkdtemp(prefix="theory_"), name="run"),
        max_epochs=max_epochs, accelerator=accelerator, gradient_clip_val=10.0,
        enable_progress_bar=False, log_every_n_steps=50, enable_checkpointing=False,
    )
    trainer.fit(controller)
    if trainer.interrupted:
        raise RuntimeError("training was interrupted; refusing to save a partial certificate")
    return controller


def make_true_system(dt=0.01, controller_period=0.05):
    params = {"m": 1.0, "L": 1.0, "b": 0.01}
    return InvertedPendulum(params, dt=dt, controller_dt=controller_period, scenarios=[params]), [params]
