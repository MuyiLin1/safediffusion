"""
Dimension-agnostic computable quantities of paperv2.tex, used for the pendulum (n=2)
and the cart-pole (n=4), including the three sharper results:

  * local margin bound      (Prop. prop:margin-local)  -> tau_loc
  * grid certification      (Lemma lem:grid)           -> tau_hat_cert
  * computable sampled data (Thm. thm:sampled)         -> tau_hat_h, J_h, ult_h, m0_h

f_hat(x,u) = psi(x)^T Xi_d + u chi(x)^T Xi_g (control-affine, scalar input), fitted by
ridge with the envelope of Prop. prop:envelope.
"""
import heapq
import math
import time
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn

DT = torch.float64
GRAV = 9.80665


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------
def pendulum_f(x, u):
    th, thd = x[..., 0], x[..., 1]
    return torch.stack([thd, GRAV * torch.sin(th) - 0.01 * thd + u[..., 0]], -1)


def pendulum_lib(x):
    th, thd = x[..., 0], x[..., 1]
    psi = torch.stack([torch.ones_like(th), torch.sin(th), thd], -1)
    return psi, torch.ones_like(th)[..., None]


CP_l, CP_g, CP_b = 0.5, 9.81, 0.05


def cartpole_f(x, u):
    """Inverted pendulum on a cart with cart acceleration as input; state (p, theta, p_dot,
    theta_dot), theta = 0 upright. Underactuated: one input for two positions."""
    th, pd, thd = x[..., 1], x[..., 2], x[..., 3]
    thdd = (CP_g * torch.sin(th) - torch.cos(th) * u[..., 0]) / CP_l - CP_b * thd
    return torch.stack([pd, thd, u[..., 0], thdd], -1)


def cartpole_lib(x):
    th, pd, thd = x[..., 1], x[..., 2], x[..., 3]
    one = torch.ones_like(th)
    return torch.stack([one, torch.sin(th), pd, thd], -1), torch.stack([one, torch.cos(th)], -1)


@dataclass
class Spec:
    name: str
    n: int
    f: Callable
    lib: Callable
    lo: Tuple[float, ...]
    hi: Tuple[float, ...]
    unsafe: Callable
    ic_lo: Tuple[float, ...]
    ic_hi: Tuple[float, ...]
    u_max: float
    n_grid: Tuple[int, ...]
    n_u: int
    quadP: object = None
    misspecified: bool = False
    expl: float = 0.5
    rho: float = 1e-2
    safe_box: torch.Tensor = None

    @property
    def lo_t(self):
        return torch.tensor(self.lo, dtype=DT)

    @property
    def hi_t(self):
        return torch.tensor(self.hi, dtype=DT)


PENDULUM = Spec(
    name="pendulum", n=2, f=pendulum_f, lib=pendulum_lib,
    lo=(-2.0, -2.0), hi=(2.0, 2.0),
    unsafe=lambda x: x.norm(dim=-1) >= 1.5,
    ic_lo=(-0.35, -0.35), ic_hi=(0.35, 0.35), u_max=30.0,
    n_grid=(161, 161), n_u=61, quadP=[[3.0, 1.0], [1.0, 1.0]],
)

CP_SAFE = torch.tensor([1.0, 0.5, 1.5, 2.0], dtype=DT)
CARTPOLE = Spec(
    name="cartpole", n=4, f=cartpole_f, lib=cartpole_lib,
    lo=(-1.25, -0.625, -1.875, -2.5), hi=(1.25, 0.625, 1.875, 2.5),
    unsafe=lambda x: (x.abs() >= CP_SAFE).any(-1),
    ic_lo=tuple((-0.8 * CP_SAFE).tolist()), ic_hi=tuple((0.8 * CP_SAFE).tolist()), u_max=20.0,
    n_grid=(21, 21, 21, 21), n_u=41, expl=4.0, rho=1e-4,
)

SPECS = {"pendulum": PENDULUM, "cartpole": CARTPOLE}


TP_l1, TP_l2 = 0.5, 0.35


def twopole_f(x, u):
    """Two poles of different lengths on one cart, cart acceleration as the single input;
    state (p, theta1, theta2, p_dot, theta1_dot, theta2_dot)."""
    t1, t2, pd, t1d, t2d = x[..., 1], x[..., 2], x[..., 3], x[..., 4], x[..., 5]
    a = u[..., 0]
    t1dd = (CP_g * torch.sin(t1) - torch.cos(t1) * a) / TP_l1 - CP_b * t1d
    t2dd = (CP_g * torch.sin(t2) - torch.cos(t2) * a) / TP_l2 - CP_b * t2d
    return torch.stack([pd, t1d, t2d, a, t1dd, t2dd], -1)


def twopole_lib(x):
    t1, t2, pd, t1d, t2d = x[..., 1], x[..., 2], x[..., 3], x[..., 4], x[..., 5]
    one = torch.ones_like(t1)
    psi = torch.stack([one, torch.sin(t1), torch.sin(t2), pd, t1d, t2d], -1)
    return psi, torch.stack([one, torch.cos(t1), torch.cos(t2)], -1)


TP_SAFE = torch.tensor([1.0, 0.4, 0.4, 1.5, 2.0, 2.0], dtype=DT)
TWOPOLE = Spec(
    name="twopole", n=6, f=twopole_f, lib=twopole_lib,
    lo=tuple((-1.25 * TP_SAFE).tolist()), hi=tuple((1.25 * TP_SAFE).tolist()),
    unsafe=lambda x: (x.abs() >= TP_SAFE).any(-1),
    ic_lo=tuple((-0.8 * TP_SAFE).tolist()), ic_hi=tuple((0.8 * TP_SAFE).tolist()), u_max=30.0,
    n_grid=(9,) * 6, n_u=41, expl=6.0, rho=1e-4, safe_box=TP_SAFE,
)
SPECS["twopole"] = TWOPOLE


def jacobians_at_goal(fn, n):
    x0 = torch.zeros(1, n, dtype=DT)
    u0 = torch.zeros(1, 1, dtype=DT)
    A = torch.autograd.functional.jacobian(lambda x: fn(x, u0)[0], x0)[:, 0, :]
    B = torch.autograd.functional.jacobian(lambda u: fn(x0, u)[0], u0)[:, 0, :]
    return A.numpy(), B.numpy()


def lqr(fn, n, q=1.0, r=1.0):
    A, B = jacobians_at_goal(fn, n)
    P = scipy.linalg.solve_continuous_are(A, B, q * np.eye(n), r * np.eye(1))
    K = np.linalg.solve(r * np.eye(1), B.T @ P)
    return torch.tensor(K, dtype=DT), A, B


# ---------------------------------------------------------------------------
# Data, ridge model, envelope (Prop. prop:envelope)
# ---------------------------------------------------------------------------
def collect(spec, num_traj, length, sigma, seed, dt=0.01, expl=None):
    """Closed-loop LQR (true linearization) + exploration; exact derivatives + noise."""
    gen = torch.Generator().manual_seed(seed)
    K, _, _ = lqr(spec.f, spec.n)
    lo, hi = torch.tensor(spec.ic_lo, dtype=DT), torch.tensor(spec.ic_hi, dtype=DT)
    x = lo + (hi - lo) * torch.rand(num_traj, spec.n, generator=gen, dtype=DT)
    expl = expl if expl is not None else spec.expl
    active = torch.ones(num_traj, dtype=torch.bool)
    Xs, Us, Fs = [], [], []
    for _ in range(length):
        u = -(x @ K.T) + expl * torch.randn(num_traj, 1, generator=gen, dtype=DT)
        u = u.clamp(-spec.u_max, spec.u_max)
        fx = spec.f(x, u)
        Xs.append(x[active]); Us.append(u[active]); Fs.append(fx[active])
        x = x + dt * fx
        active &= ((x > spec.lo_t) & (x < spec.hi_t)).all(-1)
        if not bool(active.any()):
            break
    X, U, F = torch.cat(Xs), torch.cat(Us), torch.cat(Fs)
    rng = np.random.default_rng(10_000 + seed)
    Y = F.numpy() + sigma * rng.standard_normal(F.shape)
    return X, U, torch.tensor(Y, dtype=DT)


class Model:
    def __init__(self, spec, Xi):
        self.spec, self.Xi = spec, torch.as_tensor(Xi, dtype=DT)
        self.pd = spec.lib(torch.zeros(1, spec.n, dtype=DT))[0].shape[-1]

    def phi(self, x, u):
        psi, chi = self.spec.lib(x)
        return torch.cat([psi, u * chi], -1)

    def drift_gain(self, x):
        psi, chi = self.spec.lib(x)
        return psi @ self.Xi[:self.pd], chi @ self.Xi[self.pd:]

    def f(self, x, u):
        f0, g = self.drift_gain(x)
        return f0 + g * u


class Envelope:
    """r(x,u) = c_N ||phi(x,u)||_{G^{-1}} + m_bar."""

    def __init__(self, spec, Ginv, c_N, beta_N, m_bar, kappa):
        self.spec, self.Ginv, self.c_N = spec, torch.as_tensor(Ginv, dtype=DT), c_N
        self.beta_N, self.m_bar, self.kappa = beta_N, m_bar, kappa
        self.pd = spec.lib(torch.zeros(1, spec.n, dtype=DT))[0].shape[-1]

    def r(self, x, U):
        """x (B,n), U (B,J) or (J,) -> (B,J)."""
        psi, chi = self.spec.lib(x)
        M, p = self.Ginv, self.pd
        a = ((psi @ M[:p, :p]) * psi).sum(-1)
        b = ((psi @ M[:p, p:]) * chi).sum(-1)
        cc = ((chi @ M[p:, p:]) * chi).sum(-1)
        if U.dim() == 1:
            U = U.expand(x.shape[0], -1)
        q = a[:, None] + 2 * b[:, None] * U + cc[:, None] * U ** 2
        return self.c_N * torch.sqrt(q.clamp(min=0)) + self.m_bar


def fit(spec, X, U, Y, sigma, S, alpha=0.1, rho=None, m_bar=0.0):
    rho = spec.rho if rho is None else rho
    model0 = Model(spec, torch.zeros(1))
    Phi = model0.phi(X, U).numpy()
    N, p = Phi.shape
    n = Y.shape[1]
    G = Phi.T @ Phi + rho * np.eye(p)
    Xi = np.linalg.solve(G, Phi.T @ Y.numpy())
    _, logdet = np.linalg.slogdet(G)
    beta = sigma * math.sqrt(max(2 * math.log(n / alpha) + logdet - p * math.log(rho), 0.0)) \
        + math.sqrt(rho) * S
    c_N = math.sqrt(n) * (beta + math.sqrt(N) * m_bar)
    kappa = float(np.linalg.eigvalsh(Phi.T @ Phi).min() / N)
    return Model(spec, Xi), Envelope(spec, np.linalg.inv(G), c_N, beta, m_bar, kappa)


def oracle_misspecification(spec, n_samp=200_000, seed=123):
    """Least-squares projection of the true field on the library over X x U (oracle)."""
    gen = torch.Generator().manual_seed(seed)
    x = spec.lo_t + (spec.hi_t - spec.lo_t) * torch.rand(n_samp, spec.n, generator=gen, dtype=DT)
    u = spec.u_max * (2 * torch.rand(n_samp, 1, generator=gen, dtype=DT) - 1)
    Phi = Model(spec, torch.zeros(1)).phi(x, u)
    F = spec.f(x, u)
    Xi = torch.linalg.lstsq(Phi, F).solution
    m_bar = float((F - Phi @ Xi).norm(dim=-1).max())
    return Xi, m_bar


# ---------------------------------------------------------------------------
# Certificates
# ---------------------------------------------------------------------------
class QuadCert:
    def __init__(self, P):
        self.P = torch.as_tensor(P, dtype=DT)
        self.H = float(torch.linalg.eigvalsh(self.P).max())

    def value_grad(self, x):
        x = x.to(DT)
        return 0.5 * ((x @ self.P) * x).sum(-1), x @ self.P

    def hess_norm(self, x):
        return torch.full((x.shape[0],), self.H, dtype=DT)


class NetCert(nn.Module):
    """V(x) = 1/2 |phi(x/s) - phi(0)|^2 + eps/2 |x/s|^2, s = box half-widths."""

    def __init__(self, spec, hidden=64, eps=0.1):
        super().__init__()
        self.scale = (spec.hi_t - spec.lo_t) / 2
        self.phi = nn.Sequential(nn.Linear(spec.n, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                 nn.Tanh(), nn.Linear(hidden, hidden)).double()
        self.eps = eps

    def forward(self, x):
        z = x.to(DT) / self.scale
        d = self.phi(z) - self.phi(torch.zeros(1, z.shape[-1], dtype=DT))
        return 0.5 * (d * d).sum(-1) + 0.5 * self.eps * (z * z).sum(-1)

    def value_grad(self, x, create_graph=False):
        with torch.enable_grad():
            x = x.to(DT).detach().requires_grad_(True)
            V = self.forward(x)
            (g,) = torch.autograd.grad(V.sum(), x, create_graph=create_graph)
        if create_graph:
            return V, g
        return V.detach(), g.detach()

    def hess_norm(self, x, step=1e-4):
        cols = []
        for j in range(x.shape[1]):
            e = torch.zeros_like(x)
            e[:, j] = step
            cols.append((self.value_grad(x + e)[1] - self.value_grad(x - e)[1]) / (2 * step))
        Hm = torch.stack(cols, -1)
        Hm = 0.5 * (Hm + Hm.transpose(1, 2))
        return torch.linalg.eigvalsh(Hm).abs().amax(-1)


class ResCert(NetCert):
    """V(x) = 1/2 x^T P x * exp(a(x)), a a tanh network with a(0) = 0: a learned reshaping of
    a quadratic certificate. The last layer starts at zero, so training starts from P."""

    def __init__(self, spec, P, hidden=64):
        super().__init__(spec, hidden=hidden, eps=0.0)
        self.P = torch.as_tensor(P, dtype=DT)
        self.phi[-1] = nn.Linear(hidden, 1).double()
        nn.init.zeros_(self.phi[-1].weight)
        nn.init.zeros_(self.phi[-1].bias)

    def forward(self, x):
        x = x.to(DT)
        z = x / self.scale
        a = (self.phi(z) - self.phi(torch.zeros(1, z.shape[-1], dtype=DT)))[:, 0]
        return 0.5 * ((x @ self.P) * x).sum(-1) * torch.exp(a)


def quad_cert_from_model(spec, model):
    if spec.quadP is not None:
        return QuadCert(spec.quadP)
    K, A, B = lqr(model.f, spec.n)
    Acl = A - B @ K.numpy()
    P = scipy.linalg.solve_continuous_lyapunov(Acl.T, -np.eye(spec.n))
    return QuadCert(P / np.abs(P).max())


def min_d_rob(cert, model, env, x, ugrid, lam, robust=True, create_graph=False, r_scale=1.0):
    if create_graph:
        V, gV = cert.value_grad(x, create_graph=True)
    else:
        V, gV = cert.value_grad(x)
    f0, g = model.drift_gain(x)
    d = (gV * f0).sum(-1)[:, None] + (gV * g).sum(-1)[:, None] * ugrid[None, :] + lam * V[:, None]
    if robust:
        d = d + r_scale * gV.norm(dim=-1)[:, None] * env.r(x, ugrid)
    return d, V, gV


def train_netcert(spec, model, env, lam, robust=True, steps=3000, batch=4096, lr=3e-3,
                  margin=0.05, size_frac=0.7, w=10.0, seed=0, n_u=41, c=1.0, log_every=1000,
                  r_scale=1.0, init=None, size_scale=1.0, w_size=None):
    w_size = w if w_size is None else w_size
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    cert = init if init is not None else NetCert(spec)
    opt = torch.optim.Adam(cert.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    ug = torch.linspace(-spec.u_max, spec.u_max, n_u, dtype=DT)
    ic_lo = size_scale * torch.tensor(spec.ic_lo, dtype=DT)
    ic_hi = size_scale * torch.tensor(spec.ic_hi, dtype=DT)
    t0 = time.time()
    for it in range(steps + 1):
        x = spec.lo_t + (spec.hi_t - spec.lo_t) * torch.rand(batch, spec.n, generator=gen, dtype=DT)
        d, V, _ = min_d_rob(cert, model, env, x, ug, lam, robust, create_graph=True, r_scale=r_scale)
        dmin = d.min(-1).values
        inside = (V <= c).detach()
        L_dec = (torch.relu(dmin + margin * V) * inside).sum() / inside.sum().clamp(min=1)
        uns = spec.unsafe(x)
        L_lvl = torch.relu(c - V[uns]).mean() if bool(uns.any()) else V.sum() * 0
        xi = ic_lo + (ic_hi - ic_lo) * torch.rand(batch // 8, spec.n, generator=gen, dtype=DT)
        L_size = torch.relu(cert(xi) - size_frac * c).mean()
        loss = L_dec + w * L_lvl + w_size * L_size
        if it < steps:
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if it % log_every == 0 or it == steps:
            print(f"    [netcert {'robust' if robust else 'nominal'}] it {it} loss={loss.item():.4g} "
                  f"dec={L_dec.item():.3g} lvl={L_lvl.item():.3g} size={L_size.item():.3g} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    cert.eval()
    return cert


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------
def make_grid(lo, hi, n_grid):
    axes = [torch.linspace(a, b, k, dtype=DT) for a, b, k in zip(lo, hi, n_grid)]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([m.reshape(-1) for m in mesh], -1), [float(a[1] - a[0]) for a in axes]


def chunked(fn, x, size=20000):
    outs = [fn(x[i:i + size]) for i in range(0, x.shape[0], size)]
    if isinstance(outs[0], tuple):
        return tuple(torch.cat(o) for o in zip(*outs))
    return torch.cat(outs)


def minimax_level(V, shape):
    """Lowest level L with each grid point in the goal's component of {V <= L}."""
    Vg = V.numpy().reshape(shape)
    best = np.full(shape, np.inf)
    start = tuple(k // 2 for k in shape)
    best[start] = Vg[start]
    heap = [(Vg[start], start)]
    while heap:
        lv, idx = heapq.heappop(heap)
        if lv > best[idx]:
            continue
        for ax in range(len(shape)):
            for d in (-1, 1):
                j = idx[ax] + d
                if 0 <= j < shape[ax]:
                    nb = idx[:ax] + (j,) + idx[ax + 1:]
                    nl = max(lv, Vg[nb])
                    if nl < best[nb]:
                        best[nb] = nl
                        heapq.heappush(heap, (nl, nb))
    return torch.as_tensor(best.reshape(-1), dtype=DT)


def neighbour_max(vals, shape):
    """max over each grid point and its 2n axis neighbours."""
    a = vals.numpy().reshape(shape)
    out = a.copy()
    for ax in range(len(shape)):
        for d in (-1, 1):
            out = np.maximum(out, np.roll(a, d, axis=ax))
    return torch.as_tensor(out.reshape(-1), dtype=DT)


class InvalidCertificate(Exception):
    pass


def _cell_bounds(spec, cert, model, env, lam, ug, L_f, L_r, H, x, w):
    """Lemma lem:grid on boxes: centre x (B,n), half-widths w (B,n).
    Returns (fails, M, V_lo): fails = min_u [d_rob + L rho] > 0 on the cell."""
    rho = w.norm(dim=-1)
    d, V, gV = min_d_rob(cert, model, env, x, ug, lam)
    gn = gV.norm(dim=-1)
    B, J = d.shape
    fn = model.f(x[:, None, :].expand(B, J, spec.n).reshape(-1, spec.n),
                 ug[None, :, None].expand(B, J, 1).reshape(-1, 1)).norm(dim=-1).reshape(B, J)
    rr = env.r(x, ug)
    L = H * (fn + L_f * rho[:, None] + rr + L_r * rho[:, None]) \
        + (gn + H * rho)[:, None] * (L_f + lam + L_r)
    fails = (d + L * rho[:, None]).min(-1).values > 0
    M = V + gn * rho + 0.5 * H * rho ** 2
    V_lo = V - gn * rho - 0.5 * H * rho ** 2
    return fails, M, V_lo


def adaptive_cert(spec, cert, model, env, lam, ug, L_f, L_r, H, c, n_grid,
                  budget=3_000_000, top_frac=0.25, lo=None, hi=None, A=None):
    """Upper bound on tau_hat by adaptive refinement of Lemma lem:grid (variable radii).
    Cells entirely above level c are outside K_c and dropped; passing cells certify their
    states; failing cells with the largest bound M are split into 2^n children. With A,
    cells live in z-coordinates x = A z and their radius in x is at most |A| |w|."""
    zs, steps = make_grid(lo or spec.lo, hi or spec.hi, n_grid)
    normA = float(torch.linalg.matrix_norm(A, ord=2)) if A is not None else 1.0
    tx = (lambda z: z @ A.T) if A is not None else (lambda z: z)
    w = torch.tensor(steps, dtype=DT).expand(zs.shape[0], -1) / 2

    def evaluate(z, w):
        outs = [_cell_bounds(spec, cert, model, env, lam, ug, L_f, L_r, H, tx(z[i:i + 3000]), normA * w[i:i + 3000])
                for i in range(0, z.shape[0], 3000)]
        fails, M, V_lo = (torch.cat(o) for o in zip(*outs))
        keep = fails & (V_lo <= c)
        return z[keep], w[keep], M[keep]

    xs = zs

    x, w, M = evaluate(xs, w)
    used = xs.shape[0]
    signs = torch.tensor(list(np.ndindex(*(2,) * spec.n)), dtype=DT) * 2 - 1
    while x.shape[0] and used < budget:
        order = torch.argsort(M, descending=True)
        k = max(1, int(top_frac * x.shape[0]))
        k = min(k, max(1, (budget - used) // signs.shape[0]))
        top, rest = order[:k], order[k:]
        xc = (x[top][:, None, :] + 0.5 * w[top][:, None, :] * signs[None]).reshape(-1, spec.n)
        wc = (0.5 * w[top])[:, None, :].expand(-1, signs.shape[0], -1).reshape(-1, spec.n)
        used += xc.shape[0]
        xn, wn, Mn = evaluate(xc, wc)
        x, w, M = torch.cat([x[rest], xn]), torch.cat([w[rest], wn]), torch.cat([M[rest], Mn])
    return (float(M.max()) if x.shape[0] else 0.0), used


# ---------------------------------------------------------------------------
# Lipschitz constants of the model and the envelope (computable)
# ---------------------------------------------------------------------------
def model_constants(spec, model, env, xs, ugrid=None):
    """Global L_f, L_r (sup over |u| <= u_max) and, per control in ugrid, the one-sided
    rate mu(u) = sup_x lambda_max(sym J_fhat(x,u)) and L_r(u). mu and L_r(.) are convex in
    u for control-affine models, so the max over a grid cell sits at its end points."""
    Gm12 = torch.as_tensor(scipy.linalg.sqrtm(env.Ginv.numpy()).real, dtype=DT)

    def jac(fn, u_val):
        x = xs.clone().requires_grad_(True)
        u = torch.full((xs.shape[0], 1), u_val, dtype=DT)
        out = fn(x, u)
        rows = [torch.autograd.grad(out[:, i].sum(), x, retain_graph=True)[0] for i in range(out.shape[1])]
        return torch.stack(rows, 1)

    us = [s * spec.u_max for s in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    L_f = max(torch.linalg.matrix_norm(jac(model.f, u), ord=2).max().item() for u in us)
    L_r = env.c_N * max(torch.linalg.matrix_norm(Gm12 @ jac(model.phi, u), ord=2).max().item() for u in us)
    if ugrid is None:
        return L_f, L_r
    mu, lr = [], []
    for u in ugrid.tolist():
        Jf = jac(model.f, u)
        mu.append(torch.linalg.eigvalsh(0.5 * (Jf + Jf.transpose(1, 2))).max().item())
        lr.append(env.c_N * torch.linalg.matrix_norm(Gm12 @ jac(model.phi, u), ord=2).max().item())
    mu, lr = torch.tensor(mu, dtype=DT), torch.tensor(lr, dtype=DT)
    # cell-wise max of neighbours, valid for any u between grid points
    mu_c = torch.maximum(mu, torch.cat([mu[1:], mu[-1:]]))
    mu_c = torch.maximum(mu_c, torch.cat([mu[:1], mu[:-1]]))
    lr_c = torch.maximum(lr, torch.cat([lr[1:], lr[-1:]]))
    lr_c = torch.maximum(lr_c, torch.cat([lr[:1], lr[:-1]]))
    return L_f, L_r, mu_c, lr_c


# ---------------------------------------------------------------------------
# Sampled-data bounds (Thm. thm:sampled)
# ---------------------------------------------------------------------------
@dataclass
class StepConsts:
    h: float
    n_sub: int
    L_f: float
    L_r: float
    H: float
    lam: float
    ugrid: torch.Tensor = None   # controls at which mu_u, lr_u are tabulated
    mu_u: torch.Tensor = None    # one-sided rate of f_hat(., u), cell-wise max
    lr_u: torch.Tensor = None    # Lipschitz constant of r(., u), cell-wise max

    def rate(self, u):
        """Gronwall rate for each control in u (K,): one-sided if tabulated, else L_f + L_r."""
        if self.mu_u is None:
            return torch.full_like(u, self.L_f + self.L_r)
        idx = torch.bucketize(u.clamp(self.ugrid[0], self.ugrid[-1]), self.ugrid).clamp(0, len(self.ugrid) - 1)
        lo = (idx - 1).clamp(min=0)
        k = torch.maximum(self.mu_u[idx] + self.lr_u[idx], self.mu_u[lo] + self.lr_u[lo])
        return k.clamp(min=0.0)


def step_bounds(spec, cert, model, env, x, ugrid, sc: StepConsts):
    """For x (B,n), controls ugrid (J,): returns d_h (B,J) and peak P (B,J)."""
    B, J, n = x.shape[0], ugrid.shape[0], spec.n
    y = x[:, None, :].expand(B, J, n).reshape(-1, n)
    u = ugrid[None, :, None].expand(B, J, 1).reshape(-1, 1)
    dt = sc.h / sc.n_sub
    pts = [y]
    for _ in range(sc.n_sub):
        k1 = model.f(y, u)
        k2 = model.f(y + 0.5 * dt * k1, u)
        k3 = model.f(y + 0.5 * dt * k2, u)
        k4 = model.f(y + dt * k3, u)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        pts.append(y)
    P_ = torch.stack(pts, 1)  # (B*J, n_sub+1, n)
    flat = P_.reshape(-1, n)
    V, gV = chunked(cert.value_grad, flat)
    V = V.reshape(B * J, -1)
    gn = gV.norm(dim=-1).reshape(B * J, -1)
    uu = u.expand(-1, sc.n_sub + 1).reshape(-1)
    r = torch.cat([_r_pointwise(env, flat[i:i + 50000], uu[i:i + 50000])
                   for i in range(0, flat.shape[0], 50000)]).reshape(B * J, -1)
    # D+e <= (mu(u) + L_r(u)) e + r(x_hat(t), u)  =>  e(s) <= s max_t r e^{kappa s}, kappa >= 0
    r_along = r.max(-1).values
    s = torch.linspace(0, sc.h, sc.n_sub + 1, dtype=DT)
    kappa = sc.rate(u[:, 0])
    eps = r_along[:, None] * s[None, :] * torch.exp(kappa[:, None] * s[None, :])
    Vp = V + (gn + sc.H * eps) * eps
    rho = eps[:, -1][:, None, None]
    inside = ((P_ - rho > spec.lo_t) & (P_ + rho < spec.hi_t)).all(-1).all(-1)
    V0 = cert.value_grad(x)[0]
    d_h = Vp[:, -1].reshape(B, J) - math.exp(-sc.lam * sc.h) * V0[:, None]
    peak = Vp.max(-1).values.reshape(B, J)
    bad = ~inside.reshape(B, J)
    d_h = d_h.masked_fill(bad, float("inf"))
    peak = peak.masked_fill(bad, float("inf"))
    return d_h, peak


def _r_pointwise(env, x, u):
    psi, chi = env.spec.lib(x)
    phi = torch.cat([psi, u[:, None] * chi], -1)
    q = ((phi @ env.Ginv) * phi).sum(-1)
    return env.c_N * torch.sqrt(q.clamp(min=0)) + env.m_bar


# ---------------------------------------------------------------------------
# Full analysis on a grid
# ---------------------------------------------------------------------------
def analyze(spec, cert, model, env, lam=0.5, h=0.05, n_sub=10, n_grid=None, n_u_true=101,
            v_floor=1e-6, sampled=True, certify=True, fit_box=False, c_scale=1.0):
    t0 = time.time()
    n_grid = n_grid or spec.n_grid
    c_exact, lo, hi, A = None, spec.lo, spec.hi, None
    if fit_box:
        # quadratic V, box safe set: c = min_i S_i^2 / (2 (P^-1)_ii); grid the ellipsoid in
        # coordinates x = A z, A = Q diag(e)^{-1/2}, where K_c is the ball |z|^2 <= 2c
        Pinv_d = torch.diagonal(torch.linalg.inv(cert.P))
        c_exact = float((spec.safe_box ** 2 / (2 * Pinv_d)).min()) * (1 - 1e-3)
        e, Q = torch.linalg.eigh(cert.P)
        A = Q @ torch.diag(e ** -0.5)
        Rz = 1.15 * math.sqrt(2 * c_exact)
        lo, hi = (-Rz,) * spec.n, (Rz,) * spec.n
    zs, steps = make_grid(lo, hi, n_grid)
    normA = float(torch.linalg.matrix_norm(A, ord=2)) if A is not None else 1.0
    xs = zs @ A.T if A is not None else zs
    if A is not None:
        steps = [normA * s for s in steps]
    V, gV = chunked(cert.value_grad, xs)
    level = minimax_level(V, tuple(n_grid))
    boundary = ((zs <= torch.tensor(lo, dtype=DT) + 1e-9) | (zs >= torch.tensor(hi, dtype=DT) - 1e-9)).any(-1)
    exits = spec.unsafe(xs) | boundary
    c = c_exact if c_exact is not None else float(level[exits].min()) * (1 - 1e-3)
    c_full = c
    c = c_scale * c
    if fit_box and c_scale < 1.0:
        # regrid the smaller ellipsoid
        Rz = 1.15 * math.sqrt(2 * c)
        lo, hi = (-Rz,) * spec.n, (Rz,) * spec.n
        zs, steps = make_grid(lo, hi, n_grid)
        xs = zs @ A.T
        steps = [normA * s for s in steps]
        V, gV = chunked(cert.value_grad, xs)
        level = minimax_level(V, tuple(n_grid))
    K = level <= c
    V_goal = float(cert.value_grad(torch.zeros(1, spec.n, dtype=DT))[0])
    if int(K.sum()) < 10 or V_goal > c:
        raise InvalidCertificate(f"V(goal)={V_goal:.3g} c={c:.3g} |K|={int(K.sum())}")
    xK, VK, gK = xs[K], V[K], gV[K]
    ug = torch.linspace(-spec.u_max, spec.u_max, spec.n_u, dtype=DT)
    out = {"c": c, "c_full": c_full, "c_scale": c_scale, "Kc_frac": float(K.double().mean()), "n_K": int(K.sum())}

    # robust violation set and its height (eq. tauhat), with zoom near the goal
    def hrob(x):
        return min_d_rob(cert, model, env, x, ug, lam)[0].min(-1).values
    hK = chunked(hrob, xK, 5000)
    om = hK > 0
    tau_hat = float(VK[om].max()) if bool(om.any()) else 0.0
    R = (xK[om].abs().amax(0) if bool(om.any()) else torch.zeros(spec.n, dtype=DT)) + 2 * torch.tensor(steps, dtype=DT)
    for _ in range(3):
        if bool((R >= (spec.hi_t - spec.lo_t) / 4).any()):
            break
        xz, _ = make_grid(tuple((-R).tolist()), tuple(R.tolist()), n_grid)
        Vz = chunked(lambda z: cert.value_grad(z)[0], xz)
        keep = (Vz <= c) & ~spec.unsafe(xz)
        xz, Vz = xz[keep], Vz[keep]
        omz = chunked(hrob, xz, 5000) > 0
        if not bool(omz.any()):
            break
        tau_hat = max(tau_hat, float(Vz[omz].max()))
        Rn = xz[omz].abs().amax(0) + 4 * R / (torch.tensor(n_grid, dtype=DT) - 1)
        if bool((Rn >= 0.9 * R).all()):
            break
        R = Rn
    out.update({"tau_hat": tau_hat, "omega_frac_Kc": float(om.double().mean())})

    # envelope level, true error, margin constants (true f: evaluation only)
    # same control grid as h_rob, so grid versions of the bounds compare exactly
    # (a grid min over u is >= the true inf, which only lowers lambda_bar: conservative)
    ut = ug

    def env_and_err(x):
        rr = env.r(x, ut)
        y = x[:, None, :].expand(-1, ut.shape[0], -1).reshape(-1, spec.n)
        uu = ut[None, :, None].expand(x.shape[0], -1, 1).reshape(-1, 1)
        e = (spec.f(y, uu) - model.f(y, uu)).norm(dim=-1).reshape(x.shape[0], -1)
        Vx, gx = cert.value_grad(x)
        lf = (spec.f(y, uu).reshape(x.shape[0], -1, spec.n) * gx[:, None, :]).sum(-1)
        return rr, e, lf
    rK, eK, lfK = chunked(env_and_err, xK, 2000)
    out.update({"delta_c_env": float(rK.max()), "delta_c_true": float(eK.max()),
                "env_valid": int(bool((eK <= rK + 1e-9).all())),
                "env_ratio_max": float((eK / rK.clamp(min=1e-12)).max())})
    pos = VK > v_floor
    lam_bar = float((-lfK.min(-1).values[pos] / VK[pos]).min())
    beta_V = float((gK[pos].norm(dim=-1) / VK[pos].sqrt()).max())
    applicable = lam_bar > lam
    tau_delta = (2 * beta_V * out["delta_c_env"] / (lam_bar - lam)) ** 2 if applicable else float("nan")
    out.update({"lambda_bar": lam_bar, "beta_V": beta_V, "tau_delta": tau_delta,
                "delta_star": (lam_bar - lam) * math.sqrt(c) / (2 * beta_V) if applicable else float("nan"),
                "thm2_applicable": int(applicable),
                "thm2_holds": int((not applicable) or tau_hat <= tau_delta * (1 + 1e-6) + 1e-12)})

    # local margin bound (Prop. prop:margin-local): Omega_rob is contained in
    # {x : min_u [L_f V + lam V + |gradV| (r + |f - f_hat|)] > 0}, and |f - f_hat| <= r.
    gnorm = gK.norm(dim=-1)
    hb_e = (lfK + lam * VK[:, None] + gnorm[:, None] * (rK + eK)).min(-1).values
    hb_2r = (lfK + lam * VK[:, None] + 2 * gnorm[:, None] * rK).min(-1).values
    tau_loc = float(VK[hb_2r > 0].max()) if bool((hb_2r > 0).any()) else 0.0
    tau_loc_e = float(VK[hb_e > 0].max()) if bool((hb_e > 0).any()) else 0.0
    out.update({"tau_loc": tau_loc, "tau_loc_e": tau_loc_e,
                "tau_hat_grid": float(VK[om].max()) if bool(om.any()) else 0.0})
    out["tau_loc_holds"] = int(out["tau_hat_grid"] <= tau_loc_e * (1 + 1e-6) + 1e-12)

    # constants for Lemma lem:grid and Thm. thm:sampled
    xfull = spec.lo_t + (spec.hi_t - spec.lo_t) * torch.rand(20000, spec.n, generator=torch.Generator().manual_seed(7), dtype=DT)
    xfull = torch.cat([xfull, xs[::max(1, xs.shape[0] // 20000)]])
    L_f, L_r, mu_u, lr_u = model_constants(spec, model, env, xfull, ug)
    Hx = chunked(cert.hess_norm, xs, 5000)
    H_loc = 1.25 * neighbour_max(Hx, tuple(n_grid))
    H_glob = float(1.25 * Hx.max())
    out.update({"L_f": L_f, "L_r": L_r, "H_glob": H_glob})

    # grid certification (Lemma lem:grid)
    if certify:
        rho = 0.5 * math.sqrt(sum(s ** 2 for s in steps))
        gn_all = gV.norm(dim=-1)
        G = level <= c + (gn_all + H_loc * rho) * rho
        xG, VG, gnG, HG = xs[G], V[G], gn_all[G], H_loc[G]

        def cert_bound(args):
            x, Hl, gn = args
            d, _, _ = min_d_rob(cert, model, env, x, ug, lam)
            fn = model.f(x[:, None, :].expand(-1, ug.shape[0], -1).reshape(-1, spec.n),
                         ug[None, :, None].expand(x.shape[0], -1, 1).reshape(-1, 1)).norm(dim=-1).reshape(x.shape[0], -1)
            rr = env.r(x, ug)
            L = Hl[:, None] * (fn + L_f * rho + rr + L_r * rho) + (gn[:, None] + Hl[:, None] * rho) * (L_f + lam + L_r)
            return (d + L * rho).min(-1).values
        bnd = torch.cat([cert_bound((xG[i:i + 4000], HG[i:i + 4000], gnG[i:i + 4000]))
                         for i in range(0, xG.shape[0], 4000)])
        Gp = bnd > 0
        tau_cert = float((VG[Gp] + (gnG[Gp] + HG[Gp] * rho) * rho).max()) if bool(Gp.any()) else 0.0
        out.update({"tau_hat_cert": tau_cert, "cover_radius": rho})
        ta, ncells = adaptive_cert(spec, cert, model, env, lam, ug, L_f, L_r, H_glob, c, n_grid, lo=lo, hi=hi, A=A)
        out.update({"tau_hat_cert_adapt": ta, "cert_cells": ncells})

    # computable sampled-data theorem (Thm. thm:sampled)
    if sampled:
        sc = StepConsts(h=h, n_sub=n_sub, L_f=L_f, L_r=L_r, H=H_glob, lam=lam,
                        ugrid=ug, mu_u=mu_u, lr_u=lr_u)
        dh_list, pk_list = [], []
        for i in range(0, xK.shape[0], max(1, 60000 // spec.n_u)):
            dh, pk = step_bounds(spec, cert, model, env, xK[i:i + max(1, 60000 // spec.n_u)], ug, sc)
            dh_list.append(dh); pk_list.append(pk)
        dh, pk = torch.cat(dh_list), torch.cat(pk_list)
        hmin, jmin = dh.min(-1)
        omh = hmin > 0
        ehl = math.exp(-lam * h)
        tau_h = float(VK[omh].max()) if bool(omh.any()) else 0.0
        J_h = float((hmin[omh] + ehl * VK[omh]).max()) if bool(omh.any()) else 0.0
        # the sampled-data violation set can sit between grid points near the goal
        Rz = 2 * torch.tensor(steps, dtype=DT)
        nz = (41,) * spec.n if spec.n <= 2 else ((11,) * spec.n if spec.n <= 4 else (5,) * spec.n)
        for _ in range(4):
            xz, _ = make_grid(tuple((-Rz).tolist()), tuple(Rz.tolist()), nz)
            Vz = chunked(lambda z: cert.value_grad(z)[0], xz)
            xz, Vz = xz[Vz <= c], Vz[Vz <= c]
            dz = torch.cat([step_bounds(spec, cert, model, env, xz[i:i + 1500], ug, sc)[0]
                            for i in range(0, xz.shape[0], 1500)])
            hz = dz.min(-1).values
            oz = hz > 0
            if bool(oz.any()):
                tau_h = max(tau_h, float(Vz[oz].max()))
                J_h = max(J_h, float((hz[oz] + ehl * Vz[oz]).max()))
            Rz = Rz / 4
        feas = dh <= 0
        P_adm = torch.where(omh, pk.gather(1, jmin[:, None])[:, 0],
                            torch.where(feas, pk, torch.full_like(pk, -float("inf"))).max(-1).values)
        order = torch.argsort(VK)
        Vs, Ps = VK[order], torch.cummax(P_adm[order], 0).values
        floor = max(tau_h, J_h)
        i_floor = int(torch.searchsorted(Vs, torch.tensor([floor], dtype=DT), right=True)[0]) - 1
        ult = float(Ps[max(i_floor, 0)])
        ok = Ps < c
        m0 = 0.0
        if ult < c:
            bad_idx = torch.nonzero(~ok)
            last = int(bad_idx[0]) - 1 if bad_idx.numel() else Vs.shape[0] - 1
            m0 = float(Vs[max(last, 0)])
        out.update({"tau_hat_h": tau_h, "J_h": J_h, "ult_h": ult, "m0_h": m0,
                    "omega_h_frac": float(omh.double().mean()),
                    "gronwall_factor": math.exp((L_f + L_r) * h),
                    "gronwall_factor_mu": float(torch.exp(sc.rate(ug) * h).max()),
                    "gronwall_factor_mu_med": float(torch.exp(sc.rate(ug) * h).median())})
        out["_sc"] = sc
    out["_xK"], out["_VK"] = xK, VK
    # largest level c' <= c whose sublevel set is certified with margin tau_hat(c') <= c'/2
    Vo = VK[om]
    cands = torch.sort(VK).values
    best_c, best_frac = 0.0, 0.0
    for cp in cands[torch.linspace(0, cands.shape[0] - 1, 400).long()].tolist():
        t = float(Vo[Vo <= cp].max()) if bool((Vo <= cp).any()) else 0.0
        if t <= 0.5 * cp:
            best_c, best_frac = cp, float((VK <= cp).double().sum()) / xs.shape[0]
    out.update({"c_cert": best_c, "cert_vol_frac": best_frac})
    out["analysis_seconds"] = time.time() - t0
    return out


# ---------------------------------------------------------------------------
# Policies and rollouts
# ---------------------------------------------------------------------------
def make_policy(kind, spec, cert, model, env, lam, sc=None, n_u=201):
    K, _, _ = lqr(model.f, spec.n)
    ug = torch.linspace(-spec.u_max, spec.u_max, n_u, dtype=DT)
    ugd = torch.linspace(-spec.u_max, spec.u_max, 81, dtype=DT)

    def nominal(x):
        return (-(x @ K.T)).clamp(-spec.u_max, spec.u_max)

    def project(x, up, d, grid):
        cand = torch.cat([up, grid.expand(x.shape[0], -1)], -1)
        dd = torch.cat([d[0], d[1]], -1)
        feas = dd <= 0
        dist = (cand - up).abs().masked_fill(~feas, float("inf"))
        j = torch.where(feas.any(-1), dist.argmin(-1), dd.argmin(-1))
        return cand.gather(1, j[:, None]), feas.any(-1)

    def pi(x):
        up = nominal(x)
        if kind == "lqr":
            return up, None
        if kind == "lqr_filter":
            d_up = min_d_rob(cert, model, env, x, up[:, 0], lam)[0].diagonal()[:, None]
            d_g = min_d_rob(cert, model, env, x, ug, lam)[0]
            return project(x, up, (d_up, d_g), ug)
        if kind == "lqr_dfilter":
            # same control grid as analyze(), so the policy is the one Thm. thm:sampled certifies
            ugd = torch.linspace(-spec.u_max, spec.u_max, spec.n_u, dtype=DT)
            d_g = step_bounds(spec, cert, model, env, x, ugd, sc)[0]
            feas = d_g <= 0
            dist = (ugd[None, :] - up).abs().masked_fill(~feas, float("inf"))
            j = torch.where(feas.any(-1), dist.argmin(-1), d_g.argmin(-1))
            return ugd[j][:, None], feas.any(-1)
        raise ValueError(kind)

    return pi


def rollout(spec, cert, pi, x0, c, t_sim=10.0, h=0.05, dt=0.01):
    x = x0.clone()
    V0 = cert.value_grad(x)[0]
    Vs = [V0]
    unsafe = spec.unsafe(x)
    sub = int(round(h / dt))
    Vk = [V0]
    for _ in range(int(round(t_sim / h))):
        u, _ = pi(x)
        for _ in range(sub):
            k1 = spec.f(x, u)
            k2 = spec.f(x + 0.5 * dt * k1, u)
            k3 = spec.f(x + 0.5 * dt * k2, u)
            k4 = spec.f(x + dt * k3, u)
            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            bad = ~torch.isfinite(x).all(-1) | ((x < spec.lo_t) | (x > spec.hi_t)).any(-1)
            x = torch.where(bad[:, None], torch.zeros_like(x) + spec.hi_t, x)
            unsafe |= spec.unsafe(x) | bad
            Vs.append(cert.value_grad(x)[0])
        Vk.append(Vs[-1])
    Vt = torch.stack(Vs, 1)
    tail = Vt[:, int(0.8 * Vt.shape[1]):]
    safe = (Vt.max(1).values < c) & ~unsafe
    return {"V0": V0, "Vmax": Vt.max(1).values, "tail": tail.max(1).values, "safe": safe,
            "unsafe": unsafe, "Vk": torch.stack(Vk, 1)}
