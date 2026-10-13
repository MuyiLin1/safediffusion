"""
A learned certificate trained on the robust decrease condition of the theorems
(PAPER_PLAN.md, Part 3, C1-C3).

  V(x) = 1/2 |phi(x) - phi(0)|^2 + eps/2 |x|^2,   phi a tanh MLP,

so V(0) = 0 and V > 0 elsewhere by construction. Training uses only the learned model
f_hat(x,u) = f0(x) + g u and its envelope r_N (Prop. 5), never the true dynamics:

  decrease : relu(min_{|u|<=u_max} d_rob(x,u) + margin*V)   on {V <= c}
  level    : relu(c - V)                                     on {||x|| >= 1.5}
  size     : relu(V - size_frac*c)                           on {||x|| <= 0.5}

with d_rob = gradV.f_hat + lam V + |gradV| r_N (eq. robust-residual) and c = 1.
"robust" uses r_N; "nominal" sets r_N = 0 (trusts f_hat, like the original training).
"""
import math
import time

import torch
import torch.nn as nn

from Lyapunov_guided_diffusion.sindy.theory_tools import STATE_BOX, UNSAFE_RADIUS, Envelope


class LearnedCertificate(nn.Module):
    def __init__(self, hidden=64, layers=2, eps=0.1):
        super().__init__()
        mods, d = [], 2
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        mods.append(nn.Linear(d, hidden))
        self.phi = nn.Sequential(*mods).double()
        self.eps = eps

    def forward(self, x):
        x = x.double()
        z = self.phi(x) - self.phi(torch.zeros(1, 2, dtype=x.dtype))
        return 0.5 * (z * z).sum(-1) + 0.5 * self.eps * (x * x).sum(-1)

    def value_and_grad(self, x, create_graph=False):
        x = x.double().detach().requires_grad_(True)
        V = self.forward(x)
        (g,) = torch.autograd.grad(V.sum(), x, create_graph=create_graph)
        return V, g

    def V_with_jacobian(self, x):
        """Same interface as NeuralCLBFController: (V (B,), JV (B,1,n))."""
        with torch.enable_grad():
            V, g = self.value_and_grad(x)
        return V.detach().to(x.dtype), g.detach()[:, None, :].to(x.dtype)


def min_d_rob(V, gV, x, f0, gvec, env: Envelope, lam, u_grid, robust):
    """min over the control grid of d_rob(x,u); f_hat = f0(x) + gvec u (control-affine)."""
    LfV = (gV * f0).sum(-1)
    LgV = gV @ gvec
    d = LfV[:, None] + LgV[:, None] * u_grid[None, :] + lam * V[:, None]
    if robust:
        r = env.r(x, u_grid.expand(x.shape[0], -1))
        d = d + gV.norm(dim=-1)[:, None] * r
    return d.min(dim=-1).values


def _sample(n, gen):
    lo = torch.tensor([b[0] for b in STATE_BOX], dtype=torch.float64)
    hi = torch.tensor([b[1] for b in STATE_BOX], dtype=torch.float64)
    return lo + (hi - lo) * torch.rand(n, 2, generator=gen, dtype=torch.float64)


def train_certificate(model, env: Envelope, robust=True, lam=0.5, u_max=30.0, c=1.0,
                      steps=3000, batch=4096, lr=3e-3, margin=0.05, size_frac=0.7,
                      w_level=10.0, w_size=10.0, seed=0, n_u=61, log_every=500):
    """model: LinearFeatureModel (has drift_and_gain). Returns (cert, final-loss dict)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    cert = LearnedCertificate()
    opt = torch.optim.Adam(cert.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    u_grid = torch.linspace(-u_max, u_max, n_u, dtype=torch.float64)
    t0 = time.time()
    stats = {}
    for it in range(steps + 1):
        x = _sample(batch, gen)
        V, gV = cert.value_and_grad(x, create_graph=True)
        f0, gvec = model.drift_and_gain(x)
        d = min_d_rob(V, gV, x, f0, gvec, env, lam, u_grid, robust)
        inside = (V <= c).detach()
        L_dec = (torch.relu(d + margin * V) * inside).sum() / inside.sum().clamp(min=1)
        norms = x.norm(dim=-1)
        out = norms >= UNSAFE_RADIUS
        L_lvl = torch.relu(c - V[out]).mean() if bool(out.any()) else V.sum() * 0
        ball = norms <= 0.5
        L_size = torch.relu(V[ball] - size_frac * c).mean() if bool(ball.any()) else V.sum() * 0
        loss = L_dec + w_level * L_lvl + w_size * L_size
        if it < steps:
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if it % log_every == 0 or it == steps:
            stats = {"loss": float(loss), "L_dec": float(L_dec), "L_level": float(L_lvl),
                     "L_size": float(L_size),
                     "viol_frac": float(((d > 0) & inside).sum() / inside.sum().clamp(min=1))}
            print(f"    [cert {'robust' if robust else 'nominal'}] it {it:5d} "
                  + " ".join(f"{k}={v:.4g}" for k, v in stats.items())
                  + f" ({time.time() - t0:.0f}s)", flush=True)
    cert.eval()
    return cert, stats


def save_certificate(cert, path, meta=None):
    torch.save({"state_dict": cert.state_dict(), "eps": cert.eps, "meta": meta or {}}, path)


def load_certificate(path):
    blob = torch.load(path, map_location="cpu")
    cert = LearnedCertificate(eps=blob.get("eps", 0.1))
    cert.load_state_dict(blob["state_dict"])
    cert.eval()
    return cert


def attach(controller, cert):
    """Make the existing controller (CLF-QP and diffusion sampler) use the learned V."""
    controller.V_with_jacobian = cert.V_with_jacobian
    return controller
