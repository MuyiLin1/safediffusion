"""
Measure the empirical stabilising margin lambda_bar to make the calibration
paragraph (OPEN 6) a real, non-circular test.

The paper's formula is:
    delta* = Theta( (lambda_bar - lambda) * (c / C')^2 )

lambda (the training target) is a fixed known constant (1.0).
lambda_bar is the ACTUAL exponential decay rate of V under the TRUE system
and the trained controller, measured here without using delta* at all.
C' is estimated from the network's Lipschitz constants.

Together these give a predicted delta* that can be compared to the
observed safety cliff (~11 from Table 1), turning the "consistent with
order-one constants" sentence into a falsifiable prediction.

Usage:
    conda run -n dmwithllm python -m Lyapunov_guided_diffusion.sindy.measure_lambda_bar \
        --checkpoint logs/inverted_pendulum_sindy/commit_b6d4525/version_3/checkpoints/epoch=50-step=7191.ckpt \
        --sindy_model data/sindy_model_coefficients.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))

# Training parameters (fixed, must match what the checkpoint was trained with)
CLF_LAMBDA = 1.0          # lambda: the training dissipation rate
SAFE_LEVEL = 1.0          # c: the safe sublevel set threshold
OPERATING_DOMAIN = ((-1.0, 1.0), (-1.0, 1.0))


def _load_controller(ckpt_path: str, sindy_model_path: str):
    from neural_clbf.systems import InvertedPendulum
    from neural_clbf.systems.sindy_estimated_system import SINDyEstimatedSystem
    from neural_clbf.controllers import NeuralCLBFController
    from Lyapunov_guided_diffusion.sindy.identify_dynamics import identify_dynamics
    from Lyapunov_guided_diffusion.sindy.collect_data import collect_safe_transitions
    from Lyapunov_guided_diffusion.training.train_inverted_pendulum_sindy import build_and_train

    nominal_params = {"m": 1.0, "L": 1.0, "b": 0.01}
    simulation_dt = 0.01
    controller_period = 0.05
    scenarios = [nominal_params]

    true_system = InvertedPendulum(
        nominal_params, dt=simulation_dt,
        controller_dt=controller_period, scenarios=scenarios,
    )

    # Rebuild sindy_system from scratch at sigma=0 seed=0 so it matches what
    # the standalone checkpoint was trained on.
    import numpy as np
    torch.manual_seed(0); np.random.seed(0)
    X, U, Xdot = collect_safe_transitions(
        true_system, num_trajectories=50, trajectory_length=100,
        exploration_std=0.5, obs_noise_std=0.0, seed=0,
    )
    model = identify_dynamics(X, U, Xdot, poly_degree=1)
    sindy_system = SINDyEstimatedSystem(
        model, reference_system=true_system,
        dt=simulation_dt, controller_dt=controller_period, scenarios=scenarios,
    )

    controller = build_and_train(
        sindy_system, scenarios, max_epochs=51,
        accelerator="cpu", enable_logger=False,
    )
    # Overwrite the randomly-initialized weights with the saved checkpoint.
    ckpt = torch.load(ckpt_path, map_location="cpu")
    controller.load_state_dict(ckpt["state_dict"], strict=False)
    controller.eval()
    return controller, true_system


def fit_lambda_bar(V_values: np.ndarray, t_array: np.ndarray) -> tuple[float, float]:
    """Fit V(t) = V0 * exp(-lambda_bar * t) per rollout; return mean ± std."""
    # Only use rollouts that decayed (initial V > final V) to avoid fitting noise.
    rates = []
    for i in range(V_values.shape[0]):
        V = V_values[i]
        V0 = V[0]
        if V0 <= 1e-6:
            continue
        # log-linear fit: log(V(t)/V0) = -lambda_bar * t
        log_ratio = np.log(np.maximum(V / V0, 1e-9))
        # Weighted least squares, ignoring the last 5% (may have plateaued)
        k = max(1, int(0.95 * len(t_array)))
        slope = np.polyfit(t_array[:k], log_ratio[:k], deg=1)[0]
        rates.append(-slope)

    rates = [r for r in rates if np.isfinite(r) and r > 0]
    if not rates:
        return float("nan"), float("nan")
    return float(np.mean(rates)), float(np.std(rates))


def estimate_C_prime(controller, n_grid: int = 40) -> float:
    """Estimate C' = B^2 * (2 + L_pi) / lambda_1 from network constants.

    From Step 2 of the main theorem proof, the buffer constant is
        C_0 = ||nabla V||_inf * ||nabla f||_inf * (1 + L_pi) + ||nabla^2 V||_inf * ||f||_inf
    and C' = C_0 / lambda_1.  We estimate each factor over the operating grid.
    """
    from neural_clbf.systems import InvertedPendulum
    nominal_params = {"m": 1.0, "L": 1.0, "b": 0.01}
    true_system = InvertedPendulum(nominal_params, dt=0.01, controller_dt=0.05,
                                   scenarios=[nominal_params])

    axes = [torch.linspace(lo, hi, n_grid) for (lo, hi) in OPERATING_DOMAIN]
    mesh = torch.meshgrid(*axes, indexing="ij")
    x = torch.stack([m.reshape(-1) for m in mesh], dim=-1).float()

    # Estimate ||nabla V||_inf
    x_v = x.clone().requires_grad_(True)
    V = controller.V(x_v)
    grad_V = torch.autograd.grad(V.sum(), x_v)[0]
    B_gradV = float(grad_V.norm(dim=-1).max().item())

    # Estimate ||nabla^2 V||_inf (Frobenius norm of Hessian, diagonal approximation)
    H_diag_norms = []
    for j in range(x.shape[1]):
        g_j = grad_V[:, j]
        x_v2 = x.clone().requires_grad_(True)
        V2 = controller.V(x_v2)
        grad2 = torch.autograd.grad(V2.sum(), x_v2, create_graph=True)[0]
        h_row = torch.autograd.grad(grad2[:, j].sum(), x_v2, retain_graph=True)[0]
        H_diag_norms.append(float(h_row.norm(dim=-1).max().item()))
    B_hessV = max(H_diag_norms)

    # Estimate ||f||_inf and ||nabla f||_inf over the grid
    with torch.no_grad():
        u_zero = torch.zeros(x.shape[0], true_system.n_controls)
        xdot = true_system.closed_loop_dynamics(x, u_zero)
    B_f = float(xdot.norm(dim=-1).max().item())

    # Estimate ||nabla f||_inf numerically
    eps_fd = 1e-3
    nabla_f_norms = []
    for j in range(x.shape[1]):
        xp = x.clone(); xp[:, j] += eps_fd
        xm = x.clone(); xm[:, j] -= eps_fd
        with torch.no_grad():
            fp = true_system.closed_loop_dynamics(xp, u_zero)
            fm = true_system.closed_loop_dynamics(xm, u_zero)
        df_dxj = (fp - fm) / (2 * eps_fd)
        nabla_f_norms.append(float(df_dxj.norm(dim=-1).max().item()))
    B_nablaF = max(nabla_f_norms)

    # Estimate L_pi (Lipschitz constant of policy) from finite differences
    u_vals = []
    for j in range(x.shape[1]):
        xp = x.clone(); xp[:, j] += eps_fd
        xm = x.clone(); xm[:, j] -= eps_fd
        with torch.no_grad():
            up = controller.u(xp)
            um = controller.u(xm)
        L_j = float(((up - um) / (2 * eps_fd)).norm(dim=-1).max().item())
        u_vals.append(L_j)
    L_pi = max(u_vals)

    # Empirical lambda_1 from the sweep data as a lower bound
    lambda_1_empirical = CLF_LAMBDA * 0.003   # conservative: use the measured 0.003

    C0 = B_gradV * B_nablaF * (1 + L_pi) + B_hessV * B_f
    C_prime = C0 / lambda_1_empirical

    print(f"\n=== Network constants ===")
    print(f"  ||nabla V||_inf  = {B_gradV:.4f}")
    print(f"  ||nabla^2 V||_inf = {B_hessV:.4f}")
    print(f"  ||f||_inf         = {B_f:.4f}")
    print(f"  ||nabla f||_inf   = {B_nablaF:.4f}")
    print(f"  L_pi              = {L_pi:.4f}")
    print(f"  C_0               = {C0:.4f}")
    print(f"  lambda_1 (used)   = {lambda_1_empirical:.6f}")
    print(f"  C'                = C_0 / lambda_1 = {C_prime:.2f}")
    return C_prime


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sindy_model", default=os.path.join(_REPO, "data", "sindy_model_coefficients.npz"))
    ap.add_argument("--num_rollouts", type=int, default=30)
    ap.add_argument("--t_sim", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading checkpoint: {args.checkpoint}")
    controller, true_system = _load_controller(args.checkpoint, args.sindy_model)

    from Lyapunov_guided_diffusion.sindy.evaluate import evaluate_on_true_system
    print(f"Running {args.num_rollouts} rollouts on TRUE system at sigma=0 ...")
    results = evaluate_on_true_system(
        controller, true_system,
        num_rollouts=args.num_rollouts,
        t_sim=args.t_sim,
        controller_period=0.05,
    )

    dt_ctrl = 0.05
    num_steps = results["V_values"].shape[1]
    t_array = np.linspace(0, args.t_sim, num_steps)

    lambda_bar_mean, lambda_bar_std = fit_lambda_bar(results["V_values"], t_array)
    lambda_margin = lambda_bar_mean - CLF_LAMBDA  # lambda_bar - lambda

    print(f"\n=== Empirical decay rate ===")
    print(f"  lambda_bar (fitted) = {lambda_bar_mean:.6f} +/- {lambda_bar_std:.6f}")
    print(f"  lambda (training)   = {CLF_LAMBDA:.4f}")
    print(f"  lambda_bar - lambda = {lambda_margin:.6f}")
    print(f"  safety_rate         = {results['safe_flags'].mean():.2f}")
    print(f"  limsup V            = {results['V_values'][:, -int(0.2*num_steps):].mean():.6f}")

    C_prime = estimate_C_prime(controller)

    if lambda_margin > 0 and C_prime > 0:
        delta_star_pred = lambda_margin * (SAFE_LEVEL / C_prime) ** 2
        print(f"\n=== Predicted delta* ===")
        print(f"  delta* = (lambda_bar - lambda) * (c / C')^2")
        print(f"         = {lambda_margin:.6f} * ({SAFE_LEVEL:.1f} / {C_prime:.2f})^2")
        print(f"         = {delta_star_pred:.4f}")
        print(f"  Observed delta* from sweep: ~11")
        ratio = delta_star_pred / 11.0
        print(f"  Ratio predicted/observed: {ratio:.3f}  (1.0 = perfect)")
        print()
        if 0.1 < ratio < 10.0:
            print("  Result: within one order of magnitude — the theory is not loose by more than 10x.")
        else:
            print("  Result: more than one order of magnitude off — the bound is loose here.")
    else:
        print(f"\n  lambda_bar <= lambda or C' <= 0: cannot form a prediction.")
        print(f"  This means the strict margin assumption does not hold for this checkpoint.")


if __name__ == "__main__":
    main()
