# Paper plan: Lyapunov-guided diffusion with learned dynamics (target: ALT)

Order of work: **Part 1 (theory) first**, then Part 2 (experiments).
Paper version this plan is based on: `paperv1.tex`.
Reviewer issue tags (T1–T12) refer to the review given on 2026-10-10.

Status keys: `[ ]` not started, `[~]` in progress, `[x]` done, `[?]` needs checking / might not work.

---

## 0. The story we want ALT reviewers to see

ALT reviewers reward three things: (1) correct proofs, (2) tight rates (an upper bound
**and** a matching lower bound), (3) a clear "how much data do you need" answer.
The current paper has none of the three cleanly. The new story:

> We learn the dynamics from N noisy transitions with SINDy, build a certificate and a
> diffusion controller on the learned model, and prove:
> 1. **Where you end up**: the true system ends up inside a sublevel set whose height we can
>    compute from the learned model and its error bars (no unknown sets, no volume tricks).
> 2. **How fast it shrinks**: that height is of order delta^2 for a quadratic certificate
>    (delta = model error), and this is **tight** (we show an example where you cannot do better).
> 3. **How much data**: with N transitions the final level is of order sigma^2 (S + log 1/alpha) / N,
>    and we give a matching lower bound.
> 4. **Where the data must be**: staying safe forever only needs the model to be accurate **on the
>    edge of the safe set**, not everywhere. This explains the boundary-aware experiment with a theorem.

What is **not** new (say so honestly in the paper, cite it): the cushion ||grad V|| * r (robust CLF/CBF
literature), the comparison lemma, ultimate-boundedness arguments (Khalil, Ch. 4/9), conformal prediction,
self-normalized least-squares bounds (Abbasi-Yadkori et al. 2011).
What **is** new: putting them together for diffusion-guided control with sparse ID, the data-dependent
(coverage-dependent) guarantee, the tight delta^2 / 1/N rates, and the lower bounds.

---

## Part 1 — Theory work (do first)

### Status after the v2 theory pass (2026-10-10) — file `paperv2.tex`

Done in `paperv2.tex` (v1 left untouched):
- [x] TH1 height theorem (Thm `thm:main`), sampled-data, slack s, deterministic, safety + ultimate bound; Lemma `lem:ultimate` added
- [x] TH2 margin theorem (Thm `thm:margin`) with general exponent q; q=2 gives O(delta^2); delta_star formula
- [x] TH3 tightness (Prop `prop:tight`): estimate attained with equality; delta^2 scaling unimprovable for any (f_hat, r, V)-based bound
- [x] TH4 uniform envelope (Prop `prop:envelope`) via Abbasi-Yadkori self-normalized bound; bias/misspec floor; STLSQ handled via split or r + ||f_S - f_ridge||
- [x] TH5 sample complexity upper bound (Cor `cor:sample`) and 1-D minimax lower bound for Lipschitz policies (Prop `prop:lower`)
- [x] TH6 coverage/edge corollary (Cor `cor:edge`)
- [x] TH7 slack sampler assumption, fixed Gibbs concentration (uniform p_star(s), 3s^2/4 exponent), finite-horizon corollary (`cor:finite-horizon`)
- [x] TH8 Thm `thm:prob` hypothesis (b) fixed, Lipschitz notation fixed, infinite-horizon via filter
- [x] TH9 corrected volume theorem (Thm `thm:volume`, rate lambda, explicit C) + Lipschitz-necessity counterexample (Prop `prop:lipschitz-needed`)
- [x] TH10 housekeeping: o-minimal R_an,exp, h_f defined, invariance claim removed, abstract leftover removed, "reverse reduction" promise removed, perimeter/anti-concentration/Steiner/Robbins-Siegmund/Sard removed, conformal demoted to a remark
- [x] Numerical sanity checks passed: tightness examples, Thm A on a sampled-data 1-D system, uniform envelope coverage (100% vs 90% target), lower-bound reduction
- [x] Experiments section: theory claims aligned; red \todo markers left where results must be recomputed

Bib entries to ADD to references.bib (new keys used in v2):
- `abbasi2011improved` — Abbasi-Yadkori, Pal, Szepesvari. Improved algorithms for linear stochastic bandits. NeurIPS 2011.
- `khalil2002nonlinear` — H. K. Khalil. Nonlinear Systems, 3rd ed. Prentice Hall, 2002.
(`van1998tame`, `castaneda2021pointwise`, `choi2021robust`, `taylor2020learning` already exist.)

### Remaining theory items found in the post-v2 ALT review
- [ ] TH12 Sampled-data constants are likely vacuous for the pendulum (B ~ 10, h = 0.05 gives B^2 h ~ 5 > c). Replace global B^2 by local sup over K_c x U of ||grad V|| ||f||, or add a discrete-time version (one-step decrease V(x_{k+1}) <= (1 - lambda h) V(x_k) checked on the model with an envelope on the one-step map).
- [ ] TH13 Lower bound is 1-D, single parameter. Generalize with Assouad/Fano to p unknown coefficients to match the n p / N dependence of Cor `cor:sample`.
- [ ] TH14 Persistent excitation kappa: give a sufficient condition for closed-loop data with exploration noise (e.g. a small-ball / block-martingale argument as in Simchowitz et al. 2018) so kappa is not just assumed.
- [ ] TH15 Errors-in-variables (noise on measured states) — at least a perturbation bound adding O(sigma_x L_phi) to the envelope, since the experiments use state noise.
- [ ] TH16 p_star(s) scales like (s / L_u)^m, exponentially bad in control dimension m. Either state as a limitation or give a sharper bound under log-concave nu.
- [ ] TH17 Strict margin lambda_bar is uncheckable; consider an a-posteriori certified version (lambda_bar on {V > tau}) from (f_hat, r) only, which is exactly tau_hat — say so explicitly or drop Thm `thm:margin` to a "rate" proposition.

---

### Original plan (kept for reference)

Ordered by priority. Each item: what, why, draft statement, proof idea, risk.

### TH1. New main theorem: the "height" bound (replaces the volume bound as headline)  — fixes T1, T2, T3, T9
- [ ] Write statement + proof
- **Idea in simple words.** Instead of measuring the danger zone by its *volume*, measure it by its
  *height*: the largest value of V inside the danger zone,
  `tau_hat := sup { V(x) : x in Omega_rob }` (and 0 if Omega_rob is empty).
  Above that height there is no danger zone, so V must go down there. So V falls until it is below
  tau_hat and can never climb back above it.
- **Draft statement.** Let the envelope be valid and let the policy satisfy the sampler assumption on
  `{V > tau_hat}`. Then for every start x_0,
  `V(x_t) <= max( e^{-lambda t} V(x_0), tau_hat )` for all t >= 0,
  so `limsup V(x_t) <= tau_hat`. If also `tau_hat < c` and `V(x_0) <= c`, the safe set `{V <= c}` is
  never left (forward invariance).
- **Proof idea (half a page).** On `{V >= tau_hat}`: x is outside Omega_rob (check the boundary case
  V = tau_hat carefully, see TH2), so the transfer lemma gives `dV/dt <= -lambda V`. Comparison lemma
  while above tau_hat; once at tau_hat the derivative is negative so it cannot cross upward.
- **Why it is better.** Deterministic (no martingale), no Steiner/perimeter, no anti-concentration
  assumption, no Lipschitz policy needed, no dimension n in the exponent, rate is lambda (not lambda_1),
  and tau_hat is **computable** from (f_hat, r, V) on a grid. It also gives safety, which the old theorem did not.
- **Novelty level: low on its own** (it is a standard ultimate-boundedness argument). It is the clean
  foundation that TH2–TH5 build on. Do not sell it as the main novelty.
- **Technicality to handle:** if the policy is discontinuous (e.g. the safety filter), solutions of the
  ODE need care. Use the **sampled-data** setting (control held constant for one controller period
  `Delta t_c`, which is how the code actually works). This adds a small term `O(Delta t_c)` to the bound.
  Write this as Lemma TH1.b.
- Risk: low.

### TH2. Sharp rate under a strict margin: delta^2, not delta^(1/2)  — fixes T4, T6
- [ ] Write statement + proof
- [ ] Double-check the exponent with a 1-D numerical example before writing
- **Idea.** Near the goal a smooth certificate is flat: its gradient is small. For a quadratic-like V,
  `||grad V(x)|| <= beta * sqrt(V(x))`. The model-error term is `||grad V|| * delta <= beta sqrt(V) delta`,
  while the pull toward the goal is `(lambdabar - lambda) V`. The pull wins as soon as
  `sqrt(V) > 2 beta delta / (lambdabar - lambda)`.
- **Draft statement.** Assume V >= alpha ||x - x*||^p near the goal and
  `||grad V|| <= beta V^{(p-1)/p}` (true for p = 2 when V is smooth with a minimum at the goal), and the
  strict margin. Then `tau_hat <= ( 2 beta delta / (lambdabar - lambda) )^p`, so
  `limsup V <= ( 2 beta delta / (lambdabar - lambda) )^p`, which is **O(delta^2)** for p = 2.
- **Weaker, simpler version** (keep as a remark): using only `||grad V|| <= B` gives `tau_hat <= 2 B delta / (lambdabar - lambda)` = O(delta).
- **Safety threshold (new, computable):** safe forever if `tau_hat < c`, i.e. if
  `delta < delta_star := (lambdabar - lambda) c^{1/p} / (2 beta)`. Replaces the vague "empirical calibration" paragraph.
- **Removing the circularity (T6).** lambdabar depends on the trained V. Make it computable from data:
  since `L_f V <= L_fhat V + ||grad V|| r`, a *certified* margin on any region can be computed from
  (f_hat, r, V). State the margin as "the trained V satisfies ..." and note it can be checked on a grid.
- Risk: low. (Sanity check: scalar system `xdot = -x + d`, `|d| <= delta`, V = x^2 gives limsup V = delta^2 exactly.)

### TH3. Lower bound: delta^2 cannot be improved  — new, ALT-style
- [ ] Construct the example and prove it
- **Idea.** Show a system and a learned model with error delta where **any** certificate-guided policy
  that only trusts the model ends with `limsup V >= c0 * delta^2`.
- **Candidate construction.** Scalar `xdot = theta + u`, model `xdot = theta_hat + u` with
  `|theta - theta_hat| = delta`, V = x^2, policy enforces `L_fhat V <= -lambda V` on the model.
  The true closed loop then has an equilibrium at distance ~ delta / lambda, so V ~ delta^2 / lambda^2.
- **Care needed:** a very aggressive controller (huge gain) could beat this. Restrict the policy class
  to "achieves exactly rate lambda on the model" or bound the control effort (U compact). State the
  class clearly; the lower bound is only for that class.
- Risk: medium (the policy class must be natural, not artificial).

### TH4. Uniform, data-dependent error envelope r(x,u)  — fixes T7; key ALT piece
- [ ] Write statement + proof
- **Problem today.** Theory needs `||f - f_hat|| <= r(x,u)` at **every** (x,u). Conformal only gives
  "true for one random point with prob 1 - alpha". The finite-sample prop is missing a sqrt(n) and needs support recovery.
- **Fix (elegant).** Because `f - f_hat = (Xi - Xi_hat)^T Theta(x,u)` is *linear in the features*, one
  confidence ellipsoid for the coefficients gives an envelope that holds **at all (x,u) at once**:
  `r(x,u) = beta_N(alpha) * || Theta(x,u) ||_{(Theta^T Theta + rho I)^{-1}}`
  with `beta_N(alpha)` from the self-normalized bound (Abbasi-Yadkori, Pal, Szepesvari 2011).
  This works for data collected along trajectories (not i.i.d.), which is our case.
- **Big bonus:** this r is **small where the data are** and **large where there are no data**. That is
  exactly the "where you collect data matters" effect, now as math. Feeds TH5 and TH6.
- **Sparsity:** first prove it for least squares (or ridge) on the **full library** (no support-recovery
  assumption). Then a remark: if the support is recovered, replace p (library size) by S (active terms).
- **Noise model:** state it as noise on the derivative targets (sub-Gaussian, martingale-difference).
  Noise on the states (errors-in-variables) goes into Limitations, or add a bias term.
- **Truncation and misspecification:** add a deterministic term `r_bias = O(Delta t) + misspec` to r.
  This makes the "three sources of delta" paragraph a formula.
- Keep conformal as a remark only (it gives a cheaper but only per-point guarantee).
- Risk: low–medium (standard tools, careful constants).

### TH5. Sample complexity: how much data for a guarantee (upper + lower bound)  — new, ALT headline
- [ ] Upper bound (combine TH2 + TH4)
- [?] Lower bound (harder)
- **Upper bound.** With TH4, `delta <= Theta_max * beta_N` and `beta_N ~ sigma sqrt((p + log 1/alpha)/N)`.
  Plug into TH2 (p = 2): with prob 1 - alpha,
  `limsup V <= C * sigma^2 (p + log 1/alpha) / (N (lambdabar - lambda)^2)`, i.e. **O(1/N)**.
  And safety forever once `N >= N_star := C' sigma^2 (p + log 1/alpha) / ((lambdabar - lambda)^2 c)`.
  That is a clean "sample complexity of certified safety".
- **Lower bound.** Two-system argument (Le Cam): two scalar systems whose data look almost the same
  after N samples but need different controls; any data-driven certificate-guided policy ends with
  `limsup V >= c sigma^2 / N` on one of them. Builds on TH3.
- Risk: upper bound low; lower bound medium–high. If the lower bound does not work out, keep the upper
  bound and the delta^2 tightness from TH3.

### TH6. Coverage theorem: safety needs accuracy on the edge of the safe set  — new; explains the boundary experiment
- [ ] Write statement + proof
- **Idea.** To never leave `{V <= c}` you only need, on the edge `{V = c}`, a control that makes V go
  down for the true system (Nagumo-type invariance). With the robust residual, this only needs
  `d_rob(x,u) <= 0` for x **on the edge**, i.e. r must be small **on the edge**.
- **Draft statement.** If `inf_u d_rob(x,u) < 0` for every x with V(x) = c, and the policy picks such a
  control there, then `{V <= c}` is forward invariant for the true system. With the TH4 envelope, a
  sufficient condition is a coverage condition: the data matrix "sees" the feature directions active on
  the edge, i.e. `sup_{V(x)=c} ||Theta(x,u)||_{(Theta^T Theta)^{-1}}` is small.
- **Why this matters:** it says interior data cannot buy safety; data near the edge can. That is the
  narrow-vs-wide experiment, now predicted by theory instead of explained after the fact.
- **Be careful:** the experiment's safe set is the state box ||x|| < 1.5, not exactly `{V <= c}`.
  Either define the theorem with a general safe set (barrier function h) or make the code use `{V <= c}`.
- Risk: low–medium.

### TH7. The diffusion sampler: approximate decrease with slack  — fixes T8, T3
- [ ] Rewrite Assumption 2 as an approximate version
- [ ] Fix Prop. 17 (concentration) and carry slack into TH1
- **Approximate sampler assumption.** Outside the danger zone, the sampled control has
  `d_rob(x,u) <= s` (slack s >= 0) — with probability >= 1 - eta per step.
- **Effect on TH1.** `dV/dt <= -lambda V + s`, so the height bound becomes `max(tau_hat, s/lambda)` (+ sampled-data term).
- **Prop. 17 fix.** Keep the tail bound `P[d_rob > s] <= exp(-s^2/gamma_2)/p_star`; drop the illegal
  "s -> 0" limit. Make p_star uniform over x (assume nu gives every certified control set at least p_star mass).
- **Per-step failure → whole trajectory:** union bound over K controller steps, or a stopping-time
  argument; gives "with prob >= 1 - K eta". Pairs naturally with TH8.
- **T3 decision (Lipschitz policy):** with TH1 as the main theorem, **no Lipschitz policy is needed**,
  so the conflict with the safety filter disappears for the headline result. For the old volume theorem
  (TH9), use option (a): a smoothed filter with a proven Lipschitz constant. Option (c) (no Lipschitz at
  all) is **false in general**: a discontinuous policy can create a fake resting point inside a tiny danger
  zone far from the goal. Write this as a short counterexample — "Lipschitz is necessary for any
  volume-only bound". That counterexample is a small but real contribution.
- Risk: low.

### TH8. Probabilistic safety theorem (Thm 16) cleanup + infinite horizon  — fixes T10
- [ ] Hypothesis (b): require P_2 <= eta on the whole ball around the nominal control
- [ ] Use separate names for the Lipschitz constants of f and f_hat
- [ ] Infinite horizon: use TH1/TH6 forward invariance (the paper currently lists this as open — we can close it)
- [ ] Remove the "a > eta" informal remark or make it a stated condition
- Risk: low.

### TH9. Keep the volume bound as a secondary result (corrected)  — fixes T1, T2, T5
- [ ] Rewrite proof with the "no big ball in a small set" argument only (delete Steiner, perimeter, anti-concentration, martingale)
- [ ] Add the interior-cone condition on the state space so the ball argument works near the edge of X
- [ ] Needs a Lipschitz policy (smoothed filter from TH7)
- [ ] State exactly what C depends on: B, L_pi, lambda, n, cone constant. Note B and L_pi come from the trained V and policy.
- **When it is better than TH1:** when a small piece of the danger zone sits high up (large V) far
  from the goal; height bound is then bad but volume bound is still small. Present TH1 and TH9 as complementary.
- Risk: low.

### TH10. Housekeeping (small but reviewers will notice)
- [ ] Prop. 11(iii): replace "semialgebraic" by "definable in R_{an,exp}" (covers sin on compact sets and tanh networks)
- [ ] Define h_f and the goal set G
- [ ] Remove the claim that Prop. 11 gives "invariance"; stop relying on Cheng's connected/invariant hypotheses
- [ ] Fix the sqrt(n) factor in the finite-sample proposition (or replace it entirely with TH4)
- [ ] Define sigma (currently the noise is called eta)
- [ ] Remove "with B bounding ||grad V||" from the abstract; drop "almost surely" where TH1 is deterministic
- [ ] Delete the promise of a "reverse reduction" in Appendix A, or write it
- [ ] Theorem 1 currently claims convergence only; make sure safety is stated with its own condition (TH1/TH6)
- [ ] Add citations for robust CLF/CBF cushions (Jankovic 2018; Taylor et al. 2020; Castaneda et al. 2021; Kolathaya & Ames ISSf 2019)

### TH11. Positioning text (after the math is done)
- [ ] Rewrite contributions: TH2+TH3 (tight rate), TH5 (sample complexity), TH6 (coverage), TH1 as foundation
- [ ] Related work paragraph: say plainly the cushion is standard; our novelty is the data-dependent analysis
- [ ] Limitations: errors-in-variables, policy class in lower bound, sampled-data term, one system (until Part 2)

### Suggested order
1. TH1 + TH2 (one week-ish; unlocks everything) → sanity-check numerically with a 1-D example
2. TH4 (envelope) → TH5 upper bound
3. TH6 (coverage)
4. TH7 + TH8 (sampler, probability)
5. TH3 + TH5 lower bound (riskiest; do last, drop if stuck)
6. TH9, TH10, TH11

---

## Part 2 — Experiments (parked until Part 1 is done)

Short list; details later. Each one should test a theorem from Part 1.

- [ ] **E1. Measure the new bound.** Compute tau_hat (height of the danger zone) on a grid for every
      sweep run, with r from TH4. Plot measured limsup V against tau_hat. (Tests TH1.)
- [ ] **E2. Rate check.** Log-log of limsup V vs delta; theory now says slope ~2 (TH2). Current fit is ~1.3.
- [ ] **E3. Predicted safety threshold.** Compute delta_star from TH2 using measured lambdabar, beta;
      compare to the measured cliff (~11). (Tests TH2.)
- [ ] **E4. Sample-size sweep.** Vary N at fixed small noise; expect limsup V ~ 1/N (TH5).
- [ ] **E5. Coverage.** Compute the edge-coverage quantity from TH6 for narrow vs wide data; it should
      predict which arm stays safe. (Tests TH6.)
- [ ] **E6. Implement the safety filter** in the controller (sampled-data, TH7) and rerun the cliff band.
- [ ] **E7. Cart-pole (n = 4)** — the deleted TODO. Same pipeline. Cart-pole has a denominator
      (M + m sin^2 theta), so the library is misspecified by default → also covers E8.
- [ ] **E8. Misspecified library.** Pendulum without sin(theta); vary N with no noise; delta should
      hit a floor that more data cannot remove.
- [ ] **E9. Time-step sweep.** Vary Delta t to show the O(Delta t) floor.
- [ ] **E10. Data fixes.** Use `unified_narrow.csv` + `unified_wide.csv` consistently (6% → 0%, 0.67 → 0.64);
      add the sigma = 0.15 row to Table 1; check `bad_set_fraction` (~0.7 at all noise levels) against Figure 2.
- [ ] **E11. Report the negative ablation**: volume-minimizing SINDy hurt safety.
- [ ] **E12. Fill** compute hours/hardware and the anonymous code link.

### Status after the theory-check runs (2026-10-11)

Done with the fixed quadratic certificate (Section "Direct checks of the theory" in `paperv2.tex`):
E1, E2 (slope ~3 locally, consistent with O(delta^2)), E4 (tau_hat ~ N^-1.08), E6 (filter), E8 (floor 0.75 vs 0.13).
Not done: E3 for neural certificates (delta_star undefined: lambdabar < lambda), E5 with the envelope, E7, E9.

---

## Part 3 — A learned certificate that the theorems actually cover

### Why the neural networks failed (diagnosis, 2026-10-11)

The theorems need, on the safe set K_c, that from every state some allowed input pushes V down fast:
lambdabar = inf_x ( - min_{|u|<=30} gradV(x).f(x,u) ) / V(x) > lambda = 0.5.
Every trained network failed this (lambdabar between -0.15 and +0.02). Three causes, all in how the
original code trains V, none in the theory:

1. **V is not zero at the goal.** The network gives V(goal) ~ 0.75 and V ~ 0.8 at the edge of K_c.
   Near the goal V cannot shrink at rate lambda*V because it is stuck at 0.75.
2. **Training never asks for the condition we check.** The original loss asks for decrease along a
   few simulated diffusion rollouts on the learned model, plus "V <= 1 safe, V >= 1 unsafe". It never
   asks for "some input with |u| <= 30 makes V go down at rate lambda", and never uses the envelope r_N.
3. **Unbounded inputs in training.** The controller's limits are +-1000, so training can rely on huge
   inputs. Shifting V to 0 at the goal restores the condition only with |u| up to ~300, and huge
   held inputs then break sampled-data safety.

A hand-made quadratic V passes easily (lambdabar = 2), so a valid certificate exists for this system.
The fix is to train the network on the theorem's own condition.

### The new approach in one sentence

Train a small network certificate directly on the robust decrease condition the theorem checks,
d_rob(x,u) = gradV.f_hat + lambda V + |gradV| r_N(x,u) <= 0 for some |u| <= 30, using only the learned
model f_hat and its data envelope r_N, with V = 0 at the goal by construction.

### Steps

- [ ] **C1. Certificate architecture (new file `sindy/robust_certificate.py`).**
      V(x) = 1/2 |phi(x) - phi(0)|^2 + eps * 1/2 x^T P x, with phi a small tanh MLP on (sin theta, cos theta, theta_dot)
      and P the LQR matrix of f_hat. Then V(0) = 0 exactly and V > 0 elsewhere by construction (fixes cause 1).
- [ ] **C2. Training loss (fixes causes 2 and 3).** Sample states in the box, plus extra samples near the
      unsafe ring. With the safe level fixed at c = 1:
      - decrease: relu( min_{|u|<=30} d_rob(x,u) + margin ) on states with V(x) <= c
        (closed form: d_rob is affine in u plus the convex term |gradV| r_N(x,u), so minimize on a 1-D grid);
      - level: relu(c - V(x)) on ||x|| >= 1.5, so K_c stays inside the safe set;
      - size: relu(V(x) - 0.5c) on the safe ball ||x|| <= 0.5, so K_c is not trivially small.
      Plain torch + Adam on CPU, no QP or diffusion in the loop, so minutes rather than hours.
- [ ] **C3. Two training arms, same data.** (a) nominal: r_N = 0 (trusts f_hat, like the original code);
      (b) robust: the r_N of Prop. 5. Prediction from the theory: (b) gives a smaller tau_hat and stays
      safe at larger noise. This is a new experiment that uses the envelope as a training signal.
- [ ] **C4. Check with the existing tools.** Add `--cert learned --cert_file ...` to `exp_theory.py`
      and rerun stages 2 and 3 (noise and sample-size sweeps). Success criteria:
      lambdabar > 0.5 with |u| <= 30 on K_c; tau_hat <= c/2 for sigma <= 0.5 at N = 5000;
      K_c at least as large as the quadratic one; all rollouts safe when tau_hat <= c/2.
- [ ] **C5. Diffusion sampler.** Plug the learned V into the existing controller (wrapper on
      `V_with_jacobian`, the same trick as `CertView`) so the diffusion sampler is guided by it.
      Run `diffusion` and `diffusion_filter` on a small grid (sigma in {0, 0.5, 1}, 3 seeds, 10 rollouts),
      since one diffusion job takes 10+ minutes on CPU. Overnight run.
- [ ] **C6. Paper.** Replace the paragraph "Learned certificates do not meet the margin" with the results
      (keep the diagnosis of the original training as one sentence). Describe the loss in the existing
      appendix "Certificate training objective" (`app:cert`). If C3(b) beats C3(a), add it as a short
      result: training against the envelope shrinks the violation set.

### Decision points and fallbacks

- If the robust arm cannot be trained (r_N too large, the condition infeasible on most of the box):
  train on the larger-data setting first (N = 20000, where r_N is ~2x smaller), and report the
  data size at which a certificate becomes trainable. That is itself a sample-complexity result.
- If C1 trains but lambdabar stays below lambda: lower lambda to 0.2 (the theorem holds for any
  lambda < lambdabar) and say so.
- If everything fails: keep the current paper text (honest negative result) and move on to E7.
- Order: C1 -> C2 -> C3 on one seed (quick check) -> C4 -> C5 overnight -> C6.

### Status (2026-10-11): C1-C6 done, no fallback needed
- C1-C3 worked on the first try (`sindy/robust_certificate.py`, 15 s per certificate on CPU).
- C4: fixed learned certificate, lambdabar 1.86-2.24 > 0.5; tau_hat <= 0.017c for sigma <= 0.5;
  all 79 configurations with tau_hat <= c/2 fully safe; data threshold between N = 2000 and 5000.
- C3: nominal arm tau_hat/c >= 0.98 at every sigma; robust arm 0.003-0.02 for sigma <= 1.
- C5: diffusion sampler needs its control set matched to U (`--match_u_limits`, stage 12); with the
  default +-1000 sampling range it stays safe but does not settle (long-run V ~ 0.3-0.8 c).
- C6: `paperv2.tex` updated (tab:learned, robust vs nominal paragraph, appendix loss eq:robust-cert-loss).
- Runs: `data/theory_learned`, report `figures/theory_learned/summary.txt`.
