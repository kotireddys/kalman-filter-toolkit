# First-Estimates Jacobians (FEJ-EKF)

Implementation: `use_fej=True` on [`filters/ekf.py`](../filters/ekf.py)'s
`ExtendedKalmanFilter` — a flag, not a separate class, so that comparing standard
EKF vs. FEJ-EKF is exactly a one-variable ablation. Follows Huang, Mourikis &
Roumeliotis, *Observability-based Rules for Designing Consistent EKF-based SLAM
Estimators* (2010), and Hesch et al., *Camera-IMU-based Localization* (2014) — full
citations in [`references.bib`](references.bib)
(`huang2010observability`, `hesch2014observability`).

## The mechanism

`ExtendedKalmanFilter` maintains a linearization-point store `x_lin`, separate from
the running estimate `x`. Both `F_jac` and `H_jac` are evaluated at `x_lin` instead
of `x` when the flag is on:

- `x_lin` is propagated forward through the *same* dynamics `f` on every
  `predict()` call, so it tracks a "first-estimates trajectory."
- `x_lin` is **never** touched by `update()` — only the running estimate `x` is
  corrected by measurements.
- `augment_state(new_elements, ...)` sets the new block of `x_lin` to
  `new_elements` itself: the FEJ rule that a newly-initialized state's first-ever
  estimate is its initialization value.

## The failure this fixes

In a standard EKF, `F_jac`/`H_jac` are evaluated at whatever the *current* estimate
happens to be. For any state element that is **revisited** — an IMU bias touched by
every `predict()`, a landmark touched by every sighting — the current estimate keeps
moving as updates correct it, so different observations of the same underlying
geometry get linearized around different points at different times.

This time-varying inconsistency makes the observability matrix of the
*linearized* system have a **smaller null space** than the true nonlinear system
has. Concretely: the nonlinear system has an exact symmetry (see below) that the
linearization should respect, but respecting it requires using the *same*
linearization point for the same physical quantity across time. A standard EKF
doesn't, so the filter picks up apparent information along directions that are
genuinely unobservable — most visibly **global position** and **yaw about
gravity** — and becomes overconfident (its reported covariance shrinks along
directions where the true uncertainty cannot shrink).

FEJ removes the inconsistency by fixing one linearization point per state element,
for all time.

## The observability test: deriving the expected null-space dimension

`tests/test_observability.py` sets up an IMU-driven rigid body observing one static
landmark in the body frame ($y = R^\top(p_L - p)$), with **no** absolute position or
heading reference. State (core, excluding biases to keep the derivation clean):
$[\phi, v, p, p_L] \in \mathbb{R}^{12}$.

**Claim: the unobservable subspace is exactly 4-dimensional** (3 translation + 1
yaw about gravity). Derivation: consider replacing the whole trajectory and
landmark by

$$R \to R_{\text{yaw}}R, \quad v \to R_{\text{yaw}}v, \quad p \to R_{\text{yaw}}(p - c) + c, \quad p_L \to R_{\text{yaw}}(p_L - c) + c$$

for an arbitrary fixed point $c \in \mathbb{R}^3$ and an arbitrary fixed rotation
$R_{\text{yaw}}$ about the gravity axis ($R_{\text{yaw}}g = g$, since $g$ is an
eigenvector of any rotation about its own axis). Substituting into the measurement:

$$R_{\text{new}}^\top(p_{L,\text{new}} - p_{\text{new}}) = (R_{\text{yaw}}R)^\top R_{\text{yaw}}(p_L - p) = R^\top(p_L - p),$$

unchanged. Substituting into the dynamics $\dot v = Ra + g$:

$$\dot v_{\text{new}} = R_{\text{yaw}}\dot v = R_{\text{yaw}}(Ra + g) = R_{\text{new}}a + g$$

(using $R_{\text{yaw}}g = g$), also unchanged. So this transformation is an *exact*
symmetry of the whole input-output map. Its generators are 3 translation directions
($c$) plus 1 rotation direction ($R_{\text{yaw}}$ about the fixed axis $g$) — **4**
total. No other direction is free: rotating about any axis other than $g$ changes
$\dot v$'s relationship to gravity and breaks the dynamics.

## Result

Assembling the local observability matrix $O = [H_0;\, H_1\Phi_0;\, H_2\Phi_1\Phi_0;
\ldots]$ over a fixed window and computing its null-space dimension, robust across 5
random seeds:

| Estimator | Null-space dimension found | Correct? |
|---|---|---|
| Standard EKF | 3 | No — manufactures 1 spurious observable direction |
| FEJ-EKF | 4 | Yes |
| InEKF (right-invariant core) | 4 | Yes, but *only* with the correct landmark embedding — see below |

**A hand-derived Jacobian bug, caught before it mattered.** The first version of
the generic EKF/FEJ-EKF model's analytic Jacobians had a genuine bug (missing the
SO(3) right-Jacobian factor in the attitude columns) — off by up to 1.26 in raw
terms, caught by checking against finite differences before trusting the result.
Fixed and reverified to $\sim 10^{-9}$;
see `tests/test_observability.py`.

**InEKF needs more than an invariant core.** A naive extension — reusing the
already-verified invariant core's constant $H_R$ but with a plain Euclidean
landmark error — reproduces the *same* failure as standard EKF (null-space
dimension 3, not 4). This is checked directly, not assumed:
`test_naive_landmark_embedding_reproduces_the_standard_ekf_failure`. The fix is the
actual InEKF-SLAM trick: track the landmark via the invariant tangent coordinate
$\xi_{\text{landmark}} := \delta p_L + [p_{L,\text{ref}}]_\times\,\delta\phi$ for a
fixed reference $p_{L,\text{ref}}$, which makes $H_{\text{joint}} = [0,\,0,\,I,\,-I]$
exactly constant — no residual dependence on the landmark estimate's numeric value
at all.

## Interface note: dynamic state augmentation

FEJ's landmark bookkeeping needs a state that can grow at runtime (a landmark's
first estimate is only known once it's first sighted). `ExtendedKalmanFilter` gained
`augment_state(new_elements, P_init_block, cross_cov=None)` and
`marginalize_state(indices)` for this — a real, backward-compatible interface
addition (existing fixed-dimension use of the class is unaffected), rather than a
parallel API.
