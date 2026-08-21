# Invariant EKF on SE₂(3)

Implementation: [`filters/inekf.py`](../filters/inekf.py), built on the SE₂(3) Lie
group utilities in [`filters/lie/`](../filters/lie/). Following Barrau & Bonnabel,
*The Invariant Extended Kalman Filter as a Stable Observer* (2017), and the
"imperfect InEKF" treatment in Hartley et al.'s contact-aided work (2020) — full
citations in [`references.bib`](references.bib)
(`barrau2017invariant`, `hartley2020contact`).

## State

The core navigation state is a matrix Lie group element

$$X = \begin{pmatrix} R & v & p \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} \in SE_2(3),$$

with $R \in SO(3)$ the attitude, $v \in \mathbb{R}^3$ the world-frame velocity, and
$p \in \mathbb{R}^3$ the world-frame position. IMU biases $(b_g, b_a)$ are appended
as a plain Euclidean component, not part of the group.

## The group-affine condition

The bias-free IMU dynamics

$$\dot R = R\,[\omega]_\times, \qquad \dot v = Ra + g, \qquad \dot p = v$$

is *group affine*: writing it as a vector field $f_u(X)$ (in code,
`_continuous_dynamics_hat`), it satisfies

$$f_u(X_1 X_2) = f_u(X_1)\,X_2 + X_1\,f_u(X_2) - X_1\,f_u(\mathrm{Id})\,X_2 \tag{$\ast$}$$

for all $X_1, X_2 \in SE_2(3)$ and any fixed input $u = (\omega, a)$.

**A correction to the commonly stated form.** $(\ast)$ requires the trailing $X_2$
on the last term. Some informal statements of this identity omit it
(i.e. claim $f_u(X_1X_2) = f_u(X_1)X_2 + X_1f_u(X_2) - X_1f_u(\mathrm{Id})$). We checked
both forms numerically for this exact dynamics: $(\ast)$ holds to machine precision
(residual $\sim 10^{-14}$), while the version without the trailing $X_2$ fails by
$O(1)$ (residual $\sim 2.6$ in our test). See
[`tests/test_inekf.py::test_group_affine_property`](../tests/test_inekf.py).

### Why group-affineness matters

Let $\eta = \hat X X^{-1}$ be the **right-invariant** error between an estimate
$\hat X$ and the truth $X$, both driven by the same (noise-free, bias-free) inputs.
Differentiating and applying $(\ast)$ with $X_1 = \eta$, $X_2 = X$:

$$\dot\eta = f_u(\hat X)X^{-1} - \eta f_u(X)X^{-1} = f_u(\eta) - \eta f_u(\mathrm{Id}),$$

a **closed, autonomous ODE in $\eta$ alone** — it does not depend on $\hat X$ or $X$
individually, only on their ratio. Linearizing at $\eta = \mathrm{Id}$ ($\eta =
\exp(\xi)$, $\xi$ small) gives $\dot\xi = A_R\xi$ with

$$A_R = \begin{pmatrix} 0 & 0 & 0 \\ [g]_\times & 0 & 0 \\ 0 & I & 0 \end{pmatrix}
\quad \text{(blocks ordered } [\phi, v, p] \text{)},$$

independent of the inputs $(\omega, a)$ and of the trajectory. The analogous
left-invariant computation ($\eta_L = X^{-1}\hat X$) gives

$$A_L = \begin{pmatrix} -[\omega]_\times & 0 & 0 \\ -[a]_\times & -[\omega]_\times & 0
\\ 0 & I & -[\omega]_\times \end{pmatrix},$$

which depends on the known inputs but still not on the state estimate.

**This is the property that distinguishes InEKF from a repackaged ESKF**: for a
standard EKF/ESKF, the linearized error dynamics matrix depends on the current
state estimate, so it is only valid for *small* errors. For InEKF, $\dot\xi = A\xi$
is *exact* — it holds for arbitrarily large $\xi$, not just infinitesimally, because
the underlying nonlinear ODE for $\eta$ really is linear once written in $\xi$.
We verify this directly:
[`tests/test_inekf.py::test_log_linearity_bias_free_large_errors`](../tests/test_inekf.py)
propagates a true state and an estimate from a *large* initial error (up to ~70° /
4m) under identical inputs, and checks that $\xi(\Delta t) = \Phi(\Delta t)\,\xi(0)$
with $\Phi = \exp(A\Delta t)$ to $10^{-6}$ — not just to first order.

Both $A_R$ and $A_L$ above were **derived numerically** (finite-differencing the
true nonlinear error-propagation map, not assumed from a paper) and then confirmed
to have exactly this closed form; see
`tests/test_inekf.py::test_error_dynamics_matrix_independent_of_state_estimate`.

## Selecting the error convention

The rule (derived per measurement model below, not asserted): the right convention
is whichever makes the measurement's linearized innovation state-independent.

**Right-invariant — body-frame observation of a world-fixed point.** A known
landmark at $p_L$, observed as $y = R^\top(p_L - p) + \text{noise}$. Lift to
homogeneous coordinates $Y = [y;0;1]$, $b = [p_L;0;1]$; since $Y = X^{-1}b$
(noise-free), the pseudo-innovation

$$Z := \hat X Y - b = (\eta - I)b + \hat X[\text{noise};0;0] \approx \underbrace{[-[p_L]_\times \;\; 0 \;\; I]}_{H_R}\,\xi + \hat R\,\text{noise}$$

is linear in $\xi$ with $H_R$ depending only on the *known* landmark position — not
on the state estimate at all.

**Left-invariant — world-frame observation of a body-fixed point (GNSS).** A direct
position fix $z = p + \text{noise}$. Using $Y = [z;0;1]$ and the group's own fixed
selection vector $b = e_5 = [0,0,0,0,1]$ (so $Y = Xb$ noise-free):

$$Z := e_5 - \hat X^{-1}Y \approx \underbrace{[0 \;\; 0 \;\; I]}_{H_L}\,\xi_L - \hat R^\top\text{noise},$$

again exactly constant.

Both $H_R$ and $H_L$ were verified numerically (not just derived): perturbing $\xi$
at several different, large-and-small scales and several different reference
states, the residual $|Z - H\xi|$ shrinks as $O(|\xi|^2)$ regardless of the
reference state — confirming $H$ is the correct, state-independent linearization
(`tests/test_inekf.py::test_landmark_measurement_jacobian_is_constant_and_correct`,
`::test_gnss_measurement_jacobian_is_constant_and_correct`).

**Injection sign.** The Kalman correction $\delta = K \cdot Z$ is an estimate of the
*current* error $\xi = \log(\eta_{\text{old}})$, so removing it requires the inverse
direction: $\hat X_{\text{new}} = \exp(-\delta_{\text{core}})\,\hat X_{\text{old}}$
(right-invariant, left-multiply) or $\hat X_{\text{old}}\exp(-\delta_{\text{core}})$
(left-invariant, right-multiply). This was checked end-to-end, not assumed: applying
the "+" sign measurably *increases* true estimation error in a synthetic test, while
"−" decreases it (`tests/test_inekf.py::test_update_reduces_error_*`).

## The "imperfect" InEKF: biases break exact group-affineness

Once biases are estimated, the true and estimated trajectories use *different*
effective inputs ($\omega_m - b_g$ vs. $\omega_m - \hat b_g$), and the clean
cancellation behind $\dot\eta = f(\eta) - \eta f(\mathrm{Id})$ no longer holds
exactly. The correctly-linearized (and numerically re-verified against direct
finite-differencing of the true nonlinear propagation, to $O(\Delta t^2)$ — the
usual frozen-Jacobian discretization error, not a modeling error) 15-dimensional
matrices are:

**Right-invariant**, blocks ordered $[\phi, v, p, b_g, b_a]$:

$$A_R = \begin{pmatrix}
0 & 0 & 0 & -\hat R & 0 \\
{[g]_\times} & 0 & 0 & -[\hat v]_\times \hat R & -\hat R \\
0 & I & 0 & -[\hat p]_\times \hat R & 0 \\
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0
\end{pmatrix}$$

**Left-invariant**:

$$A_L = \begin{pmatrix}
-[\hat\omega]_\times & 0 & 0 & -I & 0 \\
-[\hat a]_\times & -[\hat\omega]_\times & 0 & 0 & -I \\
0 & I & -[\hat\omega]_\times & 0 & 0 \\
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0
\end{pmatrix}$$

**A genuine, reported (not hidden) finding**: the right-invariant bias coupling
depends on the *full* current state estimate $(\hat R, \hat v, \hat p)$, while the
left-invariant one depends only on the known inputs and the current bias estimate.
Quantified directly — across several random state references, the spread in $A$
(right-invariant) is roughly **38× larger** than the spread in $A$ (left-invariant)
— see `tests/test_inekf.py::test_log_linearity_degrades_with_bias`, which prints
the actual numbers. In other words: **once biases are estimated, the left-invariant
convention stays much closer to the ideal state-independent linearization than the
right-invariant one.** This is a genuine asymmetry between the two conventions that
isn't obvious from the bias-free theory alone, and the benchmark (GNSS position
updates, left-invariant) is set up to benefit from it.

My first attempt at the right-invariant bias-coupling block was wrong (missing the
$[\hat v]_\times\hat R$ and $[\hat p]_\times\hat R$ cross-terms) — caught by checking
the residual against finite differences and its scaling with $\Delta t$ (it didn't
shrink like $\Delta t^2$ until the missing terms were added). See git history /
`tests/test_inekf.py::test_bias_error_dynamics_matches_nonlinear_propagation`.

**A second, more serious bug** was in `update()`'s bias injection itself: it used
`self.b_g = self.b_g + delta[9:12]` — the wrong sign, inconsistent with the core
state's `exp(-delta[:9])` correction. Since `delta` estimates the *current* error
($\hat b - b_{\text{true}}$), removing it requires subtracting, not adding; the "+"
turned every bias correction into positive feedback. This unit test suite entirely
missed it (every existing bias test used short runs or large artificial P0 where the
effect wasn't visible) — it was only caught by the Monte Carlo benchmark harness,
where ANEES exploded to $\sim 10^4$ (dof 15) on the 20s aggressive trajectory and
the gyro bias estimate was found to drift monotonically to 15× its true magnitude
instead of converging. Fixed, and now covered directly by
`tests/test_inekf.py::test_bias_estimate_converges_not_diverges_over_many_updates`.
The lesson generalizes: a short unit test with a single update or an oversized
initial covariance can hide a sign error that only compounds into something visible
over many correction cycles — exactly the kind of thing a longer-horizon consistency
benchmark is for.

## Exact closed-form exp, log, Adjoint

`filters/lie/so3.py` and `filters/lie/se23.py` provide closed-form
$\exp$/$\log$/Adjoint for $SO(3)$ and $SE_2(3)$, with Taylor-series fallbacks below
$\theta = 10^{-4}$ rad (and a best-effort, not fully hardened, fallback near
$\theta = \pi$, documented as such — not expected to matter for IMU-rate
integration). All three properties are tested directly, not assumed:

- exp/log round-trip for **large** rotations (up to 0.95π) and translations, not
  just infinitesimal ones.
- the Adjoint identity $X\exp(\xi)X^{-1} = \exp(\mathrm{Ad}_X\xi)$, checked by direct
  matrix conjugation against the closed-form Adjoint — not assumed correct just
  because it matches a formula from a paper.
- $\Phi = \exp(A\Delta t)$ validated against `scipy.linalg.expm` as the matrix-
  exponential reference throughout.

## Non-goals for this pass

Only an IMU propagation model, one body-frame landmark measurement (right-
invariant), and one GNSS-style world-position measurement (left-invariant) are
implemented. Full InEKF-SLAM (landmarks *in* the state, given a proper invariant
tangent coordinate for them) is demonstrated only as a bounded, test-scoped
construction in `tests/test_observability.py` to validate the FEJ/InEKF
observability comparison — it is not exposed as a general kernel in this pass (see
the toolkit's stated non-goals).
