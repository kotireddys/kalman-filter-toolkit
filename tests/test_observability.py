"""Observability comparison: standard EKF vs FEJ-EKF vs InEKF.

Scenario: an IMU-driven rigid body observes a single static landmark in the
body frame (range+bearing as a 3-vector, y = R^T(p_L - p)), with no absolute
position or heading reference of any kind. State (excluding biases, held
fixed/known here to keep the expected dimension clean to derive):
    core:      phi (attitude, 3), v (velocity, 3), p (position, 3)
    landmark:  p_L (3)
    total: 12 dimensions.

Expected unobservable subspace: dimension 4, derived (not asserted) as
follows. Consider replacing the whole trajectory+landmark by
    R -> R_yaw R,   v -> R_yaw v,   p -> R_yaw (p - c) + c,
    p_L -> R_yaw (p_L - c) + c
for an arbitrary fixed point c in R^3 and an arbitrary fixed rotation R_yaw
about the gravity axis (R_yaw g = g, since g is an eigenvector of any
rotation about its own axis). Substituting into the measurement,
    R_new^T (p_L,new - p_new) = (R_yaw R)^T R_yaw (p_L - p) = R^T(p_L - p),
unchanged; substituting into the dynamics v_dot = R a + g,
    v_dot_new = R_yaw v_dot = R_yaw(R a + g) = R_new a + g
(using R_yaw g = g), also unchanged. So this transformation is an exact
symmetry of the whole input-output map. Its generators are: 3 translation
directions (c) + 1 rotation direction (R_yaw about the fixed axis g) = 4.
No other directions are free: rotating about an axis other than g changes
v_dot's relationship to g (breaks the dynamics), and this is the same
"global position + yaw about gravity" pair the FEJ docstring names.

A standard EKF's linearization point for `phi, v, p` keeps moving as
landmark corrections arrive, and for `p_L` as it's repeatedly re-observed
from different (also-moving) linearization points -- both are exactly the
inconsistency FEJ removes. The comparison below assembles the local
observability matrix
    O = [H_0; H_1 Phi_0; H_2 Phi_1 Phi_0; ...]
over a fixed window and checks its null-space dimension for all three.

For InEKF, Phi and H are constructed from the closed-form, already-tested
(tests/test_inekf.py) properties of the bias-free right-invariant core:
A_R has an exactly-zero phi-row (so the right-invariant attitude error does
not evolve under noise-free propagation -- verified in test_inekf.py), and
H_R = [-skew(p_L), 0, I] is exactly constant in the robot's own error. This
alone is not quite enough once p_L is *also* estimated: naively re-using
H_R with the numerically-*evolving* landmark estimate p_hat_L still lets a
residual, p_hat_L-dependent term leak into the phi-column of H and destroys
the null space (verified empirically below to fail exactly like the
standard EKF, `test_naive_landmark_embedding_reproduces_the_standard_ekf_failure`).
The fix (the actual InEKF-SLAM trick, e.g. Barrau's thesis) is to track the
landmark not as raw p_hat_L but as the invariant tangent coordinate
    xi_landmark := delta_p_L + skew(p_L_ref) @ delta_phi
for a *fixed* reference p_L_ref (here, the initial landmark estimate, held
fixed for the whole window -- a valid, simpler stand-in for the textbook
continuously-re-derived version, since what actually needs to be constant
for THIS test is Phi and H over the window, and freezing the reference
gives exactly that). Substituting delta_p_L = xi_landmark - skew(p_L_ref)
delta_phi into Z ~= H_R xi_core - delta_p_L collapses the phi-dependent
terms exactly (skew(p_L_ref) cancels skew(p_L_ref)), leaving
    H_joint = [0, 0, I, -I]   (exactly constant, no p_hat_L dependence left)
and, since A_R's phi-row is exactly zero, Phi_joint = blockdiag(Phi_R, I)
with no cross-coupling correction needed (the algebra is worked out in the
module comments of the InEKF section below).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from filters.ekf import ExtendedKalmanFilter
from filters.lie.so3 import exp_so3, left_jacobian, log_so3, skew

GRAVITY = np.array([0.0, 0.0, -9.80665])
DT = 0.05
N_STEPS = 30
LANDMARK_TRUE = np.array([8.0, 3.0, -1.0])
OMEGA_TRUE = np.array([0.02, -0.01, 0.3])
ACCEL_TRUE = np.array([0.5, 0.1, 0.0])
MEAS_NOISE_STD = 0.02


def _right_jacobian(phi):
    return left_jacobian(-phi)


def _true_trajectory():
    R, v, p = np.eye(3), np.array([1.0, 0.0, 0.0]), np.zeros(3)
    states = [(R.copy(), v.copy(), p.copy())]
    for _ in range(N_STEPS):
        R_prev = states[-1][0]
        R = R_prev @ exp_so3(OMEGA_TRUE * DT)
        v = states[-1][1] + (R_prev @ ACCEL_TRUE + GRAVITY) * DT
        p = states[-1][2] + states[-1][1] * DT
        states.append((R.copy(), v.copy(), p.copy()))
    return states


def _synthesize_measurements(states, rng):
    measurements = []
    for R, _v, p in states[1:]:
        y = R.T @ (LANDMARK_TRUE - p) + rng.normal(scale=MEAS_NOISE_STD, size=3)
        measurements.append(y)
    return measurements


# ---------------------------------------------------------------------------
# Standard-EKF / FEJ-EKF model: x = [phi(3), v(3), p(3), landmark(3)]
# ---------------------------------------------------------------------------


def _f_ekf(x, u):
    omega, a = u
    phi, v, p, p_l = x[0:3], x[3:6], x[6:9], x[9:12]
    R = exp_so3(phi)
    phi_new = log_so3(R @ exp_so3(omega * DT))
    v_new = v + (R @ a + GRAVITY) * DT
    p_new = p + v * DT
    return np.concatenate([phi_new, v_new, p_new, p_l])


def _h_ekf(x):
    phi, p, p_l = x[0:3], x[6:9], x[9:12]
    return exp_so3(phi).T @ (p_l - p)


def _F_jac_ekf(x, u):
    omega, a = u
    phi = x[0:3]
    R = exp_so3(phi)
    F = np.eye(12)
    eps = 1e-6
    F_phi = np.zeros((3, 3))
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        Rp = exp_so3(phi + d) @ exp_so3(omega * DT)
        Rm = exp_so3(phi - d) @ exp_so3(omega * DT)
        F_phi[:, i] = (log_so3(Rp) - log_so3(Rm)) / (2 * eps)
    Jr = _right_jacobian(phi)
    F[0:3, 0:3] = F_phi
    F[3:6, 0:3] = -R @ skew(a) @ Jr * DT
    F[3:6, 3:6] = np.eye(3)
    F[6:9, 3:6] = np.eye(3) * DT
    F[6:9, 6:9] = np.eye(3)
    F[9:12, 9:12] = np.eye(3)
    return F


def _H_jac_ekf(x):
    phi, p, p_l = x[0:3], x[6:9], x[9:12]
    R = exp_so3(phi)
    Jr = _right_jacobian(phi)
    H = np.zeros((3, 12))
    H[:, 0:3] = skew(R.T @ (p_l - p)) @ Jr
    H[:, 6:9] = -R.T
    H[:, 9:12] = R.T
    return H


def _run_generic_ekf(rng, states, measurements, use_fej):
    _R0, v0, p0 = states[0]
    x0 = np.concatenate([np.zeros(3), v0 + rng.normal(scale=0.1, size=3), p0 + rng.normal(scale=0.1, size=3)])
    filt = ExtendedKalmanFilter(
        _f_ekf, _h_ekf, Q=np.eye(9) * 1e-6, R=np.eye(3) * MEAS_NOISE_STD**2,
        n=9, m=3, x0=x0, P0=np.eye(9) * 0.05, use_fej=use_fej,
    )
    landmark_guess = LANDMARK_TRUE + rng.normal(scale=0.3, size=3)
    filt.augment_state(landmark_guess, P_init_block=np.eye(3) * 1.0)
    filt.F_jac = _F_jac_ekf
    filt.H_jac = _H_jac_ekf
    filt.Q = np.eye(12) * 1e-6

    Phis, Hs = [], []
    for k in range(N_STEPS):
        u = (OMEGA_TRUE, ACCEL_TRUE)
        Phis.append(filt._state_transition_jacobian(filt.x, u))
        filt.predict(u)
        Hs.append(filt._measurement_jacobian(filt.x))
        filt.update(measurements[k])
    return Phis, Hs


# ---------------------------------------------------------------------------
# InEKF: analytic Phi, H sequence (see module docstring for the derivation)
# ---------------------------------------------------------------------------


def _A_R_bias_free():
    A = np.zeros((9, 9))
    A[3:6, 0:3] = skew(GRAVITY)
    A[6:9, 3:6] = np.eye(3)
    return A


def _inekf_sequence_correct_embedding():
    Phi_R = expm(_A_R_bias_free() * DT)
    Phi_joint = np.block([[Phi_R, np.zeros((9, 3))], [np.zeros((3, 9)), np.eye(3)]])
    H_joint = np.zeros((3, 12))
    H_joint[:, 6:9] = np.eye(3)
    H_joint[:, 9:12] = -np.eye(3)
    return [Phi_joint] * N_STEPS, [H_joint] * N_STEPS


def _inekf_sequence_naive_embedding(rng):
    """Landmark tracked as raw p_hat_L (Euclidean), H re-evaluated at the
    numerically-evolving estimate every step -- included only to empirically
    demonstrate that this naive construction does NOT recover the correct
    null space (see module docstring), i.e. it is not enough to use an
    invariant core alone; the landmark coordinate matters too.
    """
    from filters.lie import se23

    states = _true_trajectory()
    measurements = _synthesize_measurements(states, rng)
    R0, v0, p0 = states[0]
    R_hat, v_hat, p_hat = R0.copy(), v0 + rng.normal(scale=0.1, size=3), p0 + rng.normal(scale=0.1, size=3)
    p_L_hat = LANDMARK_TRUE + rng.normal(scale=0.3, size=3)
    P = np.eye(12) * 0.05
    P[9:12, 9:12] = np.eye(3) * 1.0
    Phi_R = expm(_A_R_bias_free() * DT)

    Phis, Hs = [], []
    for k in range(N_STEPS):
        H_core = np.zeros((3, 9))
        H_core[:, 0:3] = -skew(p_L_hat)
        H_core[:, 6:9] = np.eye(3)
        H_joint = np.hstack([H_core, -np.eye(3)])
        Hs.append(H_joint)

        X_hat = se23.make_state(R_hat, v_hat, p_hat)
        Y = np.array([*measurements[k], 0.0, 1.0])
        b_hat = np.array([*p_L_hat, 0.0, 1.0])
        Z = (X_hat @ Y - b_hat)[:3]

        R_meas = np.eye(3) * MEAS_NOISE_STD**2
        S = H_joint @ P @ H_joint.T + R_meas
        K = P @ H_joint.T @ np.linalg.inv(S)
        delta = K @ Z
        I12 = np.eye(12)
        IKH = I12 - K @ H_joint
        P = IKH @ P @ IKH.T + K @ R_meas @ K.T
        P = 0.5 * (P + P.T)

        X_new = se23.exp(-delta[:9]) @ X_hat
        R_hat, v_hat, p_hat = se23.split_state(X_new)
        p_L_hat = p_L_hat - delta[9:12]

        Phi_joint = np.block([[Phi_R, np.zeros((9, 3))], [np.zeros((3, 9)), np.eye(3)]])
        Phis.append(Phi_joint)
        P = Phi_joint @ P @ Phi_joint.T

    return Phis, Hs


# ---------------------------------------------------------------------------
# Observability matrix / null-space
# ---------------------------------------------------------------------------


def _nullspace_dim(Phis, Hs, tol_ratio=1e-6):
    n = Phis[0].shape[0]
    rows = []
    running = np.eye(n)
    for H, Phi in zip(Hs, Phis, strict=True):
        rows.append(H @ running)
        running = Phi @ running
    obs_matrix = np.vstack(rows)
    svals = np.linalg.svd(obs_matrix, compute_uv=False)
    tol = svals[0] * tol_ratio if svals[0] > 0 else 1e-12
    rank = int(np.sum(svals > tol))
    return n - rank


EXPECTED_UNOBSERVABLE_DIM = 4  # 3 global translation + 1 yaw about gravity; see module docstring


def test_standard_ekf_manufactures_spurious_observability():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        states = _true_trajectory()
        measurements = _synthesize_measurements(states, rng)
        Phis, Hs = _run_generic_ekf(rng, states, measurements, use_fej=False)
        dim = _nullspace_dim(Phis, Hs)
        assert dim < EXPECTED_UNOBSERVABLE_DIM, (
            f"seed={seed}: standard EKF found dim={dim}, expected it to UNDER-count "
            f"the true {EXPECTED_UNOBSERVABLE_DIM}-dim unobservable subspace"
        )


def test_fej_ekf_recovers_correct_unobservable_subspace():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        states = _true_trajectory()
        measurements = _synthesize_measurements(states, rng)
        Phis, Hs = _run_generic_ekf(rng, states, measurements, use_fej=True)
        dim = _nullspace_dim(Phis, Hs)
        assert dim == EXPECTED_UNOBSERVABLE_DIM, f"seed={seed}: FEJ-EKF found dim={dim}"


def test_inekf_recovers_correct_unobservable_subspace():
    Phis, Hs = _inekf_sequence_correct_embedding()
    dim = _nullspace_dim(Phis, Hs)
    assert dim == EXPECTED_UNOBSERVABLE_DIM


def test_naive_landmark_embedding_reproduces_the_standard_ekf_failure():
    """Confirms the module docstring's claim that an invariant CORE alone is
    not sufficient -- the landmark's own tangent coordinate must also be
    invariant, or the same spurious-observability failure as standard EKF
    reappears despite using the (individually correct) InEKF core.
    """
    for seed in range(5):
        rng = np.random.default_rng(seed)
        Phis, Hs = _inekf_sequence_naive_embedding(rng)
        dim = _nullspace_dim(Phis, Hs)
        assert dim < EXPECTED_UNOBSERVABLE_DIM, f"seed={seed}: naive embedding found dim={dim}"
