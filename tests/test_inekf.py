from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from diagnostics.covariance_health import check_covariance
from filters.inekf import InvariantEKF
from filters.lie import se23
from filters.lie.so3 import exp_so3, skew

GRAVITY = np.array([0.0, 0.0, -9.80665])


def _random_xi(rng, max_angle=1.0, max_translation=3.0):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0.0, max_angle)
    phi = axis * angle
    nu = rng.uniform(-max_translation, max_translation, size=3)
    rho = rng.uniform(-max_translation, max_translation, size=3)
    return np.concatenate([phi, nu, rho])


def _random_state(rng, **kwargs):
    return se23.exp(_random_xi(rng, **kwargs))


def _f(X, omega, a, g=GRAVITY):
    R, v, _p = se23.split_state(X)
    out = np.zeros((5, 5))
    out[:3, :3] = R @ skew(omega)
    out[:3, 3] = R @ a + g
    out[:3, 4] = v
    return out


def _flow(X0, omega, a, dt, substeps=50, g=GRAVITY, bg=None, ba=None, omega_m=None, a_m=None):
    """RK4 integration of the exact bias-free (or bias-parametrized) ODE,
    used only as an independent reference to certify the filter's closed
    forms -- not the production propagation implementation.
    """
    if bg is not None:
        def deriv(X):
            return _f(X, omega_m - bg, a_m - ba, g)
    else:
        def deriv(X):
            return _f(X, omega, a, g)

    h = dt / substeps
    X = X0.copy()
    for _ in range(substeps):
        k1 = deriv(X)
        k2 = deriv(X + 0.5 * h * k1)
        k3 = deriv(X + 0.5 * h * k2)
        k4 = deriv(X + h * k3)
        X = X + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        X[3, :] = [0, 0, 0, 1, 0]
        X[4, :] = [0, 0, 0, 0, 1]
    return X


# ---------------------------------------------------------------------------
# Group-affine property
# ---------------------------------------------------------------------------


def test_group_affine_property():
    """f(X1 X2) = f(X1) X2 + X1 f(X2) - X1 f(Id) X2, to tight tolerance.

    Note: the trailing X2 on the last term is required -- dropping it (as
    some informal statements of this identity do) fails by O(1), not by a
    numerical-precision-sized residual; see the module docstring.
    """
    rng = np.random.default_rng(0)
    omega = rng.normal(size=3)
    a = rng.normal(size=3)
    max_err = 0.0
    max_err_dropped = 0.0
    for _ in range(200):
        X1 = _random_state(rng)
        X2 = _random_state(rng)
        lhs = _f(X1 @ X2, omega, a)
        fX1, fX2, fId = _f(X1, omega, a), _f(X2, omega, a), _f(np.eye(5), omega, a)
        rhs = fX1 @ X2 + X1 @ fX2 - X1 @ fId @ X2
        rhs_dropped = fX1 @ X2 + X1 @ fX2 - X1 @ fId
        max_err = max(max_err, np.abs(lhs - rhs).max())
        max_err_dropped = max(max_err_dropped, np.abs(lhs - rhs_dropped).max())
    assert max_err < 1e-10
    assert max_err_dropped > 0.1  # confirms the trailing-X2 form is the one that actually holds


# ---------------------------------------------------------------------------
# Log-linearity (bias-free / exact case)
# ---------------------------------------------------------------------------


def _analytic_A_bias_free(mode, omega, a, g=GRAVITY):
    A = np.zeros((9, 9))
    if mode == "right":
        A[3:6, 0:3] = skew(g)
        A[6:9, 3:6] = np.eye(3)
    else:
        A[0:3, 0:3] = -skew(omega)
        A[3:6, 0:3] = -skew(a)
        A[3:6, 3:6] = -skew(omega)
        A[6:9, 3:6] = np.eye(3)
        A[6:9, 6:9] = -skew(omega)
    return A


def test_log_linearity_bias_free_large_errors():
    """The core, distinguishing InEKF claim: propagate a TRUE state and an
    ESTIMATE (different initial conditions, large error) under identical
    inputs, and verify the invariant error obeys xi(dt) = Phi @ xi(0) with a
    Phi = expm(A dt) that is the SAME regardless of which trajectory is used
    as the reference -- to numerical precision, for large (not just
    infinitesimal) initial errors. This is what makes InEKF different from a
    repackaged ESKF, where the analogous statement is only true to first
    order in the error.
    """
    rng = np.random.default_rng(1)
    omega = np.array([0.3, -0.2, 0.5])
    a = np.array([0.1, 0.2, -0.05])
    dt = 0.05

    for mode in ("right", "left"):
        A = _analytic_A_bias_free(mode, omega, a)
        Phi = expm(A * dt)
        max_err = 0.0
        for _ in range(30):
            X_true0 = _random_state(rng)
            xi0 = _random_xi(rng, max_angle=1.2, max_translation=4.0)  # large error
            X_est0 = se23.exp(xi0) @ X_true0 if mode == "right" else X_true0 @ se23.exp(xi0)

            X_true_dt = _flow(X_true0, omega, a, dt)
            X_est_dt = _flow(X_est0, omega, a, dt)

            eta_dt = X_est_dt @ se23.inverse(X_true_dt) if mode == "right" else se23.inverse(X_true_dt) @ X_est_dt
            xi_dt_actual = se23.log(eta_dt)
            xi_dt_predicted = Phi @ xi0
            max_err = max(max_err, np.linalg.norm(xi_dt_actual - xi_dt_predicted))
        assert max_err < 1e-6, f"{mode}-invariant log-linearity failed: max_err={max_err}"


def test_error_dynamics_matrix_independent_of_state_estimate():
    """Directly certifies the headline claim -- A (bias-free) does not
    depend on which true/estimate trajectory pair it's evaluated against.
    """
    rng = np.random.default_rng(2)
    omega = np.array([0.3, -0.2, 0.5])
    a = np.array([0.1, 0.2, -0.05])
    dt = 0.01

    for mode in ("right", "left"):
        Phis = []
        for _ in range(5):
            X_true0 = _random_state(rng)
            eps = 1e-6
            Phi = np.zeros((9, 9))
            for i in range(9):
                d = np.zeros(9)
                d[i] = eps
                X_est_p = se23.exp(d) @ X_true0 if mode == "right" else X_true0 @ se23.exp(d)
                X_est_m = se23.exp(-d) @ X_true0 if mode == "right" else X_true0 @ se23.exp(-d)
                X_true_dt = _flow(X_true0, omega, a, dt)
                Xp_dt = _flow(X_est_p, omega, a, dt)
                Xm_dt = _flow(X_est_m, omega, a, dt)
                if mode == "right":
                    xi_p = se23.log(Xp_dt @ se23.inverse(X_true_dt))
                    xi_m = se23.log(Xm_dt @ se23.inverse(X_true_dt))
                else:
                    xi_p = se23.log(se23.inverse(X_true_dt) @ Xp_dt)
                    xi_m = se23.log(se23.inverse(X_true_dt) @ Xm_dt)
                Phi[:, i] = (xi_p - xi_m) / (2 * eps)
            Phis.append(Phi)
        for Phi in Phis[1:]:
            assert np.allclose(Phi, Phis[0], atol=1e-4)


# ---------------------------------------------------------------------------
# Degradation with bias ("imperfect" InEKF)
# ---------------------------------------------------------------------------


def _analytic_A_with_bias(mode, R_hat, v_hat, p_hat, omega_hat, a_hat, g=GRAVITY):
    A = np.zeros((15, 15))
    if mode == "right":
        A[3:6, 0:3] = skew(g)
        A[6:9, 3:6] = np.eye(3)
        A[0:3, 9:12] = -R_hat
        A[3:6, 9:12] = -skew(v_hat) @ R_hat
        A[3:6, 12:15] = -R_hat
        A[6:9, 9:12] = -skew(p_hat) @ R_hat
    else:
        A[0:3, 0:3] = -skew(omega_hat)
        A[3:6, 0:3] = -skew(a_hat)
        A[3:6, 3:6] = -skew(omega_hat)
        A[6:9, 3:6] = np.eye(3)
        A[6:9, 6:9] = -skew(omega_hat)
        A[0:3, 9:12] = -np.eye(3)
        A[3:6, 12:15] = -np.eye(3)
    return A


def _build_Phi15_numeric(mode, X_true0, bg_true, ba_true, bg_est0, ba_est0, omega_m, a_m, dt):
    eps = 1e-6
    Phi = np.zeros((15, 15))
    def perturbed(dd):
        X_est0 = se23.exp(dd[:9]) @ X_true0 if mode == "right" else X_true0 @ se23.exp(dd[:9])
        return X_est0, bg_est0 + dd[9:12], ba_est0 + dd[12:15]

    for i in range(15):
        d = np.zeros(15)
        d[i] = eps

        X_p, bg_p, ba_p = perturbed(d)
        X_m, bg_m, ba_m = perturbed(-d)
        X_true_dt = _flow(X_true0, None, None, dt, bg=bg_true, ba=ba_true, omega_m=omega_m, a_m=a_m)
        X_p_dt = _flow(X_p, None, None, dt, bg=bg_p, ba=ba_p, omega_m=omega_m, a_m=a_m)
        X_m_dt = _flow(X_m, None, None, dt, bg=bg_m, ba=ba_m, omega_m=omega_m, a_m=a_m)
        if mode == "right":
            xi_p = se23.log(X_p_dt @ se23.inverse(X_true_dt))
            xi_m = se23.log(X_m_dt @ se23.inverse(X_true_dt))
        else:
            xi_p = se23.log(se23.inverse(X_true_dt) @ X_p_dt)
            xi_m = se23.log(se23.inverse(X_true_dt) @ X_m_dt)
        full_p = np.concatenate([xi_p, bg_p - bg_true, ba_p - ba_true])
        full_m = np.concatenate([xi_m, bg_m - bg_true, ba_m - ba_true])
        Phi[:, i] = (full_p - full_m) / (2 * eps)
    return Phi


def test_bias_error_dynamics_matches_nonlinear_propagation():
    """The 15-dim (with-bias) A used by InvariantEKF.predict must correctly
    linearize the TRUE nonlinear bias-augmented propagation at the current
    estimate -- checked against direct finite-differencing of the nonlinear
    flow, for both conventions. Residual should shrink as O(dt^2) (the
    expected frozen-Jacobian discretization error, the same approximation
    every filter in this repo already makes), not stay O(1).
    """
    rng = np.random.default_rng(3)
    omega_m = np.array([0.3, -0.2, 0.5])
    a_m = np.array([0.1, 0.2, -0.05])
    bg_true = np.array([0.01, -0.02, 0.005])
    ba_true = np.array([0.05, -0.03, 0.02])

    for mode in ("right", "left"):
        errs = []
        for dt in (0.01, 0.001):
            X_true0 = _random_state(rng, max_angle=0.8, max_translation=2.0)
            bg_est0 = bg_true + rng.normal(scale=0.001, size=3)
            ba_est0 = ba_true + rng.normal(scale=0.001, size=3)
            R_hat, v_hat, p_hat = se23.split_state(X_true0)
            omega_hat = omega_m - bg_est0
            a_hat = a_m - ba_est0

            Phi_num = _build_Phi15_numeric(mode, X_true0, bg_true, ba_true, bg_est0, ba_est0, omega_m, a_m, dt)
            A = _analytic_A_with_bias(mode, R_hat, v_hat, p_hat, omega_hat, a_hat)
            Phi_analytic = expm(A * dt)
            errs.append(np.abs(Phi_num - Phi_analytic).max())
        if mode == "right":
            # right-invariant has a genuine O(dt^2) frozen-Jacobian residual
            # (verified separately to scale as dt^2 down to dt=1e-4); at
            # dt=0.01 vs dt=0.001 it should shrink by roughly 100x.
            assert errs[1] < errs[0] / 20.0, f"{mode}: errors {errs} do not shrink as expected"
            assert errs[0] < 5e-3
        else:
            # left-invariant's bias coupling is already state-independent
            # (only through the known inputs/current bias estimate), so
            # there is no residual dt^2 term -- both errors sit at the
            # finite-difference noise floor (~1e-6), not a shrinking trend.
            assert errs[0] < 1e-4
            assert errs[1] < 1e-4


def test_log_linearity_degrades_with_bias():
    """Quantify (not hide) how much the log-linear property degrades once
    bias is estimated, for both conventions. Reports the degradation via
    stdout so it's visible in verbose test output; asserts only that the
    right-invariant convention degrades at least as much as left-invariant
    for this same scenario, consistent with the analytic finding that its
    bias-coupling block depends on the full (R,v,p) estimate while the
    left-invariant one depends only on the current bias estimate.
    """
    rng = np.random.default_rng(4)
    omega_m = np.array([0.3, -0.2, 0.5])
    a_m = np.array([0.1, 0.2, -0.05])
    bg_true = np.array([0.02, -0.01, 0.03])
    ba_true = np.array([0.04, -0.02, 0.01])

    degradation = {}
    for mode in ("right", "left"):
        residuals = []
        for _ in range(10):
            X_true0 = _random_state(rng, max_angle=0.8, max_translation=2.0)
            bg_est0 = bg_true + rng.normal(scale=0.02, size=3)
            ba_est0 = ba_true + rng.normal(scale=0.02, size=3)
            R_hat, v_hat, p_hat = se23.split_state(X_true0)
            omega_hat = omega_m - bg_est0
            a_hat = a_m - ba_est0
            A_at_this_ref = _analytic_A_with_bias(mode, R_hat, v_hat, p_hat, omega_hat, a_hat)
            residuals.append(A_at_this_ref)
        spread = max(np.abs(r - residuals[0]).max() for r in residuals[1:])
        degradation[mode] = spread
        print(f"\n[log-linearity degradation] {mode}-invariant: A spread across state references = {spread:.4f}")

    assert degradation["right"] >= degradation["left"]


# ---------------------------------------------------------------------------
# Measurement Jacobian constancy
# ---------------------------------------------------------------------------


def test_landmark_measurement_jacobian_is_constant_and_correct():
    rng = np.random.default_rng(5)
    p_L = np.array([5.0, -2.0, 1.0])
    H_R = np.zeros((3, 9))
    H_R[:, 0:3] = -skew(p_L)
    H_R[:, 6:9] = np.eye(3)

    for scale in (1.0, 0.1, 0.01):
        residuals = []
        for _ in range(15):
            X_true = _random_state(rng)
            xi = _random_xi(rng, max_angle=0.8 * scale, max_translation=2.0 * scale)
            X_hat = se23.exp(xi) @ X_true
            R_true, _v_true, p_true = se23.split_state(X_true)
            y = R_true.T @ (p_L - p_true)
            Y = np.array([*y, 0.0, 1.0])
            b = np.array([*p_L, 0.0, 1.0])
            Z = (X_hat @ Y - b)[:3]
            residuals.append(np.linalg.norm(Z - H_R @ xi))
        # residual should shrink quadratically in scale (confirms H is the
        # correct constant linearization, with only 2nd-order error left)
        ratio = np.mean(residuals) / scale**2
        assert ratio < 2.0


def test_gnss_measurement_jacobian_is_constant_and_correct():
    rng = np.random.default_rng(6)
    H_L = np.zeros((3, 9))
    H_L[:, 6:9] = np.eye(3)
    e5 = np.array([0.0, 0.0, 0.0, 0.0, 1.0])

    for scale in (1.0, 0.1, 0.01):
        residuals = []
        for _ in range(15):
            X_true = _random_state(rng)
            xi = _random_xi(rng, max_angle=0.8 * scale, max_translation=2.0 * scale)
            X_hat = X_true @ se23.exp(xi)
            _R_true, _v_true, p_true = se23.split_state(X_true)
            z = p_true
            Y = np.array([*z, 0.0, 1.0])
            Z = (e5 - se23.inverse(X_hat) @ Y)[:3]
            residuals.append(np.linalg.norm(Z - H_L @ xi))
        ratio = np.mean(residuals) / scale**2
        assert ratio < 2.0


# ---------------------------------------------------------------------------
# End-to-end filter behavior
# ---------------------------------------------------------------------------


def test_update_reduces_error_right_invariant_landmark():
    rng = np.random.default_rng(7)
    p_L = np.array([5.0, -2.0, 1.0])
    X_true = _random_state(rng, max_angle=0.5, max_translation=1.0)
    xi0 = _random_xi(rng, max_angle=0.4, max_translation=1.0)
    X_hat = se23.exp(xi0) @ X_true
    R_true, _v_true, p_true = se23.split_state(X_true)
    R_hat, v_hat, p_hat = se23.split_state(X_hat)

    filt = InvariantEKF(R0=R_hat, v0=v_hat, p0=p_hat, P0=np.eye(15), error_convention="right")
    err_before = np.linalg.norm(se23.log(filt.X @ se23.inverse(X_true)))
    y_body = R_true.T @ (p_L - p_true)
    filt.update_landmark_body(y_body, p_L, np.eye(3) * 1e-4)
    err_after = np.linalg.norm(se23.log(filt.X @ se23.inverse(X_true)))
    assert err_after < err_before


def test_update_reduces_error_left_invariant_gnss():
    rng = np.random.default_rng(8)
    X_true = _random_state(rng, max_angle=0.5, max_translation=1.0)
    xi0 = _random_xi(rng, max_angle=0.4, max_translation=1.0)
    X_hat = X_true @ se23.exp(xi0)
    R_true, _v_true, p_true = se23.split_state(X_true)
    R_hat, v_hat, p_hat = se23.split_state(X_hat)

    filt = InvariantEKF(R0=R_hat, v0=v_hat, p0=p_hat, P0=np.eye(15), error_convention="left")
    err_before = np.linalg.norm(se23.log(se23.inverse(X_true) @ filt.X))
    filt.update_position_world(p_true, np.eye(3) * 1e-4)
    err_after = np.linalg.norm(se23.log(se23.inverse(X_true) @ filt.X))
    assert err_after < err_before


def test_predict_smoke_and_state_shape():
    for mode in ("right", "left"):
        filt = InvariantEKF(error_convention=mode)
        for _ in range(50):
            filt.predict(gyro=np.array([0.1, 0.0, -0.05]), accel=np.array([0.2, 0.0, 9.8]), dt=0.01)
        assert filt.state_vector.shape == (21,)  # R(9) + v(3) + p(3) + b_g(3) + b_a(3)
        assert np.all(np.isfinite(filt.state_vector))
        assert np.allclose(filt.R @ filt.R.T, np.eye(3), atol=1e-8)


def test_bias_estimate_converges_not_diverges_over_many_updates():
    """Regression test: the bias injection in update() previously used '+'
    instead of '-' (inconsistent with the core state's injection sign),
    turning the correction into positive feedback -- the bias estimate grew
    monotonically away from the true value over a long run instead of
    converging, caught via the benchmark harness (ANEES exploding to
    ~10^4 on a 20s aggressive trajectory) rather than by this unit test
    suite, which is exactly the gap this test closes.
    """
    rng = np.random.default_rng(11)
    for mode in ("right", "left"):
        true_bg = np.array([0.02, -0.015, 0.01])
        true_ba = np.array([0.05, -0.03, 0.02])
        # realistic bias uncertainty (std ~0.05 rad/s and m/s^2) -- an
        # artificially huge P0 (e.g. std 0.3) makes even a CORRECT filter
        # overshoot for a while before settling, which isn't what this test
        # is checking; it checks the sign of the correction, not tuning.
        P0 = np.eye(15) * 0.01
        P0[9:15, 9:15] = np.eye(6) * 0.0025
        filt = InvariantEKF(bg0=np.zeros(3), ba0=np.zeros(3), P0=P0, error_convention=mode)
        R_true, v_true, p_true = np.eye(3), np.array([1.0, 0.0, 0.0]), np.zeros(3)
        landmark = np.array([5.0, 2.0, -1.0])
        dt = 0.02
        n_steps = 1500
        bg_errs = []
        for step in range(n_steps):
            gyro_true = np.array([0.1, -0.05, 0.2])
            accel_true = np.array([0.3, 0.1, 0.0])
            R_true = R_true @ exp_so3(gyro_true * dt)
            v_true = v_true + (R_true @ accel_true + GRAVITY) * dt
            p_true = p_true + v_true * dt

            gyro_meas = gyro_true + true_bg + rng.normal(scale=0.001, size=3)
            accel_meas = accel_true + true_ba + rng.normal(scale=0.001, size=3)
            filt.predict(gyro_meas, accel_meas, dt)

            if step % 10 == 0:
                if mode == "right":
                    y_body = R_true.T @ (landmark - p_true) + rng.normal(scale=0.01, size=3)
                    filt.update_landmark_body(y_body, landmark, np.eye(3) * 1e-4)
                else:
                    filt.update_position_world(p_true + rng.normal(scale=0.05, size=3), np.eye(3) * 0.0025)

            if step % 100 == 0:
                bg_errs.append(np.linalg.norm(filt.b_g - true_bg))

        # bias error must end up much smaller than it started (the bug made
        # it grow without bound instead); the last quarter of the run is
        # used for the "settled" comparison so early transient overshoot
        # doesn't make this test flaky.
        settled = np.mean(bg_errs[-len(bg_errs) // 4 :])
        assert settled < 0.6 * bg_errs[0], f"{mode}: bias error did not converge: {bg_errs}"


def test_covariance_health_after_many_steps():
    rng = np.random.default_rng(9)
    for mode in ("right", "left"):
        filt = InvariantEKF(error_convention=mode, P0=np.eye(15) * 0.1)
        for step in range(500):
            gyro = np.array([0.1, -0.05, 0.02]) + rng.normal(scale=0.01, size=3)
            accel = np.array([0.0, 0.0, 9.80665]) + rng.normal(scale=0.05, size=3)
            filt.predict(gyro, accel, dt=0.01)
            if step % 20 == 0:
                r_meas = np.eye(3) * 0.01
                if mode == "right":
                    landmark = np.array([1.0, 2.0, 0.0])
                    filt.update_landmark_body(rng.normal(scale=0.1, size=3), landmark, r_meas)
                else:
                    filt.update_position_world(filt.p + rng.normal(scale=0.1, size=3), r_meas)
        health = check_covariance(filt.P)
        assert health["healthy"]
