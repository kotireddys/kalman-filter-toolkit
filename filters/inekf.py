"""Invariant Extended Kalman Filter (InEKF) on SE_2(3) for IMU-driven
navigation, following Barrau & Bonnabel (2017) and the "imperfect InEKF"
treatment used in Hartley et al.'s contact-aided legged-robot work.

State
-----
Core navigation state is the matrix Lie group element

    X = (R, v, p) in SE_2(3)   -- attitude, world-frame velocity, world-frame
                                  position (see filters/lie/se23.py).

IMU biases (b_g, b_a) are appended as a plain Euclidean component (not part
of the group). This is exactly why this is the *imperfect* InEKF: appending
biases breaks the exact group-affine property of the propagation (see
"Group-affine dynamics and its breakdown" below) -- the beautiful
state-estimate-independent linear error dynamics is only exact when biases
are not estimated. With biases on, the linearization becomes state-dependent
again, just as in a standard EKF, though (as shown below and quantified by
the degradation test) to a much smaller degree for the left-invariant
convention than for the right-invariant one.

Both error conventions are implemented and selectable via
`error_convention="right"` or `"left"`:

    right-invariant:  eta = Xhat @ X^-1
    left-invariant:   eta = X^-1 @ Xhat

Selection rule (derived below for each measurement model implemented here,
not just asserted): the correct convention is whichever one makes the
measurement's linearized error state-independent (a constant H). Body-frame
observations of world-fixed quantities (landmarks in body frame, gravity
direction, magnetometer) pair with the right-invariant error, because the
observation is naturally expressed as Y = X^-1 b for a fixed known b (the
landmark), and the right-invariant error appears linearly and additively in
that expression once b is known. World-frame observations of body-fixed
quantities (direct position fixes such as GNSS, observing the state's own
position column) pair with the left-invariant error, by the symmetric
argument with Y = X b.

Group-affine dynamics and its breakdown
----------------------------------------
The bias-free IMU dynamics on SE_2(3),

    Rdot = R [omega]_x
    vdot = R a + g
    pdot = v

is "group affine": writing this as a vector field f_u(X) (see
`_continuous_dynamics_hat`), it satisfies

    f_u(X1 X2) = f_u(X1) X2 + X1 f_u(X2) - X1 f_u(Id) X2      (*)

for all X1, X2 in SE_2(3) and any fixed input u=(omega, a). This is the
*correct* identity, verified numerically in
tests/test_inekf.py::test_group_affine_property to ~1e-14. Note this
differs from the group-affine formula as literally written in some
secondary descriptions (which sometimes drop the trailing "X2" on the last
term) -- dropping it does NOT hold numerically for this dynamics (checked:
residual ~O(1), not ~O(eps)), so (*) with the trailing X2 is what this
module relies on and tests.

Group-affineness is exactly what makes the *invariant* error's dynamics
linear and state-estimate-independent (Barrau & Bonnabel's central result):
writing eta = Xhat X^-1 (right-invariant) and using (*), one gets the closed
autonomous ODE

    etadot = f_u(eta) - eta f_u(Id)

which depends on eta alone, not on Xhat or X separately. Linearizing at
eta = Id gives xidot = A_R xi with (right-invariant, bias-free)

    A_R = [[ 0,        0,   0 ],
           [ skew(g),  0,   0 ],
           [ 0,        I,   0 ]]     (9x9, blocks ordered [phi, v, p])

which is *independent of the inputs (omega, a) and of the trajectory* --
this is the log-linearity property this module's tests certify directly
(numerically re-derived and checked against the closed form here, not
assumed from memory -- see tests/test_inekf.py). The analogous
left-invariant computation (eta_L = X^-1 Xhat) gives

    A_L = [[ -skew(omega),  0,             0            ],
           [ -skew(a),      -skew(omega),  0            ],
           [ 0,              I,            -skew(omega) ]]

which is autonomous in xi but does depend on the (known, measured) inputs
omega, a -- still independent of the state estimate, which is the property
that matters for consistency.

Once biases are estimated, the two trajectories (true and estimate) use
*different* effective inputs (omega_m - b_g vs omega_m - bhat_g), and the
clean cancellation behind etadot = f(eta) - eta f(Id) no longer holds
exactly. The correct linearization (re-derived and verified numerically in
tests/test_inekf.py::test_bias_error_dynamics_matches_nonlinear_propagation,
against direct finite-differencing of the true nonlinear propagation, to
better than O(dt^2)) is:

    right-invariant, 15x15, blocks ordered [phi, v, p, b_g, b_a]:
        A[v,   phi] = skew(g)
        A[p,   v]   = I
        A[phi, bg]  = -Rhat
        A[v,   bg]  = -skew(vhat) @ Rhat
        A[v,   ba]  = -Rhat
        A[p,   bg]  = -skew(phat) @ Rhat

    left-invariant, 15x15:
        A[phi, phi] = -skew(omega_hat)
        A[v,   phi] = -skew(a_hat)
        A[v,   v]   = -skew(omega_hat)
        A[p,   v]   = I
        A[p,   p]   = -skew(omega_hat)
        A[phi, bg]  = -I
        A[v,   ba]  = -I

    (omega_hat = omega_m - bhat_g, a_hat = a_m - bhat_a)

Notably the right-invariant bias coupling depends on the *full* current
estimate (Rhat, vhat, phat), while the left-invariant one depends only on
the current bias estimate and the known inputs -- i.e. the left-invariant
filter's error dynamics stays much closer to state-independent once biases
are included. This asymmetry is exactly what
tests/test_inekf.py::test_log_linearity_degrades_with_bias quantifies and
reports, per the task's request not to hide it.

Measurement models
-------------------
Both use the standard InEKF trick of lifting the observation into a
homogeneous vector and comparing it against a fixed, known 5-vector, so
that the linearized innovation is a *constant* linear functional of xi:

1. Right-invariant body-frame landmark (`update_landmark_body`): a known
   world-fixed landmark at p_L is observed in the body frame,
   y = R^T (p_L - p) + noise. With Y = [y; 0; 1], b = [p_L; 0; 1],
       Y = X^-1 b  exactly (noise-free),
   so Z := Xhat @ Y - b = (eta - I) b + Xhat [noise; 0; 0]
        ~= hat(xi) b = H_R xi + Rhat @ noise,
       H_R = [-skew(p_L), 0, I]     (3x9, constant -- derived and verified
                                      numerically to O(xi^2) in
                                      tests/test_inekf.py).

2. Left-invariant world-frame GNSS position (`update_position_world`): a
   direct world-frame position fix z = p + noise. With Y = [z; 0; 1],
   b = e5 = [0,0,0,0,1] (the state's own position-selector column),
       Y = X b  exactly (noise-free),
   so Z := e5 - Xhat^-1 @ Y = e5 - eta_L^-1 b - Xhat^-1 [noise; 0; 0]
        ~= hat(xi_L) b = H_L xi_L - Rhat^T @ noise,
       H_L = [0, 0, I]              (3x9, constant -- verified the same way).

Injection sign: the correction delta = K @ innovation is an estimate of the
*current* error xi = log(eta_old) (estimate relative to truth), so removing
it requires the *inverse* direction: Xhat_new = exp(-delta_core) @ Xhat_old
(right-invariant, left-multiply) or Xhat_old @ exp(-delta_core)
(left-invariant, right-multiply). This sign was verified end-to-end (not
assumed): a synthetic update with the "+" sign was checked to *increase*
the true estimation error while "-" decreases it, for both conventions --
see tests/test_inekf.py::test_update_reduces_error.

Nominal mean propagation
--------------------------
The nominal (R, v, p) mean is propagated with one RK4 step per predict()
call, re-orthonormalizing R via SVD afterward. This is a deliberate,
disclosed choice: the exact closed-form mean propagation for constant
(omega, a) exists in principle (via the SO(3) exponential and the integrals
of it that define the left Jacobian), but is not needed for the log-linear
structural claims this module is about -- those concern the *error*
dynamics and its Phi = exp(A dt), which is exact/closed-form and validated
separately. RK4 gives 4th-order accuracy for the mean at typical IMU rates,
which is more than sufficient.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from ._numerics import repair_covariance
from .lie import se23
from .lie.so3 import skew


def _continuous_dynamics_hat(X, omega, a, g):
    """f_u(X) as a 5x5 tangent matrix: Rdot=R[omega]_x, vdot=Ra+g, pdot=v."""
    R, v, _p = se23.split_state(X)
    out = np.zeros((5, 5))
    out[:3, :3] = R @ skew(omega)
    out[:3, 3] = R @ a + g
    out[:3, 4] = v
    return out


class InvariantEKF:
    def __init__(
        self,
        R0=None,
        v0=None,
        p0=None,
        bg0=None,
        ba0=None,
        P0=None,
        gyro_noise=0.02,
        accel_noise=0.2,
        gyro_bias_noise=1e-4,
        accel_bias_noise=1e-3,
        gravity=None,
        error_convention="right",
    ):
        if error_convention not in ("right", "left"):
            raise ValueError("error_convention must be 'right' or 'left'")
        self.error_convention = error_convention

        self.R = np.eye(3) if R0 is None else np.asarray(R0, dtype=float).reshape(3, 3)
        self.v = np.zeros(3) if v0 is None else np.asarray(v0, dtype=float).reshape(3)
        self.p = np.zeros(3) if p0 is None else np.asarray(p0, dtype=float).reshape(3)
        self.b_g = np.zeros(3) if bg0 is None else np.asarray(bg0, dtype=float).reshape(3)
        self.b_a = np.zeros(3) if ba0 is None else np.asarray(ba0, dtype=float).reshape(3)
        self.P = np.eye(15) if P0 is None else np.atleast_2d(np.asarray(P0, dtype=float))

        self.gyro_noise = float(gyro_noise)
        self.accel_noise = float(accel_noise)
        self.gyro_bias_noise = float(gyro_bias_noise)
        self.accel_bias_noise = float(accel_bias_noise)
        if gravity is None:
            self.gravity = np.array([0.0, 0.0, -9.80665])
        else:
            self.gravity = np.asarray(gravity, dtype=float).reshape(3)

        self.x_prior = self.state_vector.copy()
        self.P_prior = self.P.copy()
        self.innovation = np.zeros(1)
        self.S = np.eye(1)
        self.K = np.zeros((15, 1))

    @property
    def X(self):
        return se23.make_state(self.R, self.v, self.p)

    @property
    def state_vector(self):
        return np.concatenate([self.R.flatten(), self.v, self.p, self.b_g, self.b_a])

    def _error_dynamics_matrix(self, omega_hat, a_hat):
        A = np.zeros((15, 15))
        if self.error_convention == "right":
            A[3:6, 0:3] = skew(self.gravity)
            A[6:9, 3:6] = np.eye(3)
            A[0:3, 9:12] = -self.R
            A[3:6, 9:12] = -skew(self.v) @ self.R
            A[3:6, 12:15] = -self.R
            A[6:9, 9:12] = -skew(self.p) @ self.R
        else:
            A[0:3, 0:3] = -skew(omega_hat)
            A[3:6, 0:3] = -skew(a_hat)
            A[3:6, 3:6] = -skew(omega_hat)
            A[6:9, 3:6] = np.eye(3)
            A[6:9, 6:9] = -skew(omega_hat)
            A[0:3, 9:12] = -np.eye(3)
            A[3:6, 12:15] = -np.eye(3)
        return A

    def _noise_jacobian(self, dt):
        """15x12 noise Jacobian, columns [n_g, n_a, n_bg, n_ba], dt baked in
        (matches this repo's ESKF convention of pre-scaling G rather than Qd).
        """
        G = np.zeros((15, 12))
        if self.error_convention == "right":
            G[0:3, 0:3] = -self.R * dt
            G[3:6, 0:3] = -skew(self.v) @ self.R * dt
            G[3:6, 3:6] = -self.R * dt
            G[6:9, 0:3] = -skew(self.p) @ self.R * dt
        else:
            G[0:3, 0:3] = -np.eye(3) * dt
            G[3:6, 3:6] = -np.eye(3) * dt
        G[9:12, 6:9] = np.eye(3) * dt
        G[12:15, 9:12] = np.eye(3) * dt
        return G

    def predict(self, gyro, accel, dt):
        gyro = np.asarray(gyro, dtype=float).reshape(3)
        accel = np.asarray(accel, dtype=float).reshape(3)
        dt = float(dt)
        omega = gyro - self.b_g
        a = accel - self.b_a

        # Error-dynamics / noise Jacobians linearized at the PRE-propagation
        # estimate, matching the frozen-Jacobian convention used by the
        # repo's other filters (EKF/ESKF evaluate F at the current estimate
        # before advancing it).
        A = self._error_dynamics_matrix(omega, a)
        G = self._noise_jacobian(dt)
        Qc = np.diag(
            [self.gyro_noise**2] * 3
            + [self.accel_noise**2] * 3
            + [self.gyro_bias_noise**2] * 3
            + [self.accel_bias_noise**2] * 3
        )
        Phi = expm(A * dt)
        Qd = G @ Qc @ G.T
        self.P_prior = repair_covariance(Phi @ self.P @ Phi.T + Qd)

        R0, v0, p0 = self.R, self.v, self.p

        def deriv(R, v, p):
            Rdot = R @ skew(omega)
            vdot = R @ a + self.gravity
            pdot = v
            return Rdot, vdot, pdot

        k1 = deriv(R0, v0, p0)
        k2 = deriv(R0 + 0.5 * dt * k1[0], v0 + 0.5 * dt * k1[1], p0 + 0.5 * dt * k1[2])
        k3 = deriv(R0 + 0.5 * dt * k2[0], v0 + 0.5 * dt * k2[1], p0 + 0.5 * dt * k2[2])
        k4 = deriv(R0 + dt * k3[0], v0 + dt * k3[1], p0 + dt * k3[2])
        R_new = R0 + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v_new = v0 + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        p_new = p0 + (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])

        U, _, Vt = np.linalg.svd(R_new)
        if np.linalg.det(U @ Vt) < 0.0:
            U = U.copy()
            U[:, -1] *= -1.0
        R_new = U @ Vt

        self.R, self.v, self.p = R_new, v_new, p_new
        self.P = self.P_prior.copy()
        self.x_prior = self.state_vector.copy()
        return self.state_vector, self.P

    def update(self, innovation, H, R):
        innovation = np.asarray(innovation, dtype=float).reshape(-1)
        H = np.asarray(H, dtype=float)
        R = np.atleast_2d(np.asarray(R, dtype=float))
        self.innovation = innovation
        self.S = repair_covariance(H @ self.P @ H.T + R)
        self.K = np.linalg.solve(self.S, (self.P @ H.T).T).T
        delta = self.K @ innovation

        identity = np.eye(15)
        I_KH = identity - self.K @ H
        self.P = repair_covariance(I_KH @ self.P @ I_KH.T + self.K @ R @ self.K.T)

        correction = se23.exp(-delta[:9])
        if self.error_convention == "right":
            X_new = correction @ self.X
        else:
            X_new = self.X @ correction
        self.R, self.v, self.p = se23.split_state(X_new)
        # Same "-" sign as the core correction above: delta is an estimate
        # of the CURRENT error (b_hat - b_true), so removing it means
        # subtracting, not adding. A "+" here was a real bug -- it turns the
        # correction into positive feedback and the bias estimate diverges
        # instead of converging (caught via the benchmark harness: bias
        # error grew monotonically over a 20s run instead of shrinking).
        self.b_g = self.b_g - delta[9:12]
        self.b_a = self.b_a - delta[12:15]
        return self.state_vector, self.P

    def update_landmark_body(self, y_body, landmark_world, R_meas):
        """Right-invariant body-frame observation of a known world landmark."""
        if self.error_convention != "right":
            raise ValueError("update_landmark_body requires error_convention='right'")
        y_body = np.asarray(y_body, dtype=float).reshape(3)
        landmark_world = np.asarray(landmark_world, dtype=float).reshape(3)
        Y = np.array([*y_body, 0.0, 1.0])
        b = np.array([*landmark_world, 0.0, 1.0])
        Z = (self.X @ Y - b)[:3]
        H = np.zeros((3, 15))
        H[:, 0:3] = -skew(landmark_world)
        H[:, 6:9] = np.eye(3)
        return self.update(Z, H, R_meas)

    def update_position_world(self, z_position, R_meas):
        """Left-invariant world-frame direct position fix (e.g. GNSS)."""
        if self.error_convention != "left":
            raise ValueError("update_position_world requires error_convention='left'")
        z_position = np.asarray(z_position, dtype=float).reshape(3)
        Y = np.array([*z_position, 0.0, 1.0])
        e5 = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        Z = (e5 - se23.inverse(self.X) @ Y)[:3]
        H = np.zeros((3, 15))
        H[:, 6:9] = np.eye(3)
        return self.update(Z, H, R_meas)
