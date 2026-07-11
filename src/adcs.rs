// ADCS: attitude determination and control.
//
// Quaternion convention: scalar-first [q0, q1, q2, q3]. The propagator in
// physics.rs integrates q̇ = ½·q ⊗ [0, ω_body], which makes the satellite
// quaternion the BODY→ECI rotation (R(q) maps body-frame vectors into ECI).
// Targets and error quaternions here follow that same convention.

use crate::math::{cross, norm, normalize, quat_conj, quat_mult, normalize_q, Lcg};
use crate::models::Satellite;

/// Controller gains and actuator saturation limits.
#[derive(Debug, Clone)]
pub struct AdcsGains {
    /// Proportional gain on the quaternion error vector (1/s^2, inertia-normalized).
    pub kp: f64,
    /// Derivative gain on the body rate error (1/s, inertia-normalized).
    pub kd: f64,
    /// Per-axis reaction wheel torque saturation (Nm).
    pub rw_torque_max: f64,
    /// Per-axis magnetorquer dipole saturation (A·m^2).
    pub mtq_dipole_max: f64,
    /// Wheel momentum dumping gain for the magnetorquer cross-product law.
    pub k_dump: f64,
    /// Wheel momentum magnitude (Nms) above which the dumping law activates;
    /// below it the MTQ stays quiet so it doesn't perturb fine pointing.
    pub h_dump_threshold: f64,
}

impl Default for AdcsGains {
    fn default() -> Self {
        Self {
            kp: 0.02,
            kd: 0.2,
            rw_torque_max: 0.02,
            mtq_dipole_max: 5.0,
            k_dump: 1e-3,
            h_dump_threshold: 0.05,
        }
    }
}

/// 1-sigma sensor noise levels fed by the GUI sliders / config.
#[derive(Debug, Clone)]
pub struct SensorNoise {
    pub gyro_bias: f64,  // rad/s, constant per-axis bias
    pub gyro_noise: f64, // rad/s
    pub mag_noise: f64,  // Tesla
    pub sun_noise: f64,  // rad (coarse attitude contribution)
    pub st_noise: f64,   // rad (fine attitude contribution)
}

impl Default for SensorNoise {
    fn default() -> Self {
        Self {
            gyro_bias: 1e-5,
            gyro_noise: 1e-6,
            mag_noise: 1e-8,
            sun_noise: 1e-3,
            st_noise: 1e-4,
        }
    }
}

/// Quaternion of the nadir-pointing LVLH target frame: body +z toward the Earth
/// center, +y along the negative orbit normal, +x completing the triad
/// (roughly along-track). Body→ECI: rotate_vector_q(q_t, v_body) = v_eci.
pub fn nadir_target_quaternion(r: [f64; 3], v: [f64; 3]) -> [f64; 4] {
    let z_b = normalize([-r[0], -r[1], -r[2]]);
    let h = cross(r, v);
    let y_b = normalize([-h[0], -h[1], -h[2]]);
    let x_b = cross(y_b, z_b);
    // Body→ECI DCM: columns are the desired body axes expressed in ECI.
    let a = [
        [x_b[0], y_b[0], z_b[0]],
        [x_b[1], y_b[1], z_b[1]],
        [x_b[2], y_b[2], z_b[2]],
    ];
    quat_from_matrix(a)
}

/// Nominal body rate for nadir tracking: the LVLH frame rotates at the orbit
/// rate about the orbit normal, which is the body -y axis.
pub fn nadir_body_rate(r: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    let r_len = norm(r);
    if r_len <= 0.0 {
        return [0.0; 3];
    }
    let n = norm(cross(r, v)) / (r_len * r_len);
    [0.0, -n, 0.0]
}

/// Extract the scalar-first quaternion q with R(q) = m (Shepperd's method).
fn quat_from_matrix(m: [[f64; 3]; 3]) -> [f64; 4] {
    let trace = m[0][0] + m[1][1] + m[2][2];
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            0.25 * s,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        ]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        [
            (m[2][1] - m[1][2]) / s,
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        ]
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        [
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
        ]
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        [
            (m[1][0] - m[0][1]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
        ]
    };
    normalize_q(q)
}

/// Rotation angle (rad) between the actual attitude and the target attitude.
pub fn pointing_error_rad(q: [f64; 4], q_target: [f64; 4]) -> f64 {
    let dq = quat_mult(q, quat_conj(q_target));
    2.0 * dq[0].abs().min(1.0).acos()
}

/// Gaussian pointing-loss factor in [0, 1]: exp(-(theta/theta_ref)^2).
pub fn pointing_loss_factor(err_rad: f64, ref_rad: f64) -> f64 {
    if ref_rad <= 0.0 {
        return 1.0;
    }
    let x = err_rad / ref_rad;
    (-x * x).exp()
}

/// Simulated attitude measurement: the true quaternion perturbed by a small
/// random rotation whose magnitude blends the star tracker (fine) and sun
/// sensor (coarse, down-weighted) noise levels.
fn measure_attitude(q: [f64; 4], noise: &SensorNoise, rng: &mut Lcg) -> [f64; 4] {
    let sigma = (noise.st_noise * noise.st_noise
        + 0.01 * noise.sun_noise * noise.sun_noise)
        .sqrt();
    if sigma <= 0.0 {
        return q;
    }
    let half = [
        0.5 * sigma * rng.next_gaussian(),
        0.5 * sigma * rng.next_gaussian(),
        0.5 * sigma * rng.next_gaussian(),
    ];
    let dq = normalize_q([1.0, half[0], half[1], half[2]]);
    // Right-multiplication: small perturbation applied in the body frame.
    normalize_q(quat_mult(q, dq))
}

/// PD quaternion-feedback controller with reaction wheel saturation and
/// magnetorquer momentum dumping. Returns (rw_torque_cmd, mtq_dipole_cmd),
/// both in the body frame, ready for `step_attitude`.
pub fn compute_adcs_command(
    sat: &Satellite,
    q_target: [f64; 4],
    b_body_true: [f64; 3],
    gains: &AdcsGains,
    noise: &SensorNoise,
    rng: &mut Lcg,
) -> ([f64; 3], [f64; 3]) {
    // --- Attitude determination from noisy sensors ---
    let q_meas = measure_attitude(sat.q, noise, rng);
    let omega_meas = [
        sat.omega[0] + noise.gyro_bias + noise.gyro_noise * rng.next_gaussian(),
        sat.omega[1] + noise.gyro_bias + noise.gyro_noise * rng.next_gaussian(),
        sat.omega[2] + noise.gyro_bias + noise.gyro_noise * rng.next_gaussian(),
    ];
    let b_meas = [
        b_body_true[0] + noise.mag_noise * rng.next_gaussian(),
        b_body_true[1] + noise.mag_noise * rng.next_gaussian(),
        b_body_true[2] + noise.mag_noise * rng.next_gaussian(),
    ];

    // --- Attitude error: rotation from target frame to body frame, expressed
    // in body coordinates (q = q_target ⊗ dq under the body→ECI convention) ---
    let dq = quat_mult(quat_conj(q_target), q_meas);
    // Shortest-path unwinding: steer toward the closer of q and -q.
    let sign = if dq[0] >= 0.0 { 1.0 } else { -1.0 };

    // Desired body rate for nadir tracking: the LVLH frame rotates at the orbit
    // rate about its -y axis (y_b = -orbit normal), so omega_d ≈ [0, -n, 0].
    let r_len = norm(sat.r);
    let n_orbit = if r_len > 0.0 { norm(cross(sat.r, sat.v)) / (r_len * r_len) } else { 0.0 };
    let omega_err = [
        omega_meas[0],
        omega_meas[1] + n_orbit,
        omega_meas[2],
    ];

    // --- Momentum dumping: m = +k_dump * (h_rw x B) / |B|^2 ---
    // With attitude held by the wheels, h_rw_dot = tau_mtq = m x B = -k * h_perp,
    // which bleeds the wheel momentum. Gated on a threshold so the MTQ torque
    // does not perturb fine pointing in steady state.
    let b_sq = crate::math::dot(b_meas, b_meas);
    let mut mtq_dipole = [0.0; 3];
    if b_sq > 1e-18 && norm(sat.h_rw) > gains.h_dump_threshold {
        let h_x_b = cross(sat.h_rw, b_meas);
        for i in 0..3 {
            let m = gains.k_dump * h_x_b[i] / b_sq;
            mtq_dipole[i] = m.clamp(-gains.mtq_dipole_max, gains.mtq_dipole_max);
        }
    }

    // --- PD law, inertia-normalized, with per-axis wheel saturation ---
    // Feed-forward: the wheels counteract the predicted magnetorquer torque so
    // momentum dumping does not disturb the pointing (the compensation itself
    // transfers the momentum out of the wheels, which is the point of dumping).
    let tau_mtq_pred = cross(mtq_dipole, b_meas);
    let mut rw_torque = [0.0; 3];
    for i in 0..3 {
        let tau = sat.inertia[i]
            * (-gains.kp * sign * dq[i + 1] - gains.kd * omega_err[i])
            - tau_mtq_pred[i];
        rw_torque[i] = tau.clamp(-gains.rw_torque_max, gains.rw_torque_max);
    }

    (rw_torque, mtq_dipole)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::rotate_vector_q;
    use crate::models::{OrbitType, Satellite, SimEnvironment};
    use crate::physics::{step_attitude, step_orbit};

    fn test_env() -> SimEnvironment {
        SimEnvironment {
            mu: 3.986004418e14,
            r_earth: 6378137.0,
            j2: 0.0,
            rho0_500km: 0.0,
            h0_km: 500.0,
            scale_height_km: 70.0,
            p_srp: 0.0,
        }
    }

    fn leo_sat() -> Satellite {
        let r = [6928137.0, 0.0, 0.0]; // 550 km circular
        let v_mag = (3.986004418e14_f64 / 6928137.0).sqrt();
        Satellite {
            id: "TEST".to_string(),
            orbit_type: OrbitType::LEO,
            r,
            v: [0.0, v_mag, 0.0],
            q: [1.0, 0.0, 0.0, 0.0],
            omega: [0.0, 0.0, 0.0],
            mass: 20.0,
            area: 0.1,
            cd: 0.0,
            cr: 0.0,
            inertia: [0.4, 0.4, 0.5],
            h_rw: [0.0, 0.0, 0.0],
            is_custom: false,
            custom_color: None,
        }
    }

    fn zero_noise() -> SensorNoise {
        SensorNoise { gyro_bias: 0.0, gyro_noise: 0.0, mag_noise: 0.0, sun_noise: 0.0, st_noise: 0.0 }
    }

    #[test]
    fn nadir_target_points_z_to_earth() {
        let sat = leo_sat();
        let q_t = nadir_target_quaternion(sat.r, sat.v);
        // Body→ECI convention: body +z expressed in ECI must be -r_hat (nadir).
        let nadir_eci = normalize([-sat.r[0], -sat.r[1], -sat.r[2]]);
        let z_in_eci = rotate_vector_q(q_t, [0.0, 0.0, 1.0]);
        assert!((z_in_eci[0] - nadir_eci[0]).abs() < 1e-9);
        assert!((z_in_eci[1] - nadir_eci[1]).abs() < 1e-9);
        assert!((z_in_eci[2] - nadir_eci[2]).abs() < 1e-9);
    }

    #[test]
    fn pointing_error_zero_at_target_and_known_elsewhere() {
        let sat = leo_sat();
        let q_t = nadir_target_quaternion(sat.r, sat.v);
        assert!(pointing_error_rad(q_t, q_t) < 1e-9);

        // Rotate the target by 10° about body x: error must read 10°.
        let half = 5.0_f64.to_radians();
        let dq = [half.cos(), half.sin(), 0.0, 0.0];
        let q_off = quat_mult(dq, q_t);
        let err = pointing_error_rad(q_off, q_t);
        assert!((err - 10.0_f64.to_radians()).abs() < 1e-9, "err = {err}");
    }

    #[test]
    fn pointing_loss_factor_bounds() {
        assert!((pointing_loss_factor(0.0, 5e-3) - 1.0).abs() < 1e-12);
        let l = pointing_loss_factor(5e-3, 5e-3);
        assert!((l - (-1.0_f64).exp()).abs() < 1e-12);
        assert!(pointing_loss_factor(0.1, 5e-3) < 1e-9);
    }

    #[test]
    fn controller_stabilizes_tumbling_satellite() {
        let mut sat = leo_sat();
        // Start tumbling, attitude far from target.
        sat.q = normalize_q([0.5, 0.5, -0.5, 0.5]);
        sat.omega = [0.05, -0.05, 0.05];

        let env = test_env();
        let gains = AdcsGains::default();
        let noise = zero_noise();
        let mut rng = Lcg::new(7);
        let b_eci = [1e-5, 2e-5, -3e-5];
        let dt = 0.5;

        let mut max_cmd: f64 = 0.0;
        for _ in 0..6000 {
            let q_t = nadir_target_quaternion(sat.r, sat.v);
            let b_body = rotate_vector_q(sat.q, b_eci);
            let (rw, mtq) = compute_adcs_command(&sat, q_t, b_body, &gains, &noise, &mut rng);
            for t in rw {
                max_cmd = max_cmd.max(t.abs());
            }
            step_orbit(&mut sat, dt, &env, [1.0, 0.0, 0.0]);
            step_attitude(&mut sat, dt, b_eci, rw, mtq, [0.0; 3]);
        }

        let q_t = nadir_target_quaternion(sat.r, sat.v);
        let err = pointing_error_rad(sat.q, q_t);
        assert!(err < 0.5_f64.to_radians(), "pointing error = {}°", err.to_degrees());
        assert!(norm(sat.omega) < 0.01, "residual rate = {} rad/s", norm(sat.omega));
        assert!(max_cmd <= gains.rw_torque_max + 1e-12, "torque exceeded saturation");
    }

    /// End-to-end: a torque disturbance injected on a stabilized satellite
    /// (exactly what the GUI's "Inject Torque" does) must crash its laser link
    /// capacity through the pointing-loss coupling, and the controller must
    /// then recover both attitude and capacity.
    #[test]
    fn injected_disturbance_degrades_link_then_recovers() {
        use crate::network::{route_network, GroundNode, LinkMemory, RouteNode, RouteParams};

        let env = test_env();
        let mut sat = leo_sat();
        sat.q = nadir_target_quaternion(sat.r, sat.v);
        sat.omega = nadir_body_rate(sat.r, sat.v);

        let gains = AdcsGains::default();
        let noise = zero_noise();
        let mut rng = Lcg::new(7);
        let b_eci = [1e-5, 2e-5, -3e-5];
        let dt = 1.0;
        let ref_rad = 5.0e-3; // pointing_ref_mrad = 5.0

        // Capacity at a FIXED overhead geometry with the satellite's real attitude
        // error, so the test isolates the pointing→capacity coupling from the
        // satellite drifting out of the station's sky during the recovery.
        let capacity_of = |sat: &Satellite| -> f64 {
            let err = pointing_error_rad(sat.q, nadir_target_quaternion(sat.r, sat.v));
            let nodes = vec![RouteNode {
                id: "LEO".to_string(),
                is_relay: false,
                max_cap: 100.0,
                sgl_ref_dist: 1000.0,
                isl_ref_dist: 1000.0,
                r: [6928137.0, 0.0, 0.0],
                point_factor: pointing_loss_factor(err, ref_rad),
            }];
            // Station straight below, no weather attenuation.
            let gs = vec![GroundNode { id: "GS".to_string(), r: [6378137.0, 0.0, 0.0], k_value: 0.0, capacity: f64::INFINITY, min_elev_rad: 0.0 }];
            let params = RouteParams { prioritize_relay: false, hysteresis: 1.3, acquisition_time_s: 0.0, min_dwell_s: 0.0 };
            let mut memory = LinkMemory::new();
            route_network(&nodes, &gs, &params, &mut memory, 0.0, &env).sat_ground_rate[0]
        };

        let step = |sat: &mut Satellite, tau_ext: [f64; 3], rng: &mut Lcg| {
            let q_t = nadir_target_quaternion(sat.r, sat.v);
            let b_body = rotate_vector_q(sat.q, b_eci);
            let (rw, mtq) = compute_adcs_command(sat, q_t, b_body, &gains, &noise, rng);
            step_orbit(sat, dt, &env, [1.0, 0.0, 0.0]);
            step_attitude(sat, dt, b_eci, rw, mtq, tau_ext);
        };

        let cap_before = capacity_of(&sat);
        assert!(cap_before > 1.0, "no baseline capacity: {cap_before}");

        // Inject 3 Nm on the x axis for one step, as the GUI does.
        step(&mut sat, [3.0, 0.0, 0.0], &mut rng);
        // Let the tumble develop for a few steps before sampling.
        for _ in 0..5 {
            step(&mut sat, [0.0; 3], &mut rng);
        }
        let cap_disturbed = capacity_of(&sat);
        assert!(
            cap_disturbed < cap_before * 0.1,
            "capacity did not collapse: {cap_before} -> {cap_disturbed}"
        );

        // The controller detumbles and re-points; capacity must come back.
        for _ in 0..600 {
            step(&mut sat, [0.0; 3], &mut rng);
        }
        let cap_recovered = capacity_of(&sat);
        assert!(
            cap_recovered > cap_before * 0.9,
            "capacity did not recover: {cap_before} -> {cap_recovered}"
        );
    }

    #[test]
    fn momentum_dumping_reduces_wheel_momentum() {
        let mut sat = leo_sat();
        sat.q = nadir_target_quaternion(sat.r, sat.v);
        sat.h_rw = [0.2, -0.1, 0.15];
        let h0 = norm(sat.h_rw);

        let env = test_env();
        let gains = AdcsGains { k_dump: 5e-3, ..AdcsGains::default() };
        let noise = zero_noise();
        let mut rng = Lcg::new(7);
        let b_eci = [1e-5, 2e-5, -3e-5];
        let dt = 1.0;

        for _ in 0..5000 {
            let q_t = nadir_target_quaternion(sat.r, sat.v);
            let b_body = rotate_vector_q(sat.q, b_eci);
            let (rw, mtq) = compute_adcs_command(&sat, q_t, b_body, &gains, &noise, &mut rng);
            step_orbit(&mut sat, dt, &env, [1.0, 0.0, 0.0]);
            step_attitude(&mut sat, dt, b_eci, rw, mtq, [0.0; 3]);
        }

        let h1 = norm(sat.h_rw);
        assert!(h1 < h0 * 0.8, "wheel momentum not reduced: {h0} -> {h1}");
    }
}
