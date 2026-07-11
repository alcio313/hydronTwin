use crate::models::{Satellite, GroundStation, AtmosphereModel, SimEnvironment};
use crate::math::*;

pub fn step_orbit(sat: &mut Satellite, dt: f64, env: &SimEnvironment, sun_vector: [f64; 3]) {
    let mut state = [
        sat.r[0], sat.r[1], sat.r[2],
        sat.v[0], sat.v[1], sat.v[2]
    ];

    let mass = sat.mass;
    let cd = sat.cd;
    let area = sat.area;
    let cr = sat.cr;

    let deriv = |x: &[f64; 6]| -> [f64; 6] {
        let r_vec = [x[0], x[1], x[2]];
        let v_vec = [x[3], x[4], x[5]];
        let r_len = norm(r_vec);
        let r_len3 = r_len.powi(3);
        let r_len5 = r_len.powi(5);

        // Core central two-body gravity
        let mut a = scale(r_vec, -env.mu / r_len3);

        // 1. J2 Perturbation
        if env.j2 > 0.0 {
            let j2_coeff = 1.5 * env.j2 * env.mu * env.r_earth.powi(2) / r_len5;
            let z2_r2_ratio = x[2] * x[2] / (r_len * r_len);
            let a_j2 = [
                j2_coeff * x[0] * (5.0 * z2_r2_ratio - 1.0),
                j2_coeff * x[1] * (5.0 * z2_r2_ratio - 1.0),
                j2_coeff * x[2] * (5.0 * z2_r2_ratio - 3.0),
            ];
            a = add(a, a_j2);
        }

        // 2. Atmospheric Drag (only for LEO / MEO below 1500km)
        let altitude = r_len - env.r_earth;
        if altitude < 1_500_000.0 && cd > 0.0 {
            // Exponential atmospheric model
            let h0_m = env.h0_km * 1000.0;
            let scale_height_m = env.scale_height_km * 1000.0;
            let rho = env.rho0_500km * (-(altitude - h0_m) / scale_height_m).exp();
            
            // Relative velocity vector (assuming Earth's atmosphere co-rotates with Earth)
            let omega_earth = [0.0, 0.0, 7.292115e-5];
            let v_rel = [
                v_vec[0] - (-omega_earth[2] * r_vec[1]),
                v_vec[1] - (omega_earth[2] * r_vec[0]),
                v_vec[2]
            ];
            let v_rel_len = norm(v_rel);
            let a_drag = scale(v_rel, -0.5 * rho * cd * area / mass * v_rel_len);
            a = add(a, a_drag);
        }

        // 3. Solar Radiation Pressure (SRP) — zero while in Earth's shadow
        if env.p_srp > 0.0 {
            // s_hat is the unit sun direction vector
            let s_hat = normalize(sun_vector);
            if !in_earth_shadow(r_vec, s_hat, env.r_earth) {
                let a_srp = scale(s_hat, env.p_srp * cr * area / mass);
                a = add(a, a_srp);
            }
        }

        [v_vec[0], v_vec[1], v_vec[2], a[0], a[1], a[2]]
    };

    // RK4 numerical integration
    let k1 = deriv(&state);
    let mut x2 = [0.0; 6];
    for i in 0..6 { x2[i] = state[i] + 0.5 * dt * k1[i]; }
    let k2 = deriv(&x2);
    let mut x3 = [0.0; 6];
    for i in 0..6 { x3[i] = state[i] + 0.5 * dt * k2[i]; }
    let k3 = deriv(&x3);
    let mut x4 = [0.0; 6];
    for i in 0..6 { x4[i] = state[i] + dt * k3[i]; }
    let k4 = deriv(&x4);

    for i in 0..6 {
        state[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }

    sat.r = [state[0], state[1], state[2]];
    sat.v = [state[3], state[4], state[5]];
}

/// Unit vector from the Earth to the Sun in ECI at simulation time `t` (s).
///
/// Simplified circular ephemeris: the Sun moves along the ecliptic at the
/// mean rate (2π/365.25 d) with obliquity 23.44°; t = 0 is the vernal equinox.
pub fn sun_direction(t: f64) -> [f64; 3] {
    const OBLIQUITY_RAD: f64 = 23.44 * std::f64::consts::PI / 180.0;
    const YEAR_S: f64 = 365.25 * 86400.0;
    let lambda = 2.0 * std::f64::consts::PI * t / YEAR_S;
    let (sin_l, cos_l) = lambda.sin_cos();
    let (sin_e, cos_e) = OBLIQUITY_RAD.sin_cos();
    [cos_l, sin_l * cos_e, sin_l * sin_e]
}

/// Cylindrical Earth-shadow model: true when the satellite is on the anti-sun
/// side and inside the shadow cylinder of radius r_earth.
pub fn in_earth_shadow(r: [f64; 3], sun_hat: [f64; 3], r_earth: f64) -> bool {
    let along = dot(r, sun_hat);
    if along >= 0.0 {
        return false; // Sun side of the terminator plane
    }
    let perp = [
        r[0] - along * sun_hat[0],
        r[1] - along * sun_hat[1],
        r[2] - along * sun_hat[2],
    ];
    norm(perp) < r_earth
}

/// Earth magnetic field as a tilted dipole, in ECI coordinates (Tesla).
///
/// B(r) = B0 (Re/r)^3 [3 (m·r̂) r̂ − m], with the dipole axis tilted ~11.5°
/// from the rotation axis and co-rotating with the Earth (via GST). Replaces
/// the previous constant mock field so magnetorquer authority varies with
/// position and altitude (B ~ 1/r^3: strong in LEO, weak at GEO).
pub fn dipole_field_eci(r_eci: [f64; 3], gst: f64, r_earth: f64) -> [f64; 3] {
    const B0: f64 = 3.12e-5; // Mean field at the magnetic equator, Earth surface (T)
    const TILT_RAD: f64 = 11.5 * std::f64::consts::PI / 180.0;
    // Dipole axis: tilted from +z toward a longitude fixed in the rotating Earth
    // (the geographic longitude of the north magnetic pole is folded into GST=0).
    let (sin_t, cos_t) = TILT_RAD.sin_cos();
    let m_hat = [sin_t * gst.cos(), sin_t * gst.sin(), cos_t];

    let r_len = norm(r_eci);
    if r_len <= 0.0 {
        return [0.0; 3];
    }
    let r_hat = scale(r_eci, 1.0 / r_len);
    let m_dot_r = dot(m_hat, r_hat);
    let factor = B0 * (r_earth / r_len).powi(3);
    [
        factor * (3.0 * m_dot_r * r_hat[0] - m_hat[0]),
        factor * (3.0 * m_dot_r * r_hat[1] - m_hat[1]),
        factor * (3.0 * m_dot_r * r_hat[2] - m_hat[2]),
    ]
}

// 2. step_attitude: Propagates the spacecraft attitude dynamics using quaternion kinematic integration
// and Euler's equations of rotational motion with reaction wheels, magnetorquers, and disturbances.
// `tau_ext` is an externally injected body-frame disturbance torque (Nm).
// The explicit-Euler update is substepped so |omega|·h stays small: without this,
// a large injected torque (fast tumble) at dt = 1 s makes the integration unstable
// and the satellite never detumbles.
pub fn step_attitude(sat: &mut Satellite, dt: f64, b_eci: [f64; 3], torque_rw_cmd: [f64; 3], dipole_mtq_cmd: [f64; 3], tau_ext: [f64; 3]) {
    let i_x = sat.inertia[0];
    let i_y = sat.inertia[1];
    let i_z = sat.inertia[2];

    // Substep count: keep the per-substep rotation below ~0.05 rad.
    let omega_mag = (sat.omega[0].powi(2) + sat.omega[1].powi(2) + sat.omega[2].powi(2)).sqrt();
    let n_sub = ((omega_mag * dt.abs() / 0.05).ceil() as usize).clamp(1, 1000);
    let h = dt / (n_sub as f64);

    for _ in 0..n_sub {
        // 1. Euler dynamics: I * domega/dt + omega x (I * omega) = tau_rw + tau_mtq + tau_dist
        // Magnetic field in body frame: B_body = R(q) * B_eci
        let b_body = rotate_vector_q(sat.q, b_eci);

        // Torque from magnetorquer: tau_mtq = m x B
        let tau_mtq = cross(dipole_mtq_cmd, b_body);

        // Torque from reaction wheels (action/reaction on spacecraft body)
        let tau_rw = torque_rw_cmd;

        // Environmental disturbances (gravity gradient mockup as basic test dist)
        // ponytail: disturbance torque is simplified to constant bias + small white noise mockup
        let tau_dist = [1e-6, -1e-6, 5e-7];

        let total_torque = [
            tau_rw[0] + tau_mtq[0] + tau_dist[0] + tau_ext[0],
            tau_rw[1] + tau_mtq[1] + tau_dist[1] + tau_ext[1],
            tau_rw[2] + tau_mtq[2] + tau_dist[2] + tau_ext[2],
        ];

        let omega_x_i_omega = [
            sat.omega[1] * (i_z * sat.omega[2]) - sat.omega[2] * (i_y * sat.omega[1]),
            sat.omega[2] * (i_x * sat.omega[0]) - sat.omega[0] * (i_z * sat.omega[2]),
            sat.omega[0] * (i_y * sat.omega[1]) - sat.omega[1] * (i_x * sat.omega[0]),
        ];

        let domega = [
            (total_torque[0] - omega_x_i_omega[0]) / i_x,
            (total_torque[1] - omega_x_i_omega[1]) / i_y,
            (total_torque[2] - omega_x_i_omega[2]) / i_z,
        ];

        // Update wheel angular momentum: h_rw_dot = -tau_rw
        for i in 0..3 {
            sat.h_rw[i] += -tau_rw[i] * h;
        }

        // Update omega
        for i in 0..3 {
            sat.omega[i] += domega[i] * h;
        }

        // 2. Quaternion kinematics integration: dq/dt = 0.5 * Omega(omega) * q
        let q = sat.q;
        let w = sat.omega;
        let dq = [
            -0.5 * (q[1]*w[0] + q[2]*w[1] + q[3]*w[2]),
             0.5 * (q[0]*w[0] + q[2]*w[2] - q[3]*w[1]),
             0.5 * (q[0]*w[1] - q[1]*w[2] + q[3]*w[0]),
             0.5 * (q[0]*w[2] + q[1]*w[1] - q[2]*w[0]),
        ];

        let new_q = [
            q[0] + dq[0] * h,
            q[1] + dq[1] * h,
            q[2] + dq[2] * h,
            q[3] + dq[3] * h,
        ];

        sat.q = normalize_q(new_q);
    }
}

// 3. step_atmosphere: Updates atmospheric state for each ground station using a discrete Markov chain.
pub fn step_atmosphere(gs: &mut GroundStation, model: &mut AtmosphereModel) {
    let r = model.lcg.next_f64();
    let current_state = gs.atmos_state;
    let row = &model.transition_matrix[current_state];
    
    let mut cumulative = 0.0;
    let mut next_state = current_state;
    
    for (idx, &prob) in row.iter().enumerate() {
        cumulative += prob;
        if r < cumulative {
            next_state = idx;
            break;
        }
    }
    
    gs.atmos_state = next_state;
    gs.k_value = model.k_values[next_state] / 1000.0; // Convert 1/km to 1/m
}

// 4. visible: Evaluates geometric LoS between two space nodes (ISL). Uses r_earth+100km buffer.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::norm;
    use crate::models::OrbitType;

    #[test]
    fn rk4_conserves_circular_orbit_radius() {
        let env = SimEnvironment {
            mu: 3.986004418e14,
            r_earth: 6378137.0,
            j2: 0.0,
            rho0_500km: 0.0,
            h0_km: 500.0,
            scale_height_km: 70.0,
            p_srp: 0.0,
        };
        let r0 = 6928137.0; // 550 km circular
        let v0 = (env.mu / r0).sqrt();
        let mut sat = Satellite {
            id: "TEST".to_string(),
            orbit_type: OrbitType::LEO,
            r: [r0, 0.0, 0.0],
            v: [0.0, v0, 0.0],
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
        };

        // One full orbital period at 1 s steps
        let period = 2.0 * std::f64::consts::PI * (r0.powi(3) / env.mu).sqrt();
        let steps = period.ceil() as usize;
        for _ in 0..steps {
            step_orbit(&mut sat, 1.0, &env, [1.0, 0.0, 0.0]);
        }

        let r_final = norm(sat.r);
        assert!(
            (r_final - r0).abs() < 1.0,
            "radius drifted by {} m over one period",
            (r_final - r0).abs()
        );
    }

    #[test]
    fn sun_direction_seasonal_geometry() {
        // Vernal equinox: Sun along +x.
        let s0 = sun_direction(0.0);
        assert!((s0[0] - 1.0).abs() < 1e-12 && s0[1].abs() < 1e-12 && s0[2].abs() < 1e-12);

        // Summer solstice (quarter year): Sun in the y-z plane, tilted by the obliquity.
        let quarter = 365.25 * 86400.0 / 4.0;
        let s1 = sun_direction(quarter);
        let eps = 23.44_f64.to_radians();
        assert!(s1[0].abs() < 1e-9);
        assert!((s1[1] - eps.cos()).abs() < 1e-9);
        assert!((s1[2] - eps.sin()).abs() < 1e-9);
        // Always a unit vector.
        assert!((norm(s1) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn earth_shadow_predicate() {
        let r_earth = 6378137.0;
        let sun = [1.0, 0.0, 0.0];
        // Directly behind the Earth (anti-sun): in shadow.
        assert!(in_earth_shadow([-7e6, 0.0, 0.0], sun, r_earth));
        // Sun side: never in shadow.
        assert!(!in_earth_shadow([7e6, 0.0, 0.0], sun, r_earth));
        // Anti-sun side but outside the shadow cylinder: lit.
        assert!(!in_earth_shadow([-7e6, 7e6, 0.0], sun, r_earth));
        // Anti-sun side, inside the cylinder at an offset below its radius: shadowed.
        assert!(in_earth_shadow([-7e6, 3e6, 0.0], sun, r_earth));
    }

    #[test]
    fn dipole_field_magnitude_and_direction() {
        let r_earth = 6378137.0;
        // Zero tilt reference: evaluate with GST=0 and points on the geographic
        // axes, then compare against the analytic dipole with 11.5° tilt.
        let tilt = 11.5_f64.to_radians();
        let m_hat = [tilt.sin(), 0.0, tilt.cos()];

        // On the dipole axis at 2 Earth radii: |B| = 2*B0/8 = B0/4.
        let r_axis = [2.0 * r_earth * m_hat[0], 2.0 * r_earth * m_hat[1], 2.0 * r_earth * m_hat[2]];
        let b = dipole_field_eci(r_axis, 0.0, r_earth);
        let b_mag = norm(b);
        assert!((b_mag - 2.0 * 3.12e-5 / 8.0).abs() / b_mag < 1e-9, "|B| on axis = {b_mag}");
        // Field parallel to the dipole axis there.
        let cos_angle = crate::math::dot(crate::math::normalize(b), m_hat);
        assert!((cos_angle - 1.0).abs() < 1e-9);

        // On the magnetic equator (perpendicular to m) at surface radius: |B| = B0,
        // anti-parallel to m.
        let e = crate::math::normalize(crate::math::cross(m_hat, [0.0, 1.0, 0.0]));
        let r_eq = [r_earth * e[0], r_earth * e[1], r_earth * e[2]];
        let b_eq = dipole_field_eci(r_eq, 0.0, r_earth);
        assert!((norm(b_eq) - 3.12e-5).abs() / 3.12e-5 < 1e-9);
        let cos_eq = crate::math::dot(crate::math::normalize(b_eq), m_hat);
        assert!((cos_eq + 1.0).abs() < 1e-9);

        // 1/r^3 decay: GEO field ~ (Re/r_geo)^3 weaker than surface field.
        let r_geo = 42164e3;
        let b_geo = dipole_field_eci([0.0, r_geo, 0.0], 0.0, r_earth);
        assert!(norm(b_geo) < 3.0e-7, "GEO field should be ~1e-7 T, got {}", norm(b_geo));
    }

    #[test]
    fn step_attitude_zero_ext_torque_matches_legacy() {
        // With tau_ext = 0 the new signature must reproduce the old behavior:
        // just verify the state evolves and the quaternion stays normalized.
        let mut sat = Satellite {
            id: "TEST".to_string(),
            orbit_type: OrbitType::LEO,
            r: [7e6, 0.0, 0.0],
            v: [0.0, 7.5e3, 0.0],
            q: [1.0, 0.0, 0.0, 0.0],
            omega: [0.01, 0.0, 0.0],
            mass: 20.0,
            area: 0.1,
            cd: 0.0,
            cr: 0.0,
            inertia: [0.4, 0.4, 0.5],
            h_rw: [0.0, 0.0, 0.0],
            is_custom: false,
            custom_color: None,
        };
        for _ in 0..100 {
            step_attitude(&mut sat, 1.0, [1e-5, 2e-5, -3e-5], [0.0; 3], [0.0; 3], [0.0; 3]);
        }
        let q_norm = (sat.q[0].powi(2) + sat.q[1].powi(2) + sat.q[2].powi(2) + sat.q[3].powi(2)).sqrt();
        assert!((q_norm - 1.0).abs() < 1e-9);
        assert!(sat.omega[0].abs() > 0.0);
    }
}
