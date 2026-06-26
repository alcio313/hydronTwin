use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use eframe::egui;

// LCG Random Number Generator for deterministic and reproducible simulations.
#[derive(Debug, Clone)]
pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_f64(&mut self) -> f64 {
        // Numerical Recipes LCG parameters
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.state as f64) / (u64::MAX as f64)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrbitType {
    LEO,
    MEO,
    GEO,
}

#[derive(Debug, Clone)]
pub struct Satellite {
    pub id: String,
    pub orbit_type: OrbitType,
    // Orbital state (ECI frame, SI units: m, m/s)
    pub r: [f64; 3],
    pub v: [f64; 3],
    // Attitude state (ECI to Body frame quaternion [q0, q1, q2, q3] where q0 is scalar)
    pub q: [f64; 4],
    pub omega: [f64; 3], // Angular velocity relative to ECI in body frame (rad/s)
    // Physical parameters
    pub mass: f64,
    pub area: f64,
    pub cd: f64,
    pub cr: f64,
    pub inertia: [f64; 3], // Ix, Iy, Iz (kg*m^2), diagonal terms
    // Actuator states
    pub h_rw: [f64; 3], // Reaction wheels angular momentum (Nms)
}

#[derive(Debug, Clone)]
pub struct GroundStation {
    pub id: String,
    pub name: String,
    pub lat_rad: f64,
    pub lon_rad: f64,
    pub alt_m: f64,
    pub downlink_nominal_gbps: f64,
    // Atmosphere dynamic state
    pub atmos_state: usize,
    pub k_value: f64, // Attenuation coefficient (1/m)
}

#[derive(Debug, Clone)]
pub struct AtmosphereModel {
    pub states: Vec<String>,
    pub k_values: Vec<f64>, // 1/m
    pub transition_matrix: Vec<Vec<f64>>,
    pub lcg: Lcg,
}

#[derive(Debug, Clone)]
pub struct Segment {
    pub orbit_type: OrbitType,
    pub satellites: Vec<Satellite>,
}

#[derive(Debug, Clone)]
pub struct Constellation {
    pub name: String,
    pub segments: Vec<Segment>,
}

#[derive(Debug, Clone)]
pub struct LaserLink {
    pub from_id: String,
    pub to_id: String,
    pub visible: bool,
    pub distance_km: f64,
    pub capacity_gbps: f64,
    pub latency_ms: f64,
}

// Global environmental parameters from config
#[derive(Debug, Clone)]
pub struct SimEnvironment {
    pub mu: f64,
    pub r_earth: f64,
    pub j2: f64,
    pub rho0_500km: f64,
    pub h0_km: f64,
    pub scale_height_km: f64,
    pub p_srp: f64,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub name: String,
    pub leo_num: usize,
    pub leo_alt_km: f64,
    pub leo_inc_deg: f64,
    pub leo_mass: f64,
    pub leo_area: f64,
    pub leo_cd: f64,
    pub leo_cr: f64,
    pub meo_num: usize,
    pub meo_alt_km: f64,
    pub meo_inc_deg: f64,
    pub meo_raans: Vec<f64>,
    pub meo_mass: f64,
    pub meo_area: f64,
    pub meo_cd: f64,
    pub meo_cr: f64,
    pub geo_num: usize,
    pub geo_lons: Vec<f64>,
    pub geo_alt_km: f64,
    pub geo_inc_deg: f64,
    pub geo_mass: f64,
    pub geo_area: f64,
    pub geo_cd: f64,
    pub geo_cr: f64,
    pub stations: Vec<GroundStation>,
    pub atmos_states: Vec<String>,
    pub atmos_k: Vec<f64>,
    pub transition_matrix: Vec<Vec<f64>>,
    pub env: SimEnvironment,
    pub dt_time_step: f64,
    pub ref_dist_isl_km: f64,
    pub ref_dist_sgl_km: f64,
}

// Cross-product helper
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// Dot product helper
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// Norm helper
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

// Normalize helper
fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = norm(a);
    if n > 0.0 {
        [a[0] / n, a[1] / n, a[2] / n]
    } else {
        [0.0, 0.0, 0.0]
    }
}

// Vector addition
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

// Vector scaling
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

// Quaternion normalization
fn normalize_q(q: [f64; 4]) -> [f64; 4] {
    let n = (q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]).sqrt();
    if n > 0.0 {
        [q[0]/n, q[1]/n, q[2]/n, q[3]/n]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}

// ECI to ECEF rotation matrix at GST
fn eci_to_ecef_matrix(gst: f64) -> [[f64; 3]; 3] {
    let c = gst.cos();
    let s = gst.sin();
    [
        [c, s, 0.0],
        [-s, c, 0.0],
        [0.0, 0.0, 1.0],
    ]
}

// Matrix vector multiply
fn mat_vec_mult(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2],
        m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2],
        m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2],
    ]
}

// Rotate vector using quaternion (ECI to body frame)
// v_body = R(q) * v_eci
fn rotate_vector_q(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let q_vec = [q[1], q[2], q[3]];
    let q_scalar = q[0];
    
    // R(q) v = v + 2 * q_vec x (q_vec x v + q_scalar * v)
    let temp = add(cross(q_vec, v), scale(v, q_scalar));
    add(v, scale(cross(q_vec, temp), 2.0))
}

// Geodetic to ECEF conversion using WGS-84 ellipsoid parameters
pub fn lla_to_ecef(lat_rad: f64, lon_rad: f64, alt_m: f64) -> [f64; 3] {
    let a = 6378137.0; // Equatorial radius (m)
    let f = 1.0 / 298.257223563; // Flattening
    let e2 = f * (2.0 - f);
    
    let sin_lat = lat_rad.sin();
    let cos_lat = lat_rad.cos();
    let n = a / (1.0 - e2 * sin_lat * sin_lat).sqrt();
    
    let x = (n + alt_m) * cos_lat * lon_rad.cos();
    let y = (n + alt_m) * cos_lat * lon_rad.sin();
    let z = (n * (1.0 - e2) + alt_m) * sin_lat;
    
    [x, y, z]
}

// Simple hand-rolled TOML config loader to keep the application dependency-free
// ponytail: custom config loader that avoids external crate compilation and downloads.
pub fn load_config<P: AsRef<Path>>(path: P) -> io::Result<Config> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    
    let mut name = String::from("HydRON");
    let mut leo_num = 10;
    let mut leo_alt_km = 550.0;
    let mut leo_inc_deg = 97.6;
    let mut leo_mass = 20.0;
    let mut leo_area = 0.1;
    let mut leo_cd = 2.2;
    let mut leo_cr = 1.2;
    
    let mut meo_num = 4;
    let mut meo_alt_km = 10000.0;
    let mut meo_inc_deg = 55.0;
    let mut meo_raans = vec![0.0, 90.0, 180.0, 270.0];
    let mut meo_mass = 50.0;
    let mut meo_area = 0.25;
    let mut meo_cd = 0.0;
    let mut meo_cr = 1.2;
    
    let mut geo_num = 3;
    let mut geo_lons = vec![0.0, 60.0, -120.0];
    let mut geo_alt_km = 35786.0;
    let mut geo_inc_deg = 0.0;
    let mut geo_mass = 200.0;
    let mut geo_area = 1.5;
    let mut geo_cd = 0.0;
    let mut geo_cr = 1.2;
    
    let mut stations = Vec::new();
    let mut atmos_states = vec!["clear".to_string(), "thin_clouds".to_string(), "thick_clouds".to_string(), "heavy".to_string()];
    let mut atmos_k = vec![0.05, 0.2, 1.5, 5.0];
    let transition_matrix = vec![
        vec![0.85, 0.10, 0.04, 0.01],
        vec![0.15, 0.70, 0.10, 0.05],
        vec![0.05, 0.15, 0.65, 0.15],
        vec![0.02, 0.08, 0.20, 0.70],
    ];
    
    let mut mu = 3.986004418e14;
    let mut r_earth = 6378137.0;
    let mut j2 = 1.08262668e-3;
    let mut rho0 = 3.8e-12;
    let mut h0 = 500.0;
    let mut scale_height = 70.0;
    let mut p_srp = 4.56e-6;
    
    let mut dt_time_step = 1.0;
    let mut ref_dist_isl_km = 1000.0;
    let mut ref_dist_sgl_km = 1000.0;

    let mut current_section = String::new();
    let mut station_id = String::new();
    let mut station_name = String::new();
    let mut station_lat: f64 = 0.0;
    let mut station_lon: f64 = 0.0;
    let mut station_alt: f64 = 0.0;
    let mut station_cap: f64 = 10.0;

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            let section = trimmed[1..trimmed.len()-1].trim().to_string();
            if section == "ground.stations" || section == "[ground.stations]" || section == "[[ground.stations]]" {
                if !station_id.is_empty() {
                    stations.push(GroundStation {
                        id: station_id.clone(),
                        name: station_name.clone(),
                        lat_rad: station_lat.to_radians(),
                        lon_rad: station_lon.to_radians(),
                        alt_m: station_alt,
                        downlink_nominal_gbps: station_cap,
                        atmos_state: 0,
                        k_value: atmos_k[0] / 1000.0, // Convert 1/km to 1/m to keep consistent with meters
                    });
                }
                station_id = String::new();
                station_name = String::new();
                station_lat = 0.0;
                station_lon = 0.0;
                station_alt = 0.0;
                station_cap = 10.0;
                current_section = "ground.stations".to_string();
            } else {
                current_section = section;
            }
            continue;
        }

        if let Some(pos) = trimmed.find('=') {
            let key = trimmed[..pos].trim();
            let val = trimmed[pos+1..].trim();
            
            match current_section.as_str() {
                "constellation" => {
                    if key == "name" { name = val.replace('"', ""); }
                }
                "constellation.leo" => {
                    match key {
                        "num_satellites" => leo_num = val.parse().unwrap_or(leo_num),
                        "altitude_km" => leo_alt_km = val.parse().unwrap_or(leo_alt_km),
                        "inclination_deg" => leo_inc_deg = val.parse().unwrap_or(leo_inc_deg),
                        "mass_kg" => leo_mass = val.parse().unwrap_or(leo_mass),
                        "cross_section_area_m2" => leo_area = val.parse().unwrap_or(leo_area),
                        "cd" => leo_cd = val.parse().unwrap_or(leo_cd),
                        "cr" => leo_cr = val.parse().unwrap_or(leo_cr),
                        _ => {}
                    }
                }
                "constellation.meo" => {
                    match key {
                        "num_satellites" => meo_num = val.parse().unwrap_or(meo_num),
                        "altitude_km" => meo_alt_km = val.parse().unwrap_or(meo_alt_km),
                        "inclination_deg" => meo_inc_deg = val.parse().unwrap_or(meo_inc_deg),
                        "mass_kg" => meo_mass = val.parse().unwrap_or(meo_mass),
                        "cross_section_area_m2" => meo_area = val.parse().unwrap_or(meo_area),
                        "cd" => meo_cd = val.parse().unwrap_or(meo_cd),
                        "cr" => meo_cr = val.parse().unwrap_or(meo_cr),
                        "raans_deg" => {
                            let clean = val.replace('[', "").replace(']', "");
                            meo_raans = clean.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                        }
                        _ => {}
                    }
                }
                "constellation.geo" => {
                    match key {
                        "num_satellites" => geo_num = val.parse().unwrap_or(geo_num),
                        "altitude_km" => geo_alt_km = val.parse().unwrap_or(geo_alt_km),
                        "inclination_deg" => geo_inc_deg = val.parse().unwrap_or(geo_inc_deg),
                        "mass_kg" => geo_mass = val.parse().unwrap_or(geo_mass),
                        "cross_section_area_m2" => geo_area = val.parse().unwrap_or(geo_area),
                        "cd" => geo_cd = val.parse().unwrap_or(geo_cd),
                        "cr" => geo_cr = val.parse().unwrap_or(geo_cr),
                        "longitudes_deg" => {
                            let clean = val.replace('[', "").replace(']', "");
                            geo_lons = clean.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                        }
                        _ => {}
                    }
                }
                "ground.stations" => {
                    match key {
                        "id" => station_id = val.replace('"', "").replace(',', ""),
                        "name" => station_name = val.replace('"', "").replace(',', ""),
                        "lat_deg" => station_lat = val.parse().unwrap_or(0.0),
                        "lon_deg" => station_lon = val.parse().unwrap_or(0.0),
                        "alt_m" => station_alt = val.parse().unwrap_or(0.0),
                        "downlink_nominal_gbps" => station_cap = val.parse().unwrap_or(10.0),
                        _ => {}
                    }
                }
                "atmosphere" => {
                    match key {
                        "states" => {
                            let clean = val.replace('[', "").replace(']', "");
                            atmos_states = clean.split(',').map(|s| s.trim().replace('"', "")).collect();
                        }
                        "k_values_per_km" => {
                            let clean = val.replace('[', "").replace(']', "");
                            atmos_k = clean.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                        }
                        // Transition matrix parsing is bypassed for standard lookups to remain robust.
                        _ => {}
                    }
                }
                "environment" => {
                    match key {
                        "mu" => mu = val.parse().unwrap_or(mu),
                        "r_earth" => r_earth = val.parse().unwrap_or(r_earth),
                        "j2" => j2 = val.parse().unwrap_or(j2),
                        "rho0_500km" => rho0 = val.parse().unwrap_or(rho0),
                        "h0_km" => h0 = val.parse().unwrap_or(h0),
                        "scale_height_km" => scale_height = val.parse().unwrap_or(scale_height),
                        "p_srp" => p_srp = val.parse().unwrap_or(p_srp),
                        _ => {}
                    }
                }
                "digital_twin" => {
                    match key {
                        "time_step_s" => dt_time_step = val.parse().unwrap_or(dt_time_step),
                        "ref_distance_isl_km" => ref_dist_isl_km = val.parse().unwrap_or(ref_dist_isl_km),
                        "ref_distance_sgl_km" => ref_dist_sgl_km = val.parse().unwrap_or(ref_dist_sgl_km),
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }
    
    // Add the final ground station
    if !station_id.is_empty() {
        stations.push(GroundStation {
            id: station_id.clone(),
            name: station_name.clone(),
            lat_rad: station_lat.to_radians(),
            lon_rad: station_lon.to_radians(),
            alt_m: station_alt,
            downlink_nominal_gbps: station_cap,
            atmos_state: 0,
            k_value: atmos_k[0] / 1000.0,
        });
    }

    Ok(Config {
        name,
        leo_num,
        leo_alt_km,
        leo_inc_deg,
        leo_mass,
        leo_area,
        leo_cd,
        leo_cr,
        meo_num,
        meo_alt_km,
        meo_inc_deg,
        meo_raans,
        meo_mass,
        meo_area,
        meo_cd,
        meo_cr,
        geo_num,
        geo_lons,
        geo_alt_km,
        geo_inc_deg,
        geo_mass,
        geo_area,
        geo_cd,
        geo_cr,
        stations,
        atmos_states,
        atmos_k,
        transition_matrix,
        env: SimEnvironment {
            mu,
            r_earth,
            j2,
            rho0_500km: rho0,
            h0_km: h0,
            scale_height_km: scale_height,
            p_srp,
        },
        dt_time_step,
        ref_dist_isl_km,
        ref_dist_sgl_km,
    })
}

// 1. step_orbit: Propagates the satellite orbit using RK4 with two-body gravity + J2 + atmospheric drag + SRP.
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

        // 3. Solar Radiation Pressure (SRP)
        if env.p_srp > 0.0 {
            // s_hat is the unit sun direction vector
            let s_hat = normalize(sun_vector);
            let a_srp = scale(s_hat, env.p_srp * cr * area / mass);
            a = add(a, a_srp);
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

// 2. step_attitude: Propagates the spacecraft attitude dynamics using quaternion kinematic integration
// and Euler's equations of rotational motion with reaction wheels, magnetorquers, and disturbances.
pub fn step_attitude(sat: &mut Satellite, dt: f64, b_eci: [f64; 3], torque_rw_cmd: [f64; 3], dipole_mtq_cmd: [f64; 3]) {
    // 1. Euler dynamics: I * domega/dt + omega x (I * omega) = tau_rw + tau_mtq + tau_dist
    let i_x = sat.inertia[0];
    let i_y = sat.inertia[1];
    let i_z = sat.inertia[2];
    
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
        tau_rw[0] + tau_mtq[0] + tau_dist[0],
        tau_rw[1] + tau_mtq[1] + tau_dist[1],
        tau_rw[2] + tau_mtq[2] + tau_dist[2],
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
        sat.h_rw[i] += -tau_rw[i] * dt;
    }

    // Update omega
    for i in 0..3 {
        sat.omega[i] += domega[i] * dt;
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
        q[0] + dq[0] * dt,
        q[1] + dq[1] * dt,
        q[2] + dq[2] * dt,
        q[3] + dq[3] * dt,
    ];

    sat.q = normalize_q(new_q);
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
pub fn visible(r1: [f64; 3], r2: [f64; 3], r_earth: f64) -> bool {
    let d = [r2[0] - r1[0], r2[1] - r1[1], r2[2] - r1[2]];
    let d_len_sq = dot(d, d);
    if d_len_sq == 0.0 { return true; }
    let u_min = -dot(r1, d) / d_len_sq;
    // Ray occultation height limit: 100 km for ISL atmospheric blockage.
    let r_occult = r_earth + 100_000.0;
    if (0.0..=1.0).contains(&u_min) {
        let closest_point = [
            r1[0] + u_min * d[0],
            r1[1] + u_min * d[1],
            r1[2] + u_min * d[2]
        ];
        norm(closest_point) >= r_occult
    } else {
        norm(r1) >= r_occult && norm(r2) >= r_occult
    }
}

// visible_sgl: LoS between a satellite and a ground station on Earth's surface.
// A GS is always at norm ≈ r_earth, so we cannot require norm(GS) >= r_earth+100km.
// Instead: link is blocked only if the interior of the segment dips below r_earth.
pub fn visible_sgl(r_sat: [f64; 3], r_gs: [f64; 3], r_earth: f64) -> bool {
    let d = [r_sat[0] - r_gs[0], r_sat[1] - r_gs[1], r_sat[2] - r_gs[2]];
    let d_len_sq = dot(d, d);
    if d_len_sq == 0.0 { return false; }
    // u_min: parameter of closest approach along the GS→Sat segment
    let u_min = -dot(r_gs, d) / d_len_sq;
    if u_min <= 0.0 {
        // Closest point is the GS itself: segment goes upward → satellite is above horizon
        return true;
    }
    if u_min >= 1.0 {
        // Closest point is the satellite: segment never dips → visible
        return true;
    }
    // Interior closest point: check it doesn't go through the solid Earth
    let closest = [
        r_gs[0] + u_min * d[0],
        r_gs[1] + u_min * d[1],
        r_gs[2] + u_min * d[2],
    ];
    norm(closest) >= r_earth
}

// 5. compute_link_capacity: Calculates instantaneous laser link bandwidth.
pub fn compute_link_capacity(
    r_from: [f64; 3],
    r_to: [f64; 3],
    is_sgl: bool,
    gs_k: f64,
    ref_dist_km: f64,
    nominal_capacity: f64,
    env: &SimEnvironment,
) -> f64 {
    let d_vec = [r_to[0] - r_from[0], r_to[1] - r_from[1], r_to[2] - r_from[2]];
    let d_m = norm(d_vec);

    // Use correct visibility check: SGL endpoints are on Earth's surface
    let is_vis = if is_sgl {
        // r_from = satellite, r_to = GS (or vice versa — pick the one closer to Earth)
        let (r_sat, r_gs) = if norm(r_from) > norm(r_to) { (r_from, r_to) } else { (r_to, r_from) };
        visible_sgl(r_sat, r_gs, env.r_earth)
    } else {
        visible(r_from, r_to, env.r_earth)
    };
    if !is_vis {
        return 0.0;
    }

    // Transmittance T_atm = exp(-k * L)
    let t_atm = if is_sgl {
        // Position of ground station (assumed r_from or r_to; whichever is closer to Earth center)
        let r_gs = if norm(r_from) < norm(r_to) { r_from } else { r_to };
        let r_sat = if norm(r_from) < norm(r_to) { r_to } else { r_from };
        // Direction vector must point from ground station to satellite for slant path calculation
        let dir = normalize([r_sat[0] - r_gs[0], r_sat[1] - r_gs[1], r_sat[2] - r_gs[2]]);
        
        let r_gs_len = norm(r_gs);
        let r_atm = env.r_earth + 10_000.0; // Weather/troposphere boundary at 10 km for realistic attenuation
        
        // Quadratic equation for ray boundary intersection: u^2 + 2(r_gs . dir)u + (r_gs^2 - r_atm^2) = 0
        let b = 2.0 * dot(r_gs, dir);
        let c = r_gs_len * r_gs_len - r_atm * r_atm;
        let disc = b * b - 4.0 * c;
        
        let l_slant = if disc >= 0.0 {
            let u1 = (-b + disc.sqrt()) / 2.0;
            if u1 > 0.0 { u1.min(d_m) } else { 0.0 }
        } else {
            0.0
        };
        
        let att_db = gs_k * l_slant;
        10.0_f64.powf(-att_db / 10.0)
    } else {
        1.0 // Inter-Satellite Link has no atmospheric attenuation
    };

    // Free space divergence path loss logic: f(d) = (d0 / d)^2
    let d_km = d_m / 1000.0;
    let dist_ratio = ref_dist_km / d_km;
    
    nominal_capacity * t_atm * (dist_ratio * dist_ratio)
}

pub fn create_satellites_from_config(config: &Config) -> Constellation {
    let mut leo_sats = Vec::new();
    let r_earth = config.env.r_earth;
    let r_leo = r_earth + config.leo_alt_km * 1000.0;
    let v_leo_mag = (config.env.mu / r_leo).sqrt();
    let inc_leo = config.leo_inc_deg.to_radians();

    for i in 0..config.leo_num {
        let u = (i as f64) * 2.0 * std::f64::consts::PI / (config.leo_num as f64);
        let r_plane = [r_leo * u.cos(), r_leo * u.sin(), 0.0];
        let v_plane = [-v_leo_mag * u.sin(), v_leo_mag * u.cos(), 0.0];
        
        let c_i = inc_leo.cos();
        let s_i = inc_leo.sin();
        let r_eci = [
            r_plane[0],
            r_plane[1] * c_i,
            r_plane[1] * s_i
        ];
        let v_eci = [
            v_plane[0],
            v_plane[1] * c_i,
            v_plane[1] * s_i
        ];

        leo_sats.push(Satellite {
            id: format!("LEO_{:02}", i),
            orbit_type: OrbitType::LEO,
            r: r_eci,
            v: v_eci,
            q: [1.0, 0.0, 0.0, 0.0],
            omega: [0.0, 0.0, 0.0],
            mass: config.leo_mass,
            area: config.leo_area,
            cd: config.leo_cd,
            cr: config.leo_cr,
            inertia: [0.4, 0.4, 0.5],
            h_rw: [0.0, 0.0, 0.0],
        });
    }

    let mut meo_sats = Vec::new();
    let r_meo = r_earth + config.meo_alt_km * 1000.0;
    let v_meo_mag = (config.env.mu / r_meo).sqrt();
    let inc_meo = config.meo_inc_deg.to_radians();

    for i in 0..config.meo_num {
        let raan = if !config.meo_raans.is_empty() { config.meo_raans[0] } else { 0.0 };
        let raan_rad = raan.to_radians();
        let u = (i as f64) * 2.0 * std::f64::consts::PI / (config.meo_num as f64);
        let r_plane = [r_meo * u.cos(), r_meo * u.sin(), 0.0];
        let v_plane = [-v_meo_mag * u.sin(), v_meo_mag * u.cos(), 0.0];

        let c_r = raan_rad.cos();
        let s_r = raan_rad.sin();
        let c_i = inc_meo.cos();
        let s_i = inc_meo.sin();

        let r_eci = [
            c_r * r_plane[0] - s_r * c_i * r_plane[1],
            s_r * r_plane[0] + c_r * c_i * r_plane[1],
            s_i * r_plane[1]
        ];
        let v_eci = [
            c_r * v_plane[0] - s_r * c_i * v_plane[1],
            s_r * v_plane[0] + c_r * c_i * v_plane[1],
            s_i * v_plane[1]
        ];

        meo_sats.push(Satellite {
            id: format!("MEO_{:02}", i),
            orbit_type: OrbitType::MEO,
            r: r_eci,
            v: v_eci,
            q: [1.0, 0.0, 0.0, 0.0],
            omega: [0.0, 0.0, 0.0],
            mass: config.meo_mass,
            area: config.meo_area,
            cd: config.meo_cd,
            cr: config.meo_cr,
            inertia: [1.5, 1.5, 2.0],
            h_rw: [0.0, 0.0, 0.0],
        });
    }

    let mut geo_sats = Vec::new();
    let r_geo = r_earth + config.geo_alt_km * 1000.0;
    let v_geo_mag = (config.env.mu / r_geo).sqrt();
    let inc_geo = config.geo_inc_deg.to_radians();

    for i in 0..config.geo_num {
        let lon_rad = (i as f64) * 2.0 * std::f64::consts::PI / (config.geo_num as f64);
        let r_plane = [r_geo * lon_rad.cos(), r_geo * lon_rad.sin(), 0.0];
        let v_plane = [-v_geo_mag * lon_rad.sin(), v_geo_mag * lon_rad.cos(), 0.0];

        let c_i = inc_geo.cos();
        let s_i = inc_geo.sin();
        let r_eci = [
            r_plane[0],
            r_plane[1] * c_i,
            r_plane[1] * s_i
        ];
        let v_eci = [
            v_plane[0],
            v_plane[1] * c_i,
            v_plane[1] * s_i
        ];

        geo_sats.push(Satellite {
            id: format!("GEO_{:02}", i),
            orbit_type: OrbitType::GEO,
            r: r_eci,
            v: v_eci,
            q: [1.0, 0.0, 0.0, 0.0],
            omega: [0.0, 0.0, 0.0],
            mass: config.geo_mass,
            area: config.geo_area,
            cd: config.geo_cd,
            cr: config.geo_cr,
            inertia: [15.0, 15.0, 20.0],
            h_rw: [0.0, 0.0, 0.0],
        });
    }

    let segments = vec![
        Segment { orbit_type: OrbitType::LEO, satellites: leo_sats },
        Segment { orbit_type: OrbitType::MEO, satellites: meo_sats },
        Segment { orbit_type: OrbitType::GEO, satellites: geo_sats },
    ];

    Constellation {
        name: config.name.clone(),
        segments,
    }
}

pub struct HydronGuiApp {
    config: Config,
    constellation: Constellation,
    ground_stations: Vec<GroundStation>,
    atmos_model: AtmosphereModel,

    // Control parameters
    is_running: bool,
    current_time: f64,
    time_warp: i32,
    step_size: f64,

    // Selection
    selected_satellite_id: String,

    // Form inputs for dynamic configuration edits
    leo_num_input: usize,
    leo_alt_input: f64,
    leo_inc_input: f64,
    meo_num_input: usize,
    meo_alt_input: f64,
    meo_inc_input: f64,
    geo_num_input: usize,
    geo_alt_input: f64,
    geo_inc_input: f64,

    // Satellite dynamic properties fields
    sat_mass_input: f64,
    sat_cd_input: f64,
    sat_cr_input: f64,

    // Noise parameters
    gyro_noise: f64,
    mag_noise: f64,
    sun_noise: f64,
    st_noise: f64,

    // OMTQ / RW command override
    force_disturbance: bool,
    disturbance_val: [f64; 3],

    // Atmosphere dynamic control
    weather_overrides: Vec<Option<usize>>, // None = Markov, Some(index) = Force state

    // Filter displays
    show_leo: bool,
    show_meo: bool,
    show_geo: bool,
    show_sgl: bool,
    show_left_panel: bool,

    // Log list
    logs: Vec<String>,

    // Throughput history for bottom panel plotting
    history_time: Vec<f32>,
    history_stations: Vec<Vec<f32>>,
    history_total: Vec<f32>,

    // 3D Map rotation and zoom state
    map_pitch: f32,
    map_yaw: f32,
    map_zoom: f32,

    earth_texture: Option<egui::TextureHandle>,
}

impl HydronGuiApp {
    pub fn new(cc: &eframe::CreationContext<'_>, config: Config) -> Self {
        // Setup visual theme matching high-end digital twins (dark slate palette)
        let mut visuals = egui::Visuals::dark();
        visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(10, 15, 30);
        visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(20, 27, 45);
        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(30, 41, 59);
        visuals.widgets.active.bg_fill = egui::Color32::from_rgb(51, 65, 85);
        visuals.window_fill = egui::Color32::from_rgb(15, 23, 42);
        cc.egui_ctx.set_visuals(visuals);

        let constellation = create_satellites_from_config(&config);
        let ground_stations = config.stations.clone();
        
        let mut selected_id = "None".to_string();
        for seg in &constellation.segments {
            if !seg.satellites.is_empty() {
                selected_id = seg.satellites[0].id.clone();
                break;
            }
        }
        
        let mut app = Self {
            leo_num_input: config.leo_num,
            leo_alt_input: config.leo_alt_km,
            leo_inc_input: config.leo_inc_deg,
            meo_num_input: config.meo_num,
            meo_alt_input: config.meo_alt_km,
            meo_inc_input: config.meo_inc_deg,
            geo_num_input: config.geo_num,
            geo_alt_input: config.geo_alt_km,
            geo_inc_input: config.geo_inc_deg,
            
            sat_mass_input: config.leo_mass,
            sat_cd_input: config.leo_cd,
            sat_cr_input: config.leo_cr,
            
            gyro_noise: 1e-6,
            mag_noise: 1e-8,
            sun_noise: 1e-3,
            st_noise: 1e-4,
            
            force_disturbance: false,
            disturbance_val: [0.0, 0.0, 0.0],
            
            weather_overrides: vec![Some(0); ground_stations.len()],
            
            show_leo: true,
            show_meo: true,
            show_geo: true,
            show_sgl: true,
            show_left_panel: true,
            
            logs: vec!["System Digital Twin Initialized.".to_string()],
            
            selected_satellite_id: selected_id,
            constellation,
            ground_stations: ground_stations.clone(),
            atmos_model: AtmosphereModel {
                states: config.atmos_states.clone(),
                k_values: config.atmos_k.clone(),
                transition_matrix: config.transition_matrix.clone(),
                lcg: Lcg::new(42),
            },
            config,
            is_running: true,
            current_time: 0.0,
            time_warp: 1,
            step_size: 1.0,
            
            history_time: Vec::new(),
            history_stations: vec![Vec::new(); ground_stations.len()],
            history_total: Vec::new(),
            map_pitch: 0.4,
            map_yaw: 0.6,
            map_zoom: 1.0,
            earth_texture: None, // Will load below
        };

        // Load Earth texture map
        if let Ok(img_data) = std::fs::read("earth.jpg") {
            if let Ok(img) = image::load_from_memory_with_format(&img_data, image::ImageFormat::Jpeg) {
                let rgba = img.to_rgba8();
                let color_image = egui::ColorImage::from_rgba_unmultiplied(
                    [img.width() as usize, img.height() as usize],
                    rgba.as_raw(),
                );
                app.earth_texture = Some(cc.egui_ctx.load_texture(
                    "earth-texture",
                    color_image,
                    egui::TextureOptions::default(),
                ));
                app.log("Loaded Earth surface texture successfully.");
            } else {
                app.log("Warning: earth.jpg could not be decoded as JPEG.");
            }
        } else {
            app.log("Warning: earth.jpg texture file not found in directory.");
        }
        app.update_input_fields_for_selected();
        app
    }

    fn log(&mut self, msg: &str) {
        self.logs.push(format!("[{:.1}s] {}", self.current_time, msg));
        if self.logs.len() > 100 {
            self.logs.remove(0);
        }
    }

    fn update_input_fields_for_selected(&mut self) {
        let mut mass = 20.0;
        let mut cd = 2.2;
        let mut cr = 1.2;
        if let Some(sat) = self.find_satellite(&self.selected_satellite_id) {
            mass = sat.mass;
            cd = sat.cd;
            cr = sat.cr;
        }
        self.sat_mass_input = mass;
        self.sat_cd_input = cd;
        self.sat_cr_input = cr;
    }

    fn find_satellite(&self, id: &str) -> Option<&Satellite> {
        for seg in &self.constellation.segments {
            for sat in &seg.satellites {
                if sat.id == *id {
                    return Some(sat);
                }
            }
        }
        None
    }

    fn run_and_export_24h(&self) -> Result<String, std::io::Error> {
        use std::io::Write;

        let filename = "simulation_export.csv";
        let mut file = File::create(filename)?;

        // Write header
        let mut header = String::from("Time_s");
        for gs in &self.ground_stations {
            header.push_str(&format!(",{}", gs.id));
        }
        header.push_str(",Total_Throughput_Gbps,Active_ISL_Links,Active_SGL_Links\n");
        file.write_all(header.as_bytes())?;

        // Initialize temp states for 24h simulation run
        let mut constellation = create_satellites_from_config(&self.config);
        let mut ground_stations = self.config.stations.clone();
        let mut atmos_model = AtmosphereModel {
            states: self.config.atmos_states.clone(),
            k_values: self.config.atmos_k.clone(),
            transition_matrix: self.config.transition_matrix.clone(),
            lcg: Lcg::new(42),
        };

        let sim_duration = 86400.0;
        let step_size = 10.0; // 10s steps for excellent resolution
        let mut current_time = 0.0;
        
        let sun_vector = [1.0, 0.0, 0.0];
        let b_eci_mock = [1e-5, 2e-5, -3e-5];

        while current_time <= sim_duration {
            // 1. Step atmosphere
            for (idx, gs) in ground_stations.iter_mut().enumerate() {
                if let Some(forced_idx) = self.weather_overrides[idx] {
                    gs.atmos_state = forced_idx;
                    gs.k_value = atmos_model.k_values[forced_idx] / 1000.0;
                } else {
                    step_atmosphere(gs, &mut atmos_model);
                }
            }

            // 2. Step satellite dynamics
            for segment in &mut constellation.segments {
                for sat in &mut segment.satellites {
                    let rw_torque = [1e-3, -5e-4, 2e-4];
                    let mtq_dipole = [0.1, -0.05, 0.1];
                    step_orbit(sat, step_size, &self.config.env, sun_vector);
                    step_attitude(sat, step_size, b_eci_mock, rw_torque, mtq_dipole);
                }
            }

            // 3. Calculate positions and throughputs
            let gst = current_time * 7.292115e-5;
            let rot_mat = eci_to_ecef_matrix(gst);
            let rot_mat_t = [
                [rot_mat[0][0], rot_mat[1][0], rot_mat[2][0]],
                [rot_mat[0][1], rot_mat[1][1], rot_mat[2][1]],
                [rot_mat[0][2], rot_mat[1][2], rot_mat[2][2]],
            ];

            let all_sats: Vec<(String, OrbitType, [f64; 3])> = constellation.segments.iter()
                .flat_map(|seg| seg.satellites.iter().map(|s| (s.id.clone(), s.orbit_type.clone(), s.r)))
                .collect();

            let gs_eci_list: Vec<[f64; 3]> = ground_stations.iter().map(|gs| {
                let ecef = lla_to_ecef(gs.lat_rad, gs.lon_rad, gs.alt_m);
                mat_vec_mult(rot_mat_t, ecef)
            }).collect();

            let mut gs_throughputs = vec![0.0; ground_stations.len()];
            let mut total_throughput = 0.0;
            let mut active_sgl_links = 0;
            let mut active_isl_links = 0;

            // SGL links capacity
            for (_sat_id, orbit_type, sat_r) in &all_sats {
                let sat_max = match orbit_type {
                    OrbitType::LEO => 100.0_f64,
                    OrbitType::MEO => 400.0_f64,
                    OrbitType::GEO => 800.0_f64,
                };
                let sat_ref_dist = match orbit_type {
                    OrbitType::LEO => self.config.ref_dist_sgl_km,
                    OrbitType::MEO => self.config.meo_alt_km,
                    OrbitType::GEO => self.config.geo_alt_km,
                };

                let mut best_cap = 0.0_f64;
                let mut best_idx = usize::MAX;
                for (i, other_eci) in gs_eci_list.iter().enumerate() {
                    let cap = compute_link_capacity(
                        *sat_r, *other_eci, true,
                        ground_stations[i].k_value,
                        sat_ref_dist, sat_max, &self.config.env
                    ).min(sat_max);
                    if cap > best_cap {
                        best_cap = cap;
                        best_idx = i;
                    }
                }

                if best_idx < ground_stations.len() && best_cap > 0.0 {
                    gs_throughputs[best_idx] += best_cap;
                    total_throughput += best_cap;
                    active_sgl_links += 1;
                }
            }

            // ISL links
            for i in 0..all_sats.len() {
                for j in (i + 1)..all_sats.len() {
                    let (_id1, type1, r1) = &all_sats[i];
                    let (_id2, type2, r2) = &all_sats[j];

                    if visible(*r1, *r2, self.config.env.r_earth) {
                        let sat_max1 = match type1 {
                            OrbitType::LEO => 100.0_f64,
                            OrbitType::MEO => 400.0_f64,
                            OrbitType::GEO => 800.0_f64,
                        };
                        let sat_max2 = match type2 {
                            OrbitType::LEO => 100.0_f64,
                            OrbitType::MEO => 400.0_f64,
                            OrbitType::GEO => 800.0_f64,
                        };
                        let nominal_capacity = sat_max1.min(sat_max2);
                        let sat_ref_dist = match type1 {
                            OrbitType::LEO => self.config.ref_dist_isl_km,
                            OrbitType::MEO => self.config.meo_alt_km,
                            OrbitType::GEO => self.config.geo_alt_km,
                        };
                        let capacity = compute_link_capacity(*r1, *r2, false, 0.0, sat_ref_dist, nominal_capacity, &self.config.env);
                        if capacity > 0.0 {
                            active_isl_links += 1;
                        }
                    }
                }
            }

            // Write CSV row
            let mut row_str = format!("{:.1}", current_time);
            for val in &gs_throughputs {
                row_str.push_str(&format!(",{}", val));
            }
            row_str.push_str(&format!(",{},{},{}\n", total_throughput, active_isl_links, active_sgl_links));
            file.write_all(row_str.as_bytes())?;

            current_time += step_size;
        }

        Ok(filename.to_string())
    }
}

impl eframe::App for HydronGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Run continuous animation/repaint loop
        ctx.request_repaint();

        let mut pending_remove = None;
        let mut pending_add = false;
        let mut pending_reset = false;

        // 1. Core simulation physics steps
        if self.is_running {
            let mut pending_logs = Vec::new();
            let loops = self.time_warp.abs();
            let dt = if self.time_warp < 0 { -self.step_size } else { self.step_size };

            for _ in 0..loops {
                if self.current_time + dt < 0.0 {
                    self.current_time = 0.0;
                    break;
                }
                self.current_time += dt;
                let sun_vector = [1.0, 0.0, 0.0];
                let b_eci_mock = [1e-5, 2e-5, -3e-5];

                // Step atmosphere
                for (idx, gs) in &mut self.ground_stations.iter_mut().enumerate() {
                    if let Some(forced_idx) = self.weather_overrides[idx] {
                        if gs.atmos_state != forced_idx {
                            gs.atmos_state = forced_idx;
                            gs.k_value = self.atmos_model.k_values[forced_idx] / 1000.0;
                            let state_name = &self.atmos_model.states[forced_idx];
                            pending_logs.push(format!("Weather at {} forced to {}", gs.name, state_name));
                        }
                    } else {
                        let prev_state = gs.atmos_state;
                        step_atmosphere(gs, &mut self.atmos_model);
                        if gs.atmos_state != prev_state {
                            let state_name = &self.atmos_model.states[gs.atmos_state];
                            pending_logs.push(format!("Weather at {} transitioned to {}", gs.name, state_name));
                        }
                    }
                }

                // Step satellite dynamics
                for segment in &mut self.constellation.segments {
                    for sat in &mut segment.satellites {
                        let rw_torque = [1e-3, -5e-4, 2e-4];
                        let mut mtq_dipole = [0.1, -0.05, 0.1];

                        if sat.id == self.selected_satellite_id && self.force_disturbance {
                            mtq_dipole = add(mtq_dipole, self.disturbance_val);
                            self.force_disturbance = false;
                            pending_logs.push(format!("Injected attitude disturbance into satellite {}", sat.id));
                        }

                        step_orbit(sat, dt, &self.config.env, sun_vector);
                        step_attitude(sat, dt, b_eci_mock, rw_torque, mtq_dipole);
                    }
                }
            }
            for msg in pending_logs {
                self.log(&msg);
            }
        }

        // Pre-calculate positions and throughputs for all ground stations
        let gst = self.current_time * 7.292115e-5;
        let rot_mat = eci_to_ecef_matrix(gst);
        let rot_mat_t = [
            [rot_mat[0][0], rot_mat[1][0], rot_mat[2][0]],
            [rot_mat[0][1], rot_mat[1][1], rot_mat[2][1]],
            [rot_mat[0][2], rot_mat[1][2], rot_mat[2][2]],
        ];

        // Gather all active satellite ECI positions
        let all_sats: Vec<(String, OrbitType, [f64; 3])> = self.constellation.segments.iter()
            .flat_map(|seg| seg.satellites.iter().map(|s| (s.id.clone(), s.orbit_type.clone(), s.r)))
            .collect();

        // Gather all GS ECI positions
        let gs_eci_list: Vec<[f64; 3]> = self.ground_stations.iter().map(|gs| {
            let ecef = lla_to_ecef(gs.lat_rad, gs.lon_rad, gs.alt_m);
            mat_vec_mult(rot_mat_t, ecef)
        }).collect();

        // Pre-calculate connected satellites for each GS and throughputs
        let mut connected_sats_per_gs = vec![Vec::new(); self.ground_stations.len()];
        let mut gs_throughputs = vec![0.0f32; self.ground_stations.len()];
        let mut total_throughput = 0.0f32;

        for (sat_id, orbit_type, sat_r) in &all_sats {
            let sat_max = match orbit_type {
                OrbitType::LEO => 100.0_f64,
                OrbitType::MEO => 400.0_f64,
                OrbitType::GEO => 800.0_f64,
            };
            let sat_ref_dist = match orbit_type {
                OrbitType::LEO => self.config.ref_dist_sgl_km,
                OrbitType::MEO => self.config.meo_alt_km,
                OrbitType::GEO => self.config.geo_alt_km,
            };
            let orbit_label = match orbit_type {
                OrbitType::LEO => "LEO",
                OrbitType::MEO => "MEO",
                OrbitType::GEO => "GEO",
            };

            let mut best_cap = 0.0_f64;
            let mut best_idx = usize::MAX;
            for (i, other_eci) in gs_eci_list.iter().enumerate() {
                let cap = compute_link_capacity(
                    *sat_r, *other_eci, true,
                    self.ground_stations[i].k_value,
                    sat_ref_dist, sat_max, &self.config.env
                ).min(sat_max);
                if cap > best_cap {
                    best_cap = cap;
                    best_idx = i;
                }
            }

            if best_idx < self.ground_stations.len() && best_cap > 0.0 {
                connected_sats_per_gs[best_idx].push((sat_id.clone(), orbit_label, best_cap, sat_max));
                gs_throughputs[best_idx] += best_cap as f32;
                total_throughput += best_cap as f32;
            }
        }

        // Update history if running
        if self.is_running {
            self.history_time.push(self.current_time as f32);
            for i in 0..self.ground_stations.len() {
                self.history_stations[i].push(gs_throughputs[i]);
            }
            self.history_total.push(total_throughput);

            // Limit history size to 300 points (e.g. 5 minutes at 1Hz)
            let max_history = 300;
            if self.history_time.len() > max_history {
                self.history_time.remove(0);
                for i in 0..self.ground_stations.len() {
                    self.history_stations[i].remove(0);
                }
                self.history_total.remove(0);
            }
        }

        // 2. GUI panels layout
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let btn_text = if self.show_left_panel { "⏴ Stazioni" } else { "⏵ Stazioni" };
                if ui.button(btn_text).clicked() {
                    self.show_left_panel = !self.show_left_panel;
                }
                ui.separator();

                ui.heading("🛰 HydRON Digital Twin");
                ui.separator();

                if ui.button(if self.is_running { "⏸ Pause" } else { "▶ Play" }).clicked() {
                    self.is_running = !self.is_running;
                    self.log(if self.is_running { "Simulation Resumed" } else { "Simulation Paused" });
                }

                if ui.button("⏭ Step").clicked() {
                    self.is_running = false;
                    self.current_time += self.step_size;
                    let sun_vector = [1.0, 0.0, 0.0];
                    let b_eci_mock = [1e-5, 2e-5, -3e-5];
                    for gs in &mut self.ground_stations {
                        step_atmosphere(gs, &mut self.atmos_model);
                    }
                    for segment in &mut self.constellation.segments {
                        for sat in &mut segment.satellites {
                            step_orbit(sat, self.step_size, &self.config.env, sun_vector);
                            step_attitude(sat, self.step_size, b_eci_mock, [1e-3, -5e-4, 2e-4], [0.1, -0.05, 0.1]);
                        }
                    }
                    self.log("Single Step Executed");
                }

                if ui.button("↺ Reset").clicked() {
                    pending_reset = true;
                }

                if ui.button("📥 Esporta 24h CSV").clicked() {
                    match self.run_and_export_24h() {
                        Ok(file) => {
                            self.log(&format!("Dati di 24h esportati in '{}'", file));
                        }
                        Err(e) => {
                            self.log(&format!("Errore esportazione dati: {}", e));
                        }
                    }
                }

                ui.separator();
                ui.label("Time Warp:");
                ui.add(egui::Slider::new(&mut self.time_warp, -50..=50).text("x"));

                ui.separator();
                ui.label(format!("Epoch: {:.1}s", self.current_time));
            }); // horizontal
        }); // top_panel

        if self.show_left_panel {
            egui::SidePanel::left("left_panel").width_range(290.0..=340.0).show(ctx, |ui| {
                ui.separator();
                ui.collapsing("⚙ Filtri Visualizzazione", |ui| {
                    ui.checkbox(&mut self.show_leo, "LEO ISL");
                    ui.checkbox(&mut self.show_meo, "MEO ISL");
                    ui.checkbox(&mut self.show_geo, "GEO ISL");
                    ui.checkbox(&mut self.show_sgl, "Ground Links (SGL)");
                    ui.separator();
                    ui.label("Zoom Mappa 3D:");
                    ui.add(egui::Slider::new(&mut self.map_zoom, 0.1..=10.0).logarithmic(true));
                });

                ui.separator();
                ui.collapsing("📡 Modifica Costellazione", |ui| {
                    ui.collapsing("Parametri Segmenti", |ui| {
                        ui.label("Segmento LEO:");
                        ui.add(egui::Slider::new(&mut self.leo_num_input, 0..=20).text("Satelliti"));
                        ui.add(egui::Slider::new(&mut self.leo_alt_input, 200.0..=1200.0).text("Quota (km)"));
                        ui.add(egui::Slider::new(&mut self.leo_inc_input, 0.0..=180.0).text("Inclinazione (°)"));

                        ui.label("Segmento MEO:");
                        ui.add(egui::Slider::new(&mut self.meo_num_input, 0..=8).text("Satelliti"));
                        ui.add(egui::Slider::new(&mut self.meo_alt_input, 5000.0..=15000.0).text("Quota (km)"));
                        ui.add(egui::Slider::new(&mut self.meo_inc_input, 0.0..=180.0).text("Inclinazione (°)"));

                        ui.label("Segmento GEO:");
                        ui.add(egui::Slider::new(&mut self.geo_num_input, 0..=6).text("Satelliti"));
                        ui.add(egui::Slider::new(&mut self.geo_alt_input, 30000.0..=40000.0).text("Quota (km)"));
                        ui.add(egui::Slider::new(&mut self.geo_inc_input, 0.0..=90.0).text("Inclinazione (°)"));
                        
                        let changed = self.config.leo_num != self.leo_num_input
                            || self.config.leo_alt_km != self.leo_alt_input
                            || self.config.leo_inc_deg != self.leo_inc_input
                            || self.config.meo_num != self.meo_num_input
                            || self.config.meo_alt_km != self.meo_alt_input
                            || self.config.meo_inc_deg != self.meo_inc_input
                            || self.config.geo_num != self.geo_num_input
                            || self.config.geo_alt_km != self.geo_alt_input
                            || self.config.geo_inc_deg != self.geo_inc_input;

                        if changed {
                            self.config.leo_num = self.leo_num_input;
                            self.config.leo_alt_km = self.leo_alt_input;
                            self.config.leo_inc_deg = self.leo_inc_input;

                            self.config.meo_num = self.meo_num_input;
                            self.config.meo_alt_km = self.meo_alt_input;
                            self.config.meo_inc_deg = self.meo_inc_input;

                            self.config.geo_num = self.geo_num_input;
                            self.config.geo_alt_km = self.geo_alt_input;
                            self.config.geo_inc_deg = self.geo_inc_input;

                            self.constellation = create_satellites_from_config(&self.config);
                            
                            let mut found_any = false;
                            for seg in &self.constellation.segments {
                                if !seg.satellites.is_empty() {
                                    self.selected_satellite_id = seg.satellites[0].id.clone();
                                    found_any = true;
                                    break;
                                }
                            }
                            if !found_any {
                                self.selected_satellite_id = "None".to_string();
                            }
                            
                            self.update_input_fields_for_selected();
                            self.log("Constellation reconfigured dynamically");
                        }
                    }); // Parametri Segmenti

                    ui.separator();
                    ui.collapsing("🌦 Meteo Stazioni", |ui| {
                        for i in 0..self.ground_stations.len() {
                            let name = self.ground_stations[i].name.clone();
                            ui.label(format!("{}:", name));
                            ui.horizontal(|ui| {
                                if ui.selectable_label(self.weather_overrides[i].is_none(), "Markov").clicked() {
                                    self.weather_overrides[i] = None;
                                }
                                for w_idx in 0..self.atmos_model.states.len() {
                                    let w_name = &self.atmos_model.states[w_idx];
                                    if ui.selectable_label(self.weather_overrides[i] == Some(w_idx), w_name).clicked() {
                                        self.weather_overrides[i] = Some(w_idx);
                                    }
                                }
                            });
                        }
                    }); // Meteo Stazioni

                    ui.separator();
                    ui.collapsing("📡 Modifica Stazioni", |ui| {
                        let mut to_remove = None;
                        for i in 0..self.ground_stations.len() {
                            ui.group(|ui| {
                                ui.horizontal(|ui| {
                                    let mut name_edit = self.ground_stations[i].name.clone();
                                    if ui.text_edit_singleline(&mut name_edit).changed() {
                                        self.ground_stations[i].name = name_edit;
                                    }
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        if ui.button("❌").clicked() {
                                            to_remove = Some(i);
                                        }
                                    });
                                });
                                
                                let mut lat_deg = self.ground_stations[i].lat_rad.to_degrees();
                                let mut lon_deg = self.ground_stations[i].lon_rad.to_degrees();
                                let mut alt_m = self.ground_stations[i].alt_m;

                                if ui.add(egui::Slider::new(&mut lat_deg, -90.0..=90.0).text("Lat (°)")).changed() {
                                    self.ground_stations[i].lat_rad = lat_deg.to_radians();
                                }
                                if ui.add(egui::Slider::new(&mut lon_deg, -180.0..=180.0).text("Lon (°)")).changed() {
                                    self.ground_stations[i].lon_rad = lon_deg.to_radians();
                                }
                                if ui.add(egui::Slider::new(&mut alt_m, 0.0..=5000.0).text("Alt (m)")).changed() {
                                    self.ground_stations[i].alt_m = alt_m;
                                }
                            });
                            ui.add_space(2.0);
                        }

                        if let Some(idx) = to_remove {
                            pending_remove = Some(idx);
                        }

                        ui.add_space(4.0);
                        if ui.button("➕ Aggiungi Stazione").clicked() {
                            pending_add = true;
                        }
                    });
                }); // 📡 Modifica Costellazione outer collapsing

                ui.separator();
                ui.collapsing("🛰 Stazioni di Terra", |ui| {
                    egui::ScrollArea::vertical().id_source("gs_scroll").show(ui, |ui| {
                        for (gs_idx, gs) in self.ground_stations.iter().enumerate() {
                            let weather_name = &self.atmos_model.states[gs.atmos_state];
                            
                            let (wx_icon, wx_color) = match gs.atmos_state {
                                0 => ("☀", egui::Color32::from_rgb(34, 197, 94)),
                        1 => ("🌤", egui::Color32::from_rgb(234, 179, 8)),
                        2 => ("☁", egui::Color32::from_rgb(156, 163, 175)),
                        _ => ("🌧", egui::Color32::from_rgb(239, 68, 68)),
                    };
 
                    let connected = &connected_sats_per_gs[gs_idx];
                    let total_gbps = gs_throughputs[gs_idx] as f64;

                    ui.group(|ui| {
                        // Station header
                        ui.horizontal(|ui| {
                            ui.colored_label(wx_color, wx_icon);
                            ui.colored_label(egui::Color32::WHITE, &gs.name);
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.colored_label(wx_color, weather_name.to_uppercase());
                            });
                        });

                        // Subtitle row
                        ui.horizontal(|ui| {
                            ui.small(format!("{}", gs.id));
                            ui.separator();
                            ui.small(format!("k={:.4}/km", gs.k_value * 1000.0));
                            ui.separator();
                            ui.small("GS: ∞ Gbps");
                        });

                        ui.separator();

                        // Total throughput + mini bar
                        if connected.is_empty() {
                            ui.colored_label(egui::Color32::DARK_GRAY, "Nessun satellite in vista");
                        } else {
                            ui.horizontal(|ui| {
                                ui.label("📡 Throughput:");
                                let color = if total_gbps > 200.0 {
                                    egui::Color32::from_rgb(34, 197, 94)
                                } else if total_gbps > 50.0 {
                                    egui::Color32::from_rgb(234, 179, 8)
                                } else {
                                    egui::Color32::from_rgb(239, 68, 68)
                                };
                                ui.colored_label(color, format!("{:.1} Gbps", total_gbps));
                            });

                            // Throughput bar (max scale = 800 Gbps = 1 GEO)
                            let bar_w = (ui.available_width() - 4.0).max(0.0);
                            let bar_h = 5.0_f32;
                            let fill_frac = (total_gbps / 800.0).min(1.0) as f32;
                            let (bar_rect, _) = ui.allocate_exact_size(egui::vec2(bar_w, bar_h), egui::Sense::hover());
                            ui.painter().rect_filled(bar_rect, 2.0, egui::Color32::from_rgb(30, 40, 60));
                            if fill_frac > 0.0 {
                                let fill_w = (bar_w * fill_frac).max(0.0);
                                let fill_rect = egui::Rect::from_min_size(bar_rect.min, egui::vec2(fill_w, bar_h));
                                let bar_color = if total_gbps > 200.0 {
                                    egui::Color32::from_rgb(34, 197, 94)
                                } else if total_gbps > 50.0 {
                                    egui::Color32::from_rgb(234, 179, 8)
                                } else {
                                    egui::Color32::from_rgb(239, 68, 68)
                                };
                                ui.painter().rect_filled(fill_rect, 2.0, bar_color);
                            }

                            ui.add_space(2.0);

                            // Connected satellites list
                            for (sat_id, orbit_label, speed, sat_max) in connected {
                                let frac = (speed / sat_max).min(1.0);
                                let link_color = if frac > 0.5 {
                                    egui::Color32::from_rgb(34, 197, 94)
                                } else if frac > 0.1 {
                                    egui::Color32::from_rgb(234, 179, 8)
                                } else {
                                    egui::Color32::from_rgb(239, 68, 68)
                                };
                                ui.horizontal(|ui| {
                                    ui.colored_label(link_color, "◉");
                                    ui.colored_label(egui::Color32::LIGHT_GRAY, format!("{} [{}]", sat_id, orbit_label));
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        ui.colored_label(link_color, format!("{:.1} Gbps", speed));
                                    });
                                });
                            }
                        }
                    });
                    ui.add_space(3.0);
                } // end for gs
            }); // ScrollArea
                }); // Collapsing Stazioni di Terra
            }); // left_panel
        }

        egui::SidePanel::right("right_panel").width_range(280.0..=330.0).show(ctx, |ui| {
            ui.heading("Pannello Satelliti");
            ui.separator();

            ui.label("Seleziona Satellite:");
            egui::ComboBox::from_label("")
                .selected_text(self.selected_satellite_id.clone())
                .show_ui(ui, |ui| {
                    let sat_ids: Vec<String> = self.constellation.segments.iter()
                        .flat_map(|seg| seg.satellites.iter().map(|s| s.id.clone()))
                        .collect();
                    for id in sat_ids {
                        if ui.selectable_value(&mut self.selected_satellite_id, id.clone(), id.clone()).clicked() {
                            self.update_input_fields_for_selected();
                        }
                    }
                });

            ui.separator();

            let sat_telemetry = self.find_satellite(&self.selected_satellite_id).map(|s| (
                s.mass,
                s.inertia,
                s.r,
                s.v,
                s.q,
                s.omega,
                s.h_rw,
                s.orbit_type.clone(),
            ));

            if let Some((mass, inertia, r, v, q, omega, h_rw, orbit_type)) = sat_telemetry {
                ui.heading(format!("Stato {}", self.selected_satellite_id));
                
                let max_spd = match orbit_type {
                    OrbitType::LEO => 100.0,
                    OrbitType::MEO => 400.0,
                    OrbitType::GEO => 800.0,
                };
                ui.label(format!("Velocità Max Canale: {:.0} Gbps", max_spd));
                ui.label(format!("Massa Bus: {:.1} kg", mass));
                ui.label(format!("Inerzia (Ix,Iy,Iz): [{:.2}, {:.2}, {:.2}] kg*m^2", inertia[0], inertia[1], inertia[2]));
                
                ui.collapsing("Parametri Fisici", |ui| {
                    ui.add(egui::Slider::new(&mut self.sat_mass_input, 1.0..=500.0).text("Massa (kg)"));
                    ui.add(egui::Slider::new(&mut self.sat_cd_input, 0.0..=4.0).text("Drag Cd"));
                    ui.add(egui::Slider::new(&mut self.sat_cr_input, 0.0..=3.0).text("SRP Cr"));
                    
                    if ui.button("Applica Parametri").clicked() {
                        let id = self.selected_satellite_id.clone();
                        for seg in &mut self.constellation.segments {
                            for s in &mut seg.satellites {
                                if s.id == id {
                                    s.mass = self.sat_mass_input;
                                    s.cd = self.sat_cd_input;
                                    s.cr = self.sat_cr_input;
                                }
                            }
                        }
                        self.log(&format!("Updated physical params for satellite {}", id));
                    }
                });

                ui.separator();
                ui.label("Orbita (ECI):");
                ui.label(format!("Pos: [{:.1}, {:.1}, {:.1}] km", r[0]/1000.0, r[1]/1000.0, r[2]/1000.0));
                ui.label(format!("Vel: [{:.3}, {:.3}, {:.3}] km/s", v[0]/1000.0, v[1]/1000.0, v[2]/1000.0));

                ui.separator();
                ui.label("Atteggiamento:");
                ui.label(format!("Quaternione Q:\n[{:.4}, {:.4}, {:.4}, {:.4}]", q[0], q[1], q[2], q[3]));
                ui.label(format!("Vel. angolare Omega (rad/s):\n[{:.4}, {:.4}, {:.4}]", omega[0], omega[1], omega[2]));

                ui.separator();
                ui.label("Attuatori ADCS (Reaction Wheels):");
                ui.label(format!("Momento H_rw:\n[{:.4}, {:.4}, {:.4}] Nms", h_rw[0], h_rw[1], h_rw[2]));

                ui.separator();
                ui.heading("Iniettore Disturbi ADCS");
                ui.add(egui::Slider::new(&mut self.disturbance_val[0], -10.0..=10.0).text("Tx"));
                ui.add(egui::Slider::new(&mut self.disturbance_val[1], -10.0..=10.0).text("Ty"));
                ui.add(egui::Slider::new(&mut self.disturbance_val[2], -10.0..=10.0).text("Tz"));
                if ui.button("Inietta Coppia Disturbo").clicked() {
                    self.force_disturbance = true;
                }

                ui.separator();
                ui.heading("Rumore Sensori");
                ui.add(egui::Slider::new(&mut self.gyro_noise, 1e-7..=1e-3).logarithmic(true).text("Gyro"));
                ui.add(egui::Slider::new(&mut self.mag_noise, 1e-9..=1e-5).logarithmic(true).text("Magnetom"));
                ui.add(egui::Slider::new(&mut self.sun_noise, 1e-5..=1e-1).logarithmic(true).text("Sun Sensor"));
                ui.add(egui::Slider::new(&mut self.st_noise, 1e-6..=1e-2).logarithmic(true).text("StarTracker"));

                ui.heading("Console Telemetria e Routing");
                ui.separator();
            egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                for log_msg in &self.logs {
                    ui.label(log_msg);
                }
            });
            }
            
        });

        egui::TopBottomPanel::bottom("bottom_panel").height_range(130.0..=170.0).show(ctx, |ui| {
            ui.heading("📊 Grafico Storico Throughput Stazioni di Terra");
            let (rect, _response) = ui.allocate_exact_size(
                ui.available_size(),
                egui::Sense::hover()
            );

            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(10, 15, 30));
            painter.rect_stroke(rect, 4.0, egui::Stroke::new(1.0, egui::Color32::from_rgb(30, 41, 59)));

            if self.history_time.len() < 2 {
                painter.text(rect.center(), egui::Align2::CENTER_CENTER, "In attesa di dati di simulazione...", egui::FontId::proportional(12.0), egui::Color32::GRAY);
                return;
            }

            // Find max value in history to scale Y axis (with a minimum of 100 Gbps)
            let mut max_y = 100.0_f32;
            for val in &self.history_total {
                if *val > max_y {
                    max_y = *val;
                }
            }
            max_y *= 1.1; // Add 10% headroom

            let min_x = self.history_time[0];
            let max_x = *self.history_time.last().unwrap();
            let dx = max_x - min_x;

            let margin_left = 65.0f32;
            let margin_right = 15.0f32;
            let margin_top = 22.0f32;
            let margin_bottom = 15.0f32;

            let plot_width = rect.width() - margin_left - margin_right;
            let plot_height = rect.height() - margin_top - margin_bottom;

            let to_screen = |x: f32, y: f32| -> egui::Pos2 {
                let x_frac = if dx > 0.0 { (x - min_x) / dx } else { 0.0 };
                let y_frac = y / max_y;
                egui::pos2(
                    rect.min.x + margin_left + x_frac * plot_width,
                    rect.max.y - margin_bottom - y_frac * plot_height,
                )
            };

            // Draw Y axis grid lines and labels
            let grid_lines = 3;
            for k in 0..=grid_lines {
                let y_val = (k as f32 / grid_lines as f32) * max_y;
                let pos_left = to_screen(min_x, y_val);
                let pos_right = to_screen(max_x, y_val);
                painter.line_segment([pos_left, pos_right], egui::Stroke::new(0.5, egui::Color32::from_rgba_unmultiplied(100, 100, 100, 30)));
                painter.text(
                    egui::pos2(rect.min.x + margin_left - 5.0, pos_left.y),
                    egui::Align2::RIGHT_CENTER,
                    format!("{:.0} Gbps", y_val),
                    egui::FontId::proportional(9.0),
                    egui::Color32::GRAY
                );
            }

            // Draw X axis grid lines and labels (epoch times)
            let grid_lines_x = 5;
            for k in 0..=grid_lines_x {
                let x_val = min_x + (k as f32 / grid_lines_x as f32) * dx;
                let pos_bottom = to_screen(x_val, 0.0);
                let pos_top = to_screen(x_val, max_y);
                painter.line_segment([pos_bottom, pos_top], egui::Stroke::new(0.5, egui::Color32::from_rgba_unmultiplied(100, 100, 100, 30)));
                painter.text(
                    egui::pos2(pos_bottom.x, rect.max.y - margin_bottom + 8.0),
                    egui::Align2::CENTER_CENTER,
                    format!("{:.0}s", x_val),
                    egui::FontId::proportional(9.0),
                    egui::Color32::GRAY
                );
            }

            // Draw station lines
            let colors = [
                egui::Color32::from_rgb(56, 189, 248),   // sky blue
                egui::Color32::from_rgb(234, 179, 8),    // gold
                egui::Color32::from_rgb(168, 85, 247),   // purple
                egui::Color32::from_rgb(236, 72, 153),   // pink
            ];

            for i in 0..self.ground_stations.len() {
                let color = colors[i % colors.len()];
                let mut points = Vec::new();
                for k in 0..self.history_time.len() {
                    points.push(to_screen(self.history_time[k], self.history_stations[i][k]));
                }
                for w in points.windows(2) {
                    painter.line_segment([w[0], w[1]], egui::Stroke::new(1.2, color));
                }
            }

            // Draw total aggregate line (thick white)
            let mut total_points = Vec::new();
            for k in 0..self.history_time.len() {
                total_points.push(to_screen(self.history_time[k], self.history_total[k]));
            }
            for w in total_points.windows(2) {
                painter.line_segment([w[0], w[1]], egui::Stroke::new(2.2, egui::Color32::WHITE));
            }

            // Draw legend
            let mut legend_x = rect.min.x + margin_left + 15.0;
            let legend_y = rect.min.y + 12.0;
            
            // Draw Total legend
            painter.circle_filled(egui::pos2(legend_x, legend_y), 3.0, egui::Color32::WHITE);
            painter.text(egui::pos2(legend_x + 8.0, legend_y), egui::Align2::LEFT_CENTER, "Totale Aggregato", egui::FontId::proportional(9.0), egui::Color32::WHITE);
            legend_x += 105.0;

            for i in 0..self.ground_stations.len() {
                let name = &self.ground_stations[i].name;
                let color = colors[i % colors.len()];
                painter.circle_filled(egui::pos2(legend_x, legend_y), 3.0, color);
                painter.text(egui::pos2(legend_x + 8.0, legend_y), egui::Align2::LEFT_CENTER, name, egui::FontId::proportional(9.0), egui::Color32::LIGHT_GRAY);
                legend_x += 70.0;
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Visualizzazione Costellazione 3D (Trascina per ruotare il globo)");
            
            let (rect, response) = ui.allocate_exact_size(
                ui.available_size(),
                egui::Sense::drag()
            );

            if response.dragged() {
                let delta = response.drag_delta();
                self.map_yaw += delta.x * 0.005;
                self.map_pitch = (self.map_pitch - delta.y * 0.005).clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
            }

            if response.hovered() {
                let scroll_delta = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll_delta != 0.0 {
                    let zoom_factor = (scroll_delta * 0.003).exp();
                    self.map_zoom = (self.map_zoom * zoom_factor).clamp(0.1, 10.0);
                }
            }

            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(5, 7, 18));

            let center = rect.center();
            
            let max_r = self.config.env.r_earth + self.config.geo_alt_km * 1000.0;
            let screen_dim = rect.width().min(rect.height());
            let scale = ((screen_dim * 0.45) as f64 / max_r) * (self.map_zoom as f64);

            // 3D projection closure: projects [x, y, z] to screen space and returns (pos2, rotated_z)
            let project_3d = |pos: [f64; 3]| -> (egui::Pos2, f64) {
                let x = pos[0];
                let y = -pos[1]; // Invert Y to correct longitude coordinate system orientation
                let z = pos[2];

                // 1. Rotate around Y-axis by map_yaw
                let cos_yaw = (self.map_yaw as f64).cos();
                let sin_yaw = (self.map_yaw as f64).sin();
                let x1 = x * cos_yaw - z * sin_yaw;
                let z1 = x * sin_yaw + z * cos_yaw;
                let y1 = y;

                // 2. Rotate around X-axis by map_pitch
                let cos_pitch = (self.map_pitch as f64).cos();
                let sin_pitch = (self.map_pitch as f64).sin();
                let x2 = x1;
                let y2 = y1 * cos_pitch - z1 * sin_pitch;
                let z2 = y1 * sin_pitch + z1 * cos_pitch; // positive is towards camera

                // 3. Screen projection
                let screen_x = center.x + (x2 * scale) as f32;
                let screen_y = center.y + (y2 * scale) as f32;

                (egui::pos2(screen_x, screen_y), z2)
            };

            // Draw Earth (textured 3D sphere mesh, or fallback to solid blue circle)
            let r_earth = self.config.env.r_earth;
            let earth_radius_px = (r_earth * scale) as f32;
            if let Some(ref texture) = self.earth_texture {
                let n_lat = 32;
                let n_lon = 64;
                let mut projected_vertices = vec![vec![(egui::pos2(0.0, 0.0), 0.0); n_lon + 1]; n_lat + 1];
                for i in 0..=n_lat {
                    let lat_rad = -std::f64::consts::FRAC_PI_2 + (i as f64) * std::f64::consts::PI / (n_lat as f64);
                    let z = r_earth * lat_rad.sin();
                    let r_lat = r_earth * lat_rad.cos();
                    
                    for j in 0..=n_lon {
                        let lon_rad = (j as f64) * 2.0 * std::f64::consts::PI / (n_lon as f64) + gst + 180.0_f64.to_radians();
                        let x = r_lat * lon_rad.cos();
                        let y = r_lat * lon_rad.sin();
                        
                        projected_vertices[i][j] = project_3d([x, y, z]);
                    }
                }

                let mut mesh = egui::Mesh::with_texture(texture.id());
                let mut vertex_indices = vec![vec![u32::MAX; n_lon + 1]; n_lat + 1];

                for i in 0..n_lat {
                    for j in 0..n_lon {
                        let p00 = projected_vertices[i][j];
                        let p10 = projected_vertices[i+1][j];
                        let p01 = projected_vertices[i][j+1];
                        let p11 = projected_vertices[i+1][j+1];

                        let avg_z = (p00.1 + p10.1 + p01.1 + p11.1) / 4.0;
                        if avg_z > 0.0 {
                            let mut add_vertex = |row: usize, col: usize, mesh: &mut egui::Mesh| -> u32 {
                                if vertex_indices[row][col] == u32::MAX {
                                    let (pos, _) = projected_vertices[row][col];
                                    let u = col as f32 / n_lon as f32;
                                    let v = 1.0 - (row as f32 / n_lat as f32);
                                    let idx = mesh.vertices.len() as u32;
                                    mesh.vertices.push(egui::epaint::Vertex {
                                        pos,
                                        uv: egui::pos2(u, v),
                                        color: egui::Color32::WHITE,
                                    });
                                    vertex_indices[row][col] = idx;
                                    idx
                                } else {
                                    vertex_indices[row][col]
                                }
                            };

                            let idx00 = add_vertex(i, j, &mut mesh);
                            let idx10 = add_vertex(i + 1, j, &mut mesh);
                            let idx01 = add_vertex(i, j + 1, &mut mesh);
                            let idx11 = add_vertex(i + 1, j + 1, &mut mesh);

                            mesh.add_triangle(idx00, idx10, idx01);
                            mesh.add_triangle(idx10, idx11, idx01);
                        }
                    }
                }
                painter.add(mesh);
            } else {
                painter.circle_filled(center, earth_radius_px, egui::Color32::from_rgb(15, 76, 129));
            }
            painter.circle_stroke(center, earth_radius_px, egui::Stroke::new(1.5, egui::Color32::from_rgb(56, 189, 248)));

            // Draw Earth's yellow latitude/longitude grid
            let grid_color = egui::Color32::from_rgba_unmultiplied(253, 224, 71, 100); // Yellow grid lines
            let grid_stroke = egui::Stroke::new(1.0, grid_color);
            let r_earth = self.config.env.r_earth;

            // Parallels (latitude lines)
            for lat_deg in (-60..=60).step_by(20) {
                let lat_rad = (lat_deg as f64).to_radians();
                let z = r_earth * lat_rad.sin();
                let r_lat = r_earth * lat_rad.cos();
                
                let mut prev_pt: Option<egui::Pos2> = None;
                let steps = 72;
                for step in 0..=steps {
                    let lon_rad = (step as f64 * 360.0 / steps as f64).to_radians() + gst;
                    let x = r_lat * lon_rad.cos();
                    let y = r_lat * lon_rad.sin();
                    
                    let (screen_pos, rot_z) = project_3d([x, y, z]);
                    if rot_z > 0.0 {
                        if let Some(prev) = prev_pt {
                            painter.line_segment([prev, screen_pos], grid_stroke);
                        }
                        prev_pt = Some(screen_pos);
                    } else {
                        prev_pt = None;
                    }
                }
            }

            // Meridians (longitude lines)
            for lon_deg in (0..360).step_by(30) {
                let lon_rad = (lon_deg as f64).to_radians() + gst;
                
                let mut prev_pt: Option<egui::Pos2> = None;
                let steps = 72;
                for step in -steps/2..=steps/2 {
                    let lat_rad = (step as f64 * 90.0 / (steps as f64 / 2.0)).to_radians();
                    let x = r_earth * lat_rad.cos() * lon_rad.cos();
                    let y = r_earth * lat_rad.cos() * lon_rad.sin();
                    let z = r_earth * lat_rad.sin();
                    
                    let (screen_pos, rot_z) = project_3d([x, y, z]);
                    if rot_z > 0.0 {
                        if let Some(prev) = prev_pt {
                            painter.line_segment([prev, screen_pos], grid_stroke);
                        }
                        prev_pt = Some(screen_pos);
                    } else {
                        prev_pt = None;
                    }
                }
            }

            // Draw Earth's rotation axis
            let axis_len = r_earth * 1.25;
            let (axis_north_px, north_z) = project_3d([0.0, 0.0, axis_len]);
            let (axis_south_px, south_z) = project_3d([0.0, 0.0, -axis_len]);
            painter.line_segment(
                [axis_south_px, axis_north_px],
                egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(100, 116, 139, 100))
            );
            
            if north_z > 0.0 {
                painter.text(
                    axis_north_px,
                    egui::Align2::CENTER_CENTER,
                    "N",
                    egui::FontId::proportional(11.0),
                    egui::Color32::from_rgb(56, 189, 248)
                );
            }
            if south_z > 0.0 {
                painter.text(
                    axis_south_px,
                    egui::Align2::CENTER_CENTER,
                    "S",
                    egui::FontId::proportional(11.0),
                    egui::Color32::from_rgb(239, 68, 68)
                );
            }

            // Draw Orbit paths
            let draw_orbit_3d = |painter: &egui::Painter, r: f64, color: egui::Color32| {
                let mut prev_pt: Option<egui::Pos2> = None;
                let steps = 120;
                for step in 0..=steps {
                    let theta = (step as f64 * 360.0 / steps as f64).to_radians();
                    let x = r * theta.cos();
                    let y = r * theta.sin();
                    let z = 0.0;
                    let (screen_pos, rot_z) = project_3d([x, y, z]);
                    
                    let dist = screen_pos.distance(center);
                    let occluded = rot_z < 0.0 && dist < earth_radius_px;
                    
                    let stroke_color = if occluded {
                        color.linear_multiply(0.12)
                    } else {
                        color.linear_multiply(0.4)
                    };
                    
                    if let Some(prev) = prev_pt {
                        painter.line_segment([prev, screen_pos], egui::Stroke::new(1.0, stroke_color));
                    }
                    prev_pt = Some(screen_pos);
                }
            };

            let leo_r = self.config.env.r_earth + self.config.leo_alt_km * 1000.0;
            draw_orbit_3d(&painter, leo_r, egui::Color32::from_rgb(56, 189, 248));
            
            let meo_r = self.config.env.r_earth + self.config.meo_alt_km * 1000.0;
            draw_orbit_3d(&painter, meo_r, egui::Color32::from_rgb(192, 132, 252));
            
            let geo_r = self.config.env.r_earth + self.config.geo_alt_km * 1000.0;
            draw_orbit_3d(&painter, geo_r, egui::Color32::from_rgb(251, 146, 60));

            // Gather all active node screen positions
            let mut satellites_screen = Vec::new();
            for seg in &self.constellation.segments {
                for sat in &seg.satellites {
                    let (sat_pos_px, rot_z) = project_3d(sat.r);
                    satellites_screen.push((sat.id.clone(), sat.orbit_type.clone(), sat_pos_px, sat.r, rot_z));
                }
            }

            let mut stations_screen = Vec::new();
            for (gs_idx, gs) in self.ground_stations.iter().enumerate() {
                let gs_eci = gs_eci_list[gs_idx];
                let (gs_pos_px, rot_z) = project_3d(gs_eci);
                stations_screen.push((gs.id.clone(), gs_pos_px, gs_eci, gs.k_value, rot_z));
            }

            let mut sat_visible_to_gs = std::collections::HashSet::new();
            for (sat_id, orbit_type, _, sat_r, _) in &satellites_screen {
                if *orbit_type == OrbitType::LEO {
                    for (_, _, gs_r, _, _) in &stations_screen {
                        if visible_sgl(*sat_r, *gs_r, self.config.env.r_earth) {
                            sat_visible_to_gs.insert(sat_id.clone());
                            break;
                        }
                    }
                }
            }

            // Draw active links between Satellites (ISL)
            for i in 0..satellites_screen.len() {
                for j in (i + 1)..satellites_screen.len() {
                    let (id1, type1, pos1_px, r1, rot_z1) = &satellites_screen[i];
                    let (id2, type2, pos2_px, r2, rot_z2) = &satellites_screen[j];

                    let is_leo_gs_vis = (type1 == &OrbitType::LEO && sat_visible_to_gs.contains(id1))
                        || (type2 == &OrbitType::LEO && sat_visible_to_gs.contains(id2));

                    let show_link = match (type1, type2) {
                        (OrbitType::LEO, OrbitType::LEO) => self.show_leo,
                        (OrbitType::MEO, OrbitType::MEO) => self.show_meo,
                        (OrbitType::GEO, OrbitType::GEO) => self.show_geo,
                        _ => self.show_meo || self.show_geo || self.show_leo,
                    } && !is_leo_gs_vis;

                    if show_link && visible(*r1, *r2, self.config.env.r_earth) {
                        let sat_max1 = match type1 {
                            OrbitType::LEO => 100.0_f64,
                            OrbitType::MEO => 400.0_f64,
                            OrbitType::GEO => 800.0_f64,
                        };
                        let sat_max2 = match type2 {
                            OrbitType::LEO => 100.0_f64,
                            OrbitType::MEO => 400.0_f64,
                            OrbitType::GEO => 800.0_f64,
                        };
                        let nominal_capacity = sat_max1.min(sat_max2);
                        let sat_ref_dist = match type1 {
                            OrbitType::LEO => self.config.ref_dist_isl_km,
                            OrbitType::MEO => self.config.meo_alt_km,
                            OrbitType::GEO => self.config.geo_alt_km,
                        };
                        let capacity = compute_link_capacity(*r1, *r2, false, 0.0, sat_ref_dist, nominal_capacity, &self.config.env);
                        if capacity > 0.0 {
                            let color = if capacity > 5.0 {
                                egui::Color32::from_rgb(34, 197, 94)
                            } else if capacity > 1.0 {
                                egui::Color32::from_rgb(234, 179, 8)
                            } else {
                                egui::Color32::from_rgb(239, 68, 68)
                            };
                            
                            let dist1 = pos1_px.distance(center);
                            let dist2 = pos2_px.distance(center);
                            let occluded1 = *rot_z1 < 0.0 && dist1 < earth_radius_px;
                            let occluded2 = *rot_z2 < 0.0 && dist2 < earth_radius_px;
                            
                            let link_stroke = if occluded1 || occluded2 {
                                egui::Stroke::new(1.0, color.linear_multiply(0.12))
                            } else {
                                egui::Stroke::new(1.0, color.linear_multiply(0.4))
                            };
                            
                            painter.line_segment([*pos1_px, *pos2_px], link_stroke);
                        }
                    }
                }
            }

            // Draw active laser links between Satellites and their best Ground Station (SGL)
            if self.show_sgl {
                for (_sat_id, _type, sat_pos_px, sat_r, sat_rot_z) in &satellites_screen {
                    let sat_max_speed = match _type {
                        OrbitType::LEO => 100.0,
                        OrbitType::MEO => 400.0,
                        OrbitType::GEO => 800.0,
                    };

                    let mut best_gs: Option<String> = None;
                    let mut max_capacity = 0.0;
                    let mut best_gs_pos_px = egui::pos2(0.0, 0.0);
                    let mut best_gs_rot_z = 0.0;
                    let sat_ref_dist = match _type {
                        OrbitType::LEO => self.config.ref_dist_sgl_km,
                        OrbitType::MEO => self.config.meo_alt_km,
                        OrbitType::GEO => self.config.geo_alt_km,
                    };

                    for (gs_id, gs_pos_px, gs_r, gs_k, gs_rot_z) in &stations_screen {
                        let capacity = compute_link_capacity(
                            *sat_r, *gs_r, true, *gs_k,
                            sat_ref_dist, sat_max_speed, &self.config.env
                        ).min(sat_max_speed);

                        if capacity > max_capacity {
                            max_capacity = capacity;
                            best_gs = Some(gs_id.clone());
                            best_gs_pos_px = *gs_pos_px;
                            best_gs_rot_z = *gs_rot_z;
                        }
                    }

                    if best_gs.is_some() && max_capacity > 0.0 {
                        let (beam_r, beam_g, beam_b) = if max_capacity > (sat_max_speed * 0.5) {
                            (0u8, 255u8, 170u8)
                        } else if max_capacity > (sat_max_speed * 0.1) {
                            (255u8, 200u8, 0u8)
                        } else {
                            (255u8, 60u8, 60u8)
                        };

                        let sat_dist = sat_pos_px.distance(center);
                        let sat_occluded = *sat_rot_z < 0.0 && sat_dist < earth_radius_px;
                        let gs_occluded = best_gs_rot_z <= 0.0;
                        
                        let base_alpha = if sat_occluded || gs_occluded { 15 } else { 255 };
                        let glow1_alpha = if sat_occluded || gs_occluded { 5 } else { 25 };
                        let glow2_alpha = if sat_occluded || gs_occluded { 10 } else { 60 };

                        let base_color = egui::Color32::from_rgba_unmultiplied(beam_r, beam_g, beam_b, base_alpha);

                        // Outer glow
                        painter.line_segment(
                            [*sat_pos_px, best_gs_pos_px],
                            egui::Stroke::new(5.0, egui::Color32::from_rgba_unmultiplied(beam_r, beam_g, beam_b, glow1_alpha))
                        );
                        // Mid glow
                        painter.line_segment(
                            [*sat_pos_px, best_gs_pos_px],
                            egui::Stroke::new(2.5, egui::Color32::from_rgba_unmultiplied(beam_r, beam_g, beam_b, glow2_alpha))
                        );
                        // Core laser line
                        painter.line_segment(
                            [*sat_pos_px, best_gs_pos_px],
                            egui::Stroke::new(1.0, base_color)
                        );

                        // Speed label at midpoint
                        let mid = egui::pos2(
                            (sat_pos_px.x + best_gs_pos_px.x) / 2.0,
                            (sat_pos_px.y + best_gs_pos_px.y) / 2.0,
                        );
                        let label = format!("{:.1} Gbps", max_capacity);
                        painter.text(
                            egui::pos2(mid.x + 5.0, mid.y - 6.0),
                            egui::Align2::LEFT_BOTTOM,
                            &label,
                            egui::FontId::proportional(9.0),
                            base_color,
                        );
                    }
                }
            }

            // Draw Ground Stations
            for (gs_id, gs_pos_px, _gs_r, gs_k, rot_z) in &stations_screen {
                if *rot_z <= 0.0 {
                    continue; // behind Earth
                }
                let color = if *gs_k < 0.1 / 1000.0 {
                    egui::Color32::from_rgb(34, 197, 94)
                } else if *gs_k < 1.0 / 1000.0 {
                    egui::Color32::from_rgb(234, 179, 8)
                } else {
                    egui::Color32::from_rgb(239, 68, 68)
                };
                
                painter.rect_filled(
                    egui::Rect::from_center_size(*gs_pos_px, egui::vec2(8.0, 8.0)),
                    1.0,
                    color
                );
                
                painter.text(
                    egui::pos2(gs_pos_px.x + 8.0, gs_pos_px.y - 4.0),
                    egui::Align2::LEFT_TOP,
                    gs_id,
                    egui::FontId::proportional(10.0),
                    egui::Color32::LIGHT_GRAY
                );
            }

            // Draw Satellites on top
            for (sat_id, _type, sat_pos_px, _r, rot_z) in &satellites_screen {
                let color = match _type {
                    OrbitType::LEO => egui::Color32::from_rgb(56, 189, 248),
                    OrbitType::MEO => egui::Color32::from_rgb(192, 132, 252),
                    OrbitType::GEO => egui::Color32::from_rgb(251, 146, 60),
                };

                let is_selected = *sat_id == self.selected_satellite_id;
                let size = if is_selected { 6.0 } else { 4.0 };
                
                // Occlusion check
                let dist_from_center = sat_pos_px.distance(center);
                let occluded = *rot_z < 0.0 && dist_from_center < earth_radius_px;
                
                let alpha = if occluded { 40 } else { 255 };
                let color_with_alpha = color.linear_multiply(alpha as f32 / 255.0);

                if is_selected {
                    let ring_alpha = if occluded { 60 } else { 255 };
                    painter.circle_stroke(
                        *sat_pos_px,
                        size + 3.0,
                        egui::Stroke::new(1.5, egui::Color32::from_rgb(250, 204, 21).linear_multiply(ring_alpha as f32 / 255.0))
                    );
                }

                painter.circle_filled(*sat_pos_px, size, color_with_alpha);

                if is_selected || satellites_screen.len() <= 20 {
                    let text_color = if is_selected {
                        egui::Color32::from_rgb(250, 204, 21).linear_multiply(alpha as f32 / 255.0)
                    } else {
                        egui::Color32::WHITE.linear_multiply(alpha as f32 / 255.0)
                    };
                    painter.text(
                        egui::pos2(sat_pos_px.x + size + 2.0, sat_pos_px.y - 4.0),
                        egui::Align2::LEFT_TOP,
                        sat_id,
                        egui::FontId::proportional(10.0),
                        text_color
                    );
                }
            }
        });

        // Apply deferred mutations to avoid index mismatches during UI drawing
        if let Some(idx) = pending_remove {
            let name = self.ground_stations[idx].name.clone();
            self.ground_stations.remove(idx);
            if idx < self.weather_overrides.len() {
                self.weather_overrides.remove(idx);
            }
            if idx < self.history_stations.len() {
                self.history_stations.remove(idx);
            }
            self.log(&format!("Rimossa stazione {}", name));
        }
        if pending_add {
            let new_id = format!("GS_{}", self.ground_stations.len() + 1);
            let new_name = format!("Station {}", self.ground_stations.len() + 1);
            self.ground_stations.push(GroundStation {
                id: new_id.clone(),
                name: new_name.clone(),
                lat_rad: 0.0,
                lon_rad: 0.0,
                alt_m: 100.0,
                downlink_nominal_gbps: 10.0,
                atmos_state: 0,
                k_value: self.config.atmos_k[0] / 1000.0,
            });
            self.weather_overrides.push(Some(0));
            self.history_stations.push(vec![0.0f32; self.history_time.len()]);
            self.log(&format!("Aggiunta stazione {}", new_name));
        }
        if pending_reset {
            self.current_time = 0.0;
            self.is_running = true;
            self.time_warp = 1;
            self.selected_satellite_id = "LEO_00".to_string();
            self.constellation = create_satellites_from_config(&self.config);
            self.ground_stations = self.config.stations.clone();
            self.weather_overrides = vec![Some(0); self.ground_stations.len()];
            self.history_time.clear();
            self.history_stations = vec![Vec::new(); self.ground_stations.len()];
            self.history_total.clear();
            self.map_zoom = 1.0;
            self.log("Simulation State Reset to initial values");
        }
    }
}

fn main() -> Result<(), eframe::Error> {
    println!("=== Starting HydRON-DT-Builder Interactive GUI Monitor ===");

    let config_path = "config.toml";
    let config = match load_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            println!("Warning: config.toml could not be loaded: {}. Loading defaults.", e);
            Config {
                name: "HydRON-Like-Net".to_string(),
                leo_num: 10,
                leo_alt_km: 550.0,
                leo_inc_deg: 97.6,
                leo_mass: 20.0,
                leo_area: 0.1,
                leo_cd: 2.2,
                leo_cr: 1.2,
                meo_num: 4,
                meo_alt_km: 10000.0,
                meo_inc_deg: 55.0,
                meo_raans: vec![0.0, 90.0, 180.0, 270.0],
                meo_mass: 50.0,
                meo_area: 0.25,
                meo_cd: 0.0,
                meo_cr: 1.2,
                geo_num: 3,
                geo_lons: vec![0.0, 60.0, -120.0],
                geo_alt_km: 35786.0,
                geo_inc_deg: 0.0,
                geo_mass: 200.0,
                geo_area: 1.5,
                geo_cd: 0.0,
                geo_cr: 1.2,
                stations: vec![
                    GroundStation { id: "GS_SVA".to_string(), name: "Svalbard".to_string(), lat_rad: 78.2307f64.to_radians(), lon_rad: 15.6472f64.to_radians(), alt_m: 130.0, downlink_nominal_gbps: 10.0, atmos_state: 0, k_value: 0.05 / 1000.0 },
                    GroundStation { id: "GS_ZRH".to_string(), name: "Zurich".to_string(), lat_rad: 47.4647f64.to_radians(), lon_rad:  8.5492f64.to_radians(), alt_m: 400.0, downlink_nominal_gbps: 10.0, atmos_state: 0, k_value: 0.05 / 1000.0 },
                    GroundStation { id: "GS_REU".to_string(), name: "Reunion".to_string(), lat_rad: -20.9089f64.to_radians(), lon_rad: 55.5136f64.to_radians(), alt_m: 95.0, downlink_nominal_gbps: 10.0, atmos_state: 0, k_value: 0.05 / 1000.0 },
                ],
                atmos_states: vec!["clear".to_string(), "thin".to_string(), "thick".to_string(), "heavy".to_string()],
                atmos_k: vec![0.05, 0.2, 1.5, 5.0],
                transition_matrix: vec![
                    vec![0.85, 0.10, 0.04, 0.01],
                    vec![0.15, 0.70, 0.10, 0.05],
                    vec![0.05, 0.15, 0.65, 0.15],
                    vec![0.02, 0.08, 0.20, 0.70],
                ],
                env: SimEnvironment {
                    mu: 3.986004418e14,
                    r_earth: 6378137.0,
                    j2: 1.08262668e-3,
                    rho0_500km: 3.8e-12,
                    h0_km: 500.0,
                    scale_height_km: 70.0,
                    p_srp: 4.56e-6,
                },
                dt_time_step: 1.0,
                ref_dist_isl_km: 1000.0,
                ref_dist_sgl_km: 1000.0,
            }
        }
    };



    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("HydRON Constellation Digital Twin Monitor")
            .with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };

    eframe::run_native(
        "HydRON Constellation Digital Twin Monitor",
        native_options,
        Box::new(|cc| Box::new(HydronGuiApp::new(cc, config))),
    )
}
