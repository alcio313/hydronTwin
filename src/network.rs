use crate::math::normalize;
use crate::math::{norm, dot};
use crate::models::SimEnvironment;

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
        let disc: f64 = b * b - 4.0 * c;
        
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

