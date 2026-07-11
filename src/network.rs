use crate::math::normalize;
use crate::math::{norm, dot};
use crate::models::SimEnvironment;
use std::collections::HashMap;

/// One satellite as a node in the ground-reach routing graph.
pub struct RouteNode {
    /// MEO/GEO relays can forward traffic for other satellites; LEO terminals cannot.
    pub is_relay: bool,
    /// Maximum bitrate the satellite's payload can handle (Gbps).
    pub max_cap: f64,
    /// Reference distance used for the space-to-ground (SGL) free-space loss model.
    pub sgl_ref_dist: f64,
    /// Reference distance used for the inter-satellite (ISL) free-space loss model.
    pub isl_ref_dist: f64,
    /// ECI position (m).
    pub r: [f64; 3],
    /// Pointing-loss multiplier in [0, 1] from the satellite's ADCS attitude error.
    pub point_factor: f64,
}

/// One ground station as seen by the routing pass.
pub struct GroundNode {
    /// ECI position (m).
    pub r: [f64; 3],
    /// Atmospheric attenuation coefficient (1/m).
    pub k_value: f64,
    /// Aggregate downlink capacity (Gbps); may be infinite.
    pub capacity: f64,
    /// Minimum elevation (rad) below which optical links are unusable.
    pub min_elev_rad: f64,
}

/// Output of one routing/allocation pass over the current geometry.
pub struct RoutingResult {
    /// Traffic actually landing at each ground station (Gbps); never exceeds its capacity.
    pub gs_throughputs: Vec<f64>,
    pub total_throughput: f64,
    /// Active space-to-ground links: (sat_idx, gs_idx, allocated Gbps).
    pub sgl_links: Vec<(usize, usize, f64)>,
    /// Active inter-satellite links: (sat_i, sat_j, Gbps). For LEO first hops this
    /// is the allocated flow; for the relay backbone it is the available capacity.
    pub isl_links: Vec<(usize, usize, f64)>,
    /// Per-satellite bitrate of *own* traffic delivered to the ground network.
    /// Summing this vector gives `total_throughput`.
    pub sat_ground_rate: Vec<f64>,
    /// Per-satellite bitrate carried by the node: own traffic for terminals,
    /// own + forwarded/transited traffic for relays (payload utilization).
    pub sat_carried_rate: Vec<f64>,
}

/// Capacity of the ISL between nodes a and b under the link model, including
/// both endpoints' pointing losses. Links involving a LEO run at the LEO's
/// stable terminal capacity (no free-space distance attenuation).
fn isl_capacity(nodes: &[RouteNode], a: usize, b: usize, env: &SimEnvironment) -> f64 {
    if !visible(nodes[a].r, nodes[b].r, env.r_earth) {
        return 0.0;
    }
    let pf = nodes[a].point_factor * nodes[b].point_factor;
    let leo_a = !nodes[a].is_relay;
    let leo_b = !nodes[b].is_relay;
    if leo_a || leo_b {
        // Advanced LEO laser terminal: stable configured capacity.
        let leo_cap = match (leo_a, leo_b) {
            (true, true) => nodes[a].max_cap.min(nodes[b].max_cap),
            (true, false) => nodes[a].max_cap,
            _ => nodes[b].max_cap,
        };
        return leo_cap * pf;
    }
    let nominal = nodes[a].max_cap.min(nodes[b].max_cap);
    compute_link_capacity(nodes[a].r, nodes[b].r, false, 0.0, nodes[a].isl_ref_dist, nominal, env) * pf
}

/// Physical SGL link capacity between a satellite and a ground station,
/// including the satellite's pointing loss, its payload cap, and the
/// station's minimum elevation mask.
fn sgl_capacity(node: &RouteNode, gs: &GroundNode, env: &SimEnvironment) -> f64 {
    // Elevation of the satellite as seen from the station:
    // sin(el) = up_hat · range_hat, with up along the station's zenith.
    let d = [node.r[0] - gs.r[0], node.r[1] - gs.r[1], node.r[2] - gs.r[2]];
    let d_len = norm(d);
    let gs_len = norm(gs.r);
    if d_len <= 0.0 || gs_len <= 0.0 {
        return 0.0;
    }
    let sin_el = dot(gs.r, d) / (gs_len * d_len);
    if sin_el < gs.min_elev_rad.sin() {
        return 0.0;
    }
    compute_link_capacity(node.r, gs.r, true, gs.k_value, node.sgl_ref_dist, node.max_cap, env)
        .min(node.max_cap)
        * node.point_factor
}

/// Route the whole constellation to the ground with capacity accounting:
///
/// - Phase A: each MEO/GEO relay downlinks its own traffic to the best reachable
///   station, capped by the station's residual capacity and its own payload cap.
/// - Phase B: the visible relay-relay backbone is reported for display.
/// - Phase C: each LEO is allocated sequentially (best direct SGL first in the
///   ordering) on the *residual* widest path — direct SGL or via relay chains —
///   decrementing every resource along the chosen path: ground station residual,
///   each transited relay's payload residual, the exit relay's SGL headroom, and
///   ISL link usage. A LEO keeps its 1-terminal budget: one SGL or one ISL.
///
/// Ground station throughputs therefore never exceed their nominal capacity and
/// N terminals sharing a relay split its capacity instead of multiplying it.
pub fn route_network(
    nodes: &[RouteNode],
    gs: &[GroundNode],
    prioritize_relay: bool,
    env: &SimEnvironment,
) -> RoutingResult {
    let n = nodes.len();
    let m = gs.len();

    let mut gs_residual: Vec<f64> = gs.iter().map(|g| g.capacity).collect();
    let mut relay_residual: Vec<f64> = nodes.iter().map(|s| s.max_cap).collect();
    let mut sgl_exit_residual = vec![0.0_f64; n];
    let mut exit_flow = vec![0.0_f64; n];
    let mut relay_gs = vec![usize::MAX; n];
    let mut isl_used: HashMap<(usize, usize), f64> = HashMap::new();

    let mut result = RoutingResult {
        gs_throughputs: vec![0.0; m],
        total_throughput: 0.0,
        sgl_links: Vec::new(),
        isl_links: Vec::new(),
        sat_ground_rate: vec![0.0; n],
        sat_carried_rate: vec![0.0; n],
    };

    // --- Phase A: point each relay's SGL terminal at its best station ---
    // The relay's own traffic is NOT allocated yet: forwarded LEO traffic gets
    // priority on the shared SGL pipe, and the relay fills what is left (Phase D).
    for i in 0..n {
        if !nodes[i].is_relay {
            continue;
        }
        let mut best_j = usize::MAX;
        let mut best_link = 0.0_f64;
        let mut best_score = 0.0_f64;
        for j in 0..m {
            let cap = sgl_capacity(&nodes[i], &gs[j], env);
            let score = cap.min(gs_residual[j]);
            if cap > 0.0 && (score > best_score || (best_j == usize::MAX && cap > best_link)) {
                best_j = j;
                best_link = cap;
                best_score = score;
            }
        }
        if best_j == usize::MAX {
            continue;
        }
        relay_gs[i] = best_j;
        sgl_exit_residual[i] = best_link;
    }

    // --- Phase B: relay-relay backbone (display capacity) ---
    for i in 0..n {
        for j in (i + 1)..n {
            if !nodes[i].is_relay || !nodes[j].is_relay {
                continue;
            }
            let cap = isl_capacity(nodes, i, j, env);
            if cap > 0.0 {
                result.isl_links.push((i, j, cap));
            }
        }
    }

    // --- Phase C: sequential LEO allocation on residual capacities ---
    let mut leo_order: Vec<usize> = (0..n).filter(|&i| !nodes[i].is_relay).collect();
    let mut leo_best_direct: Vec<(usize, f64)> = vec![(usize::MAX, 0.0); n];
    for &i in &leo_order {
        let mut best_j = usize::MAX;
        let mut best = 0.0;
        for j in 0..m {
            let cap = sgl_capacity(&nodes[i], &gs[j], env);
            if cap > best {
                best = cap;
                best_j = j;
            }
        }
        leo_best_direct[i] = (best_j, best);
    }
    // Strongest direct link first (proxy of the previous greedy candidate sort).
    leo_order.sort_by(|&a, &b| {
        leo_best_direct[b].1
            .partial_cmp(&leo_best_direct[a].1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    for &i in &leo_order {
        // Direct option on residual station capacity.
        let mut direct_j = usize::MAX;
        let mut direct_val = 0.0_f64;
        if !prioritize_relay {
            for j in 0..m {
                let cap = sgl_capacity(&nodes[i], &gs[j], env).min(gs_residual[j]);
                if cap > direct_val {
                    direct_val = cap;
                    direct_j = j;
                }
            }
        }

        // Relay option: residual widest path with path tracking.
        // reach[b] = best bottleneck from relay b to the ground on current residuals.
        let mut reach = vec![0.0_f64; n];
        let mut next_hop = vec![usize::MAX; n];
        for b in 0..n {
            if nodes[b].is_relay && relay_gs[b] != usize::MAX {
                reach[b] = sgl_exit_residual[b]
                    .min(gs_residual[relay_gs[b]])
                    .min(relay_residual[b]);
            }
        }
        // Bellman-Ford-style relaxation over relay-relay hops.
        for _ in 0..=n {
            let mut changed = false;
            for a in 0..n {
                if !nodes[a].is_relay {
                    continue;
                }
                for b in 0..n {
                    if a == b || !nodes[b].is_relay || reach[b] <= 0.0 {
                        continue;
                    }
                    let key = (a.min(b), a.max(b));
                    let residual_isl =
                        isl_capacity(nodes, a, b, env) - isl_used.get(&key).copied().unwrap_or(0.0);
                    let via = relay_residual[a].min(residual_isl).min(reach[b]);
                    if via > reach[a] + 1e-9 {
                        reach[a] = via;
                        next_hop[a] = b;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        // First hop from the LEO into the relay graph.
        let mut relay_val = 0.0_f64;
        let mut first_hop = usize::MAX;
        for b in 0..n {
            if !nodes[b].is_relay || reach[b] <= 0.0 {
                continue;
            }
            let key = (i.min(b), i.max(b));
            let residual_isl =
                isl_capacity(nodes, i, b, env) - isl_used.get(&key).copied().unwrap_or(0.0);
            let via = nodes[i].max_cap.min(residual_isl).min(reach[b]);
            if via > relay_val + 1e-9 {
                relay_val = via;
                first_hop = b;
            }
        }

        // One laser terminal per LEO: pick the better of the two options.
        if relay_val > direct_val && first_hop != usize::MAX {
            let alloc = relay_val;
            if alloc <= 0.0 {
                continue;
            }
            // Walk the relay chain, consuming residuals hop by hop.
            *isl_used.entry((i.min(first_hop), i.max(first_hop))).or_insert(0.0) += alloc;
            result.isl_links.push((i, first_hop, alloc));
            let mut b = first_hop;
            loop {
                relay_residual[b] -= alloc;
                let nb = next_hop[b];
                if nb == usize::MAX {
                    // Exit relay: consume its SGL headroom and the station residual.
                    let j = relay_gs[b];
                    sgl_exit_residual[b] -= alloc;
                    exit_flow[b] += alloc;
                    gs_residual[j] -= alloc;
                    result.gs_throughputs[j] += alloc;
                    break;
                }
                *isl_used.entry((b.min(nb), b.max(nb))).or_insert(0.0) += alloc;
                b = nb;
            }
            result.total_throughput += alloc;
            result.sat_ground_rate[i] = alloc;
        } else if direct_j != usize::MAX && direct_val > 0.0 {
            let alloc = direct_val;
            gs_residual[direct_j] -= alloc;
            result.gs_throughputs[direct_j] += alloc;
            result.total_throughput += alloc;
            result.sat_ground_rate[i] = alloc;
            result.sgl_links.push((i, direct_j, alloc));
        }
    }

    // --- Phase D: relays downlink their own traffic on the leftover pipe ---
    for i in 0..n {
        if relay_gs[i] == usize::MAX {
            continue;
        }
        let j = relay_gs[i];
        let alloc = sgl_exit_residual[i]
            .min(relay_residual[i])
            .min(gs_residual[j]);
        if alloc > 0.0 {
            sgl_exit_residual[i] -= alloc;
            relay_residual[i] -= alloc;
            gs_residual[j] -= alloc;
            result.gs_throughputs[j] += alloc;
            result.total_throughput += alloc;
            result.sat_ground_rate[i] += alloc;
        }
        // The SGL entry carries the relay's own traffic plus everything it
        // forwarded for other satellites (they share the same physical beam).
        let link_total = alloc + exit_flow[i];
        if link_total > 0.0 {
            result.sgl_links.push((i, j, link_total));
        }
    }

    // Carried rate: payload utilization for relays, own delivery for terminals.
    for i in 0..n {
        result.sat_carried_rate[i] = if nodes[i].is_relay {
            nodes[i].max_cap - relay_residual[i]
        } else {
            result.sat_ground_rate[i]
        };
    }

    result
}

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

#[cfg(test)]
mod tests {
    use super::*;

    const R_EARTH: f64 = 6378137.0;

    fn test_env() -> SimEnvironment {
        SimEnvironment {
            mu: 3.986004418e14,
            r_earth: R_EARTH,
            j2: 0.0,
            rho0_500km: 0.0,
            h0_km: 500.0,
            scale_height_km: 70.0,
            p_srp: 0.0,
        }
    }

    fn node(r: [f64; 3], is_relay: bool, max_cap: f64) -> RouteNode {
        RouteNode {
            is_relay,
            max_cap,
            // Reference distances comparable to the test geometries, so the
            // free-space model does not crush the capacities.
            sgl_ref_dist: 10_000.0,
            isl_ref_dist: 10_000.0,
            r,
            point_factor: 1.0,
        }
    }

    fn gs_at_x(capacity: f64) -> GroundNode {
        GroundNode { r: [R_EARTH, 0.0, 0.0], k_value: 0.0, capacity, min_elev_rad: 0.0 }
    }

    #[test]
    fn visibility_basics() {
        // Two satellites on the same side of Earth: visible
        let a = [R_EARTH + 1e6, 0.0, 0.0];
        let b = [R_EARTH + 1e6, 2e6, 0.0];
        assert!(visible(a, b, R_EARTH));
        // Opposite sides: Earth blocks the segment
        let c = [-(R_EARTH + 1e6), 0.0, 0.0];
        assert!(!visible(a, c, R_EARTH));
        // Satellite straight above the GS: visible
        assert!(visible_sgl(a, [R_EARTH, 0.0, 0.0], R_EARTH));
        // Satellite behind the planet from the GS
        assert!(!visible_sgl(c, [R_EARTH, 0.0, 0.0], R_EARTH));
    }

    #[test]
    fn elevation_mask_blocks_low_links() {
        let env = test_env();
        let mut gs = gs_at_x(f64::INFINITY);
        gs.min_elev_rad = 10.0_f64.to_radians();
        // Satellite ~8.5° above the station's horizon: geometrically visible
        // but below the 10° mask.
        let low_sat = node([R_EARTH + 300_000.0, 2_000_000.0, 0.0], false, 40.0);
        let res = route_network(&[low_sat], &[gs], false, &env);
        assert_eq!(res.sat_ground_rate[0], 0.0, "link below the elevation mask must be unusable");

        // Same satellite straight overhead: well above the mask.
        let mut gs = gs_at_x(f64::INFINITY);
        gs.min_elev_rad = 10.0_f64.to_radians();
        let high_sat = node([R_EARTH + 300_000.0, 0.0, 0.0], false, 40.0);
        let res = route_network(&[high_sat], &[gs], false, &env);
        assert!(res.sat_ground_rate[0] > 0.0, "overhead link must pass the mask");
    }

    #[test]
    fn gs_capacity_is_enforced() {
        // Two LEOs straight above one station whose capacity is below their sum.
        let env = test_env();
        let nodes = vec![
            node([R_EARTH + 550_000.0, 0.0, 0.0], false, 8.0),
            node([R_EARTH + 560_000.0, 10_000.0, 0.0], false, 8.0),
        ];
        let gs = vec![gs_at_x(10.0)];
        let res = route_network(&nodes, &gs, false, &env);
        assert!(res.gs_throughputs[0] <= 10.0 + 1e-9, "gs throughput = {}", res.gs_throughputs[0]);
        // Both should still get something: the second LEO takes the leftover.
        assert!(res.sat_ground_rate.iter().filter(|&&r| r > 0.0).count() == 2);
        assert!((res.gs_throughputs[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn relay_capacity_is_shared_not_multiplied() {
        // Three LEOs that can only reach the ground via one MEO relay (relay-only mode).
        let env = test_env();
        let alt_leo = R_EARTH + 550_000.0;
        let nodes = vec![
            node([alt_leo, 0.0, 0.0], false, 15.0),
            node([alt_leo * 0.9, alt_leo * 0.4, 0.0], false, 15.0),
            node([alt_leo * 0.9, -alt_leo * 0.4, 0.0], false, 15.0),
            node([R_EARTH + 10_000_000.0, 0.0, 0.0], true, 20.0),
        ];
        let gs = vec![gs_at_x(f64::INFINITY)];
        let res = route_network(&nodes, &gs, true, &env);
        let forwarded: f64 = res.sat_ground_rate[..3].iter().sum();
        // The relay's own downlink plus everything it forwards must fit its 20 Gbps payload.
        let relay_total = forwarded + res.sat_ground_rate[3];
        assert!(relay_total <= 20.0 + 1e-6, "relay carries {relay_total} Gbps > its 20 Gbps cap");
        assert!(forwarded > 0.0, "no LEO traffic was forwarded at all");
        assert!((res.gs_throughputs[0] - relay_total).abs() < 1e-6);
    }

    #[test]
    fn multi_hop_chain_reaches_ground() {
        // LEO → MEO → GEO → GS: the middle relay has no ground link of its own.
        let env = test_env();
        let r_geo = R_EARTH + 35_786_000.0;
        let nodes = vec![
            // LEO on the far side relative to the GS but visible to the MEO
            node([0.0, R_EARTH + 550_000.0, 0.0], false, 50.0),
            // MEO above the LEO, no line of sight to the GS (placed opposite)
            node([0.0, R_EARTH + 10_000_000.0, 0.0], true, 30.0),
            // GEO that sees both the MEO and the station
            node([r_geo * 0.7, r_geo * 0.7, 0.0], true, 25.0),
        ];
        let gs = vec![gs_at_x(f64::INFINITY)];
        let res = route_network(&nodes, &gs, true, &env);
        assert!(res.sat_ground_rate[0] > 0.0, "LEO could not reach the ground through the chain");
        // The chain bottleneck can never exceed any relay payload on the path.
        assert!(res.sat_ground_rate[0] <= 25.0 + 1e-9);
    }

    #[test]
    fn pointing_loss_halves_link_capacity() {
        let env = test_env();
        let gs = vec![gs_at_x(f64::INFINITY)];
        let mut nodes = vec![node([R_EARTH + 1_000_000.0, 0.0, 0.0], false, 40.0)];
        let full = route_network(&nodes, &gs, false, &env).sat_ground_rate[0];
        assert!(full > 0.0);
        nodes[0].point_factor = 0.5;
        let half = route_network(&nodes, &gs, false, &env).sat_ground_rate[0];
        assert!((half - full * 0.5).abs() < 1e-9, "full {full}, degraded {half}");
    }

    #[test]
    fn widest_path_prefers_faster_relay() {
        // Two relays both reaching the ground; the LEO should route via the wider one.
        let env = test_env();
        let alt = R_EARTH + 550_000.0;
        let nodes = vec![
            node([alt, 0.0, 0.0], false, 100.0),
            node([R_EARTH + 10_000_000.0, 3_000_000.0, 0.0], true, 5.0),
            node([R_EARTH + 10_000_000.0, -3_000_000.0, 0.0], true, 50.0),
        ];
        let gs = vec![gs_at_x(f64::INFINITY)];
        let res = route_network(&nodes, &gs, true, &env);
        // Flow must traverse the 50 Gbps relay: its residual consumption shows up
        // as the LEO's first hop in isl_links.
        let leo_hop = res.isl_links.iter().find(|(a, b, _)| *a == 0 || *b == 0);
        let (a, b, flow) = leo_hop.expect("LEO got no ISL");
        let partner = if *a == 0 { *b } else { *a };
        assert_eq!(partner, 2, "LEO routed via the narrow relay");
        assert!(*flow > 5.0, "flow {flow} did not exceed the narrow relay's capacity");
    }
}
