use crate::math::normalize;
use crate::math::{norm, dot};
use crate::models::SimEnvironment;
use std::collections::HashMap;

/// One satellite as a node in the ground-reach routing graph.
pub struct RouteNode {
    /// Stable identifier (satellite id) used by the persistent link memory.
    pub id: String,
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
    /// Stable identifier (station id) used by the persistent link memory.
    pub id: String,
    /// ECI position (m).
    pub r: [f64; 3],
    /// Atmospheric attenuation coefficient (1/m).
    pub k_value: f64,
    /// Aggregate downlink capacity (Gbps); may be infinite.
    pub capacity: f64,
    /// Minimum elevation (rad) below which optical links are unusable.
    pub min_elev_rad: f64,
}

/// Routing policy knobs.
pub struct RouteParams {
    /// LEO terminals skip direct ground links and route via relays only.
    pub prioritize_relay: bool,
    /// A link is abandoned only when the best alternative is at least this
    /// factor better (e.g. 1.3 = 30% better) — or when the link is lost.
    pub hysteresis: f64,
    /// Pointing/acquisition time (s) during which a newly established laser
    /// link carries no traffic.
    pub acquisition_time_s: f64,
    /// Minimum dwell time (s) after a handover before the terminal may hand
    /// over again *voluntarily*. A physically dead link (occluded / below the
    /// elevation mask) can always be abandoned. This prevents ping-pong when
    /// link qualities fluctuate quickly (e.g. weather transitions).
    pub min_dwell_s: f64,
}

/// What a satellite's laser terminal is currently pointed at.
#[derive(Clone, PartialEq)]
pub enum LinkTarget {
    Ground(String),
    Sat(String),
}

/// A currently established (or still acquiring) link.
pub struct LinkState {
    pub target: LinkTarget,
    /// Remaining acquisition time (s); the link carries no traffic until 0.
    pub acquiring: f64,
    /// Remaining minimum-dwell time (s); voluntary handovers are blocked until 0.
    pub cooldown: f64,
}

/// Per-satellite link state persisted between routing passes: this is what
/// gives links inertia (hysteresis) and acquisition delays across frames.
/// Clear it on reset, config import, or rewind.
#[derive(Default)]
pub struct LinkMemory {
    pub links: HashMap<String, LinkState>,
}

impl LinkMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.links.clear();
    }
}

/// Output of one routing/allocation pass over the current geometry.
pub struct RoutingResult {
    /// Traffic actually landing at each ground station (Gbps); never exceeds its capacity.
    pub gs_throughputs: Vec<f64>,
    pub total_throughput: f64,
    /// Active space-to-ground links: (sat_idx, gs_idx, allocated Gbps).
    /// Capacity 0 marks a link still in acquisition.
    pub sgl_links: Vec<(usize, usize, f64)>,
    /// Active inter-satellite links: (sat_i, sat_j, Gbps). For LEO first hops this
    /// is the allocated flow (0 while acquiring); for the relay backbone it is the
    /// available capacity.
    pub isl_links: Vec<(usize, usize, f64)>,
    /// Per-satellite bitrate of *own* traffic delivered to the ground network.
    /// Summing this vector gives `total_throughput`.
    pub sat_ground_rate: Vec<f64>,
    /// Per-satellite bitrate carried by the node: own traffic for terminals,
    /// own + forwarded/transited traffic for relays (payload utilization).
    pub sat_carried_rate: Vec<f64>,
    /// Human-readable handover / acquisition events from this pass.
    pub events: Vec<String>,
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

fn target_label(target: &LinkTarget) -> &str {
    match target {
        LinkTarget::Ground(id) | LinkTarget::Sat(id) => id.as_str(),
    }
}

/// Update the link memory for `sat_id` toward `new_target`. Returns the
/// remaining acquisition time for this pass (0 = link established and usable).
/// A target change (re)arms both the acquisition and the minimum-dwell timers.
fn update_link_state(
    memory: &mut LinkMemory,
    events: &mut Vec<String>,
    sat_id: &str,
    new_target: LinkTarget,
    params: &RouteParams,
) -> f64 {
    let (acquiring, cooldown) = match memory.links.get(sat_id) {
        Some(st) if st.target == new_target => (st.acquiring, st.cooldown),
        Some(st) => {
            events.push(format!(
                "{} handover: {} → {} (acquiring)",
                sat_id,
                target_label(&st.target),
                target_label(&new_target)
            ));
            (params.acquisition_time_s, params.acquisition_time_s + params.min_dwell_s)
        }
        None => {
            events.push(format!(
                "{} acquiring link → {}",
                sat_id,
                target_label(&new_target)
            ));
            (params.acquisition_time_s, params.acquisition_time_s + params.min_dwell_s)
        }
    };
    memory.links.insert(sat_id.to_string(), LinkState { target: new_target, acquiring, cooldown });
    acquiring
}

/// Route the whole constellation to the ground with capacity accounting and
/// link persistence:
///
/// - Phase A: each MEO/GEO relay keeps its current ground station unless the
///   best alternative beats it by the hysteresis factor (or the link is lost).
///   The relay's own traffic is NOT allocated yet: forwarded LEO traffic gets
///   priority on the shared SGL pipe, and the relay fills what is left (Phase D).
/// - Phase B: the visible relay-relay backbone is reported for display.
/// - Phase C: each LEO is allocated sequentially on the *residual* widest path,
///   with the same hysteresis rule protecting its current link. A LEO keeps its
///   1-terminal budget: one SGL or one ISL.
/// - Newly pointed links (first contact or handover) spend `acquisition_time_s`
///   of simulated time carrying zero traffic before becoming usable.
///
/// `dt` is the simulated time elapsed since the previous routing pass (0 when
/// paused); it drives the acquisition countdowns stored in `memory`.
pub fn route_network(
    nodes: &[RouteNode],
    gs: &[GroundNode],
    params: &RouteParams,
    memory: &mut LinkMemory,
    dt: f64,
    env: &SimEnvironment,
) -> RoutingResult {
    let n = nodes.len();
    let m = gs.len();

    let mut gs_residual: Vec<f64> = gs.iter().map(|g| g.capacity).collect();
    let mut relay_residual: Vec<f64> = nodes.iter().map(|s| s.max_cap).collect();
    let mut sgl_exit_residual = vec![0.0_f64; n];
    let mut exit_flow = vec![0.0_f64; n];
    let mut relay_gs = vec![usize::MAX; n];
    let mut relay_acquiring = vec![false; n];
    let mut isl_used: HashMap<(usize, usize), f64> = HashMap::new();

    let mut result = RoutingResult {
        gs_throughputs: vec![0.0; m],
        total_throughput: 0.0,
        sgl_links: Vec::new(),
        isl_links: Vec::new(),
        sat_ground_rate: vec![0.0; n],
        sat_carried_rate: vec![0.0; n],
        events: Vec::new(),
    };

    // --- Link memory upkeep: drop vanished satellites, advance the timers ---
    {
        let ids: std::collections::HashSet<&str> = nodes.iter().map(|s| s.id.as_str()).collect();
        memory.links.retain(|id, _| ids.contains(id.as_str()));
        for st in memory.links.values_mut() {
            st.acquiring = (st.acquiring - dt).max(0.0);
            st.cooldown = (st.cooldown - dt).max(0.0);
        }
    }

    // --- Phase A: point each relay's SGL terminal, with hysteresis ---
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

        // Keep the current station while it is physically alive and either the
        // dwell cooldown is running or no alternative is decisively better.
        let mut chosen_j = best_j;
        let mut chosen_link = best_link;
        if let Some(LinkState { target: LinkTarget::Ground(gs_id), cooldown, .. }) = memory.links.get(&nodes[i].id) {
            if let Some(cur_j) = gs.iter().position(|g| &g.id == gs_id) {
                let cur_cap = sgl_capacity(&nodes[i], &gs[cur_j], env);
                let cur_score = cur_cap.min(gs_residual[cur_j]);
                if cur_cap > 0.0
                    && (*cooldown > 0.0 || best_score <= params.hysteresis * cur_score)
                {
                    chosen_j = cur_j;
                    chosen_link = cur_cap;
                }
            }
        }

        if chosen_j == usize::MAX {
            memory.links.remove(&nodes[i].id);
            continue;
        }
        let acquiring = update_link_state(
            memory,
            &mut result.events,
            &nodes[i].id,
            LinkTarget::Ground(gs[chosen_j].id.clone()),
            params,
        );
        relay_gs[i] = chosen_j;
        relay_acquiring[i] = acquiring > 0.0;
        sgl_exit_residual[i] = if acquiring > 0.0 { 0.0 } else { chosen_link };
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
        // Residual widest path with path tracking over the relay graph.
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

        // Value of a candidate first hop for this LEO.
        let leo_isl_value = |b: usize, isl_used: &HashMap<(usize, usize), f64>, reach: &[f64]| -> f64 {
            if !nodes[b].is_relay || reach[b] <= 0.0 {
                return 0.0;
            }
            let key = (i.min(b), i.max(b));
            let residual_isl =
                isl_capacity(nodes, i, b, env) - isl_used.get(&key).copied().unwrap_or(0.0);
            nodes[i].max_cap.min(residual_isl).min(reach[b])
        };

        // Best alternatives on current residuals.
        let mut direct_j = usize::MAX;
        let mut direct_val = 0.0_f64;
        if !params.prioritize_relay {
            for j in 0..m {
                let cap = sgl_capacity(&nodes[i], &gs[j], env).min(gs_residual[j]);
                if cap > direct_val {
                    direct_val = cap;
                    direct_j = j;
                }
            }
        }
        let mut relay_val = 0.0_f64;
        let mut first_hop = usize::MAX;
        for b in 0..n {
            let via = leo_isl_value(b, &isl_used, &reach);
            if via > relay_val + 1e-9 {
                relay_val = via;
                first_hop = b;
            }
        }
        let best_alt_val = direct_val.max(relay_val);

        // Incumbent link: physical feasibility (line of sight, elevation mask)
        // is separate from the allocatable value — a link to a relay that is
        // momentarily without ground egress stays pointed at zero traffic
        // instead of being dropped and re-acquired.
        // (target, allocatable value, idx, cooldown active)
        let mut incumbent: Option<(LinkTarget, f64, usize, bool)> = None;
        if let Some(st) = memory.links.get(&nodes[i].id) {
            let cooling = st.cooldown > 0.0;
            match &st.target {
                LinkTarget::Ground(gs_id) if !params.prioritize_relay => {
                    if let Some(j) = gs.iter().position(|g| &g.id == gs_id) {
                        let cap = sgl_capacity(&nodes[i], &gs[j], env);
                        if cap > 0.0 {
                            incumbent = Some((st.target.clone(), cap.min(gs_residual[j]), j, cooling));
                        }
                    }
                }
                LinkTarget::Sat(sat_id) => {
                    if let Some(b) = nodes.iter().position(|s| &s.id == sat_id) {
                        if nodes[b].is_relay && isl_capacity(nodes, i, b, env) > 0.0 {
                            let v = leo_isl_value(b, &isl_used, &reach);
                            incumbent = Some((st.target.clone(), v, b, cooling));
                        }
                    }
                }
                _ => {}
            }
        }

        // Decision: hold the incumbent while its dwell cooldown runs or while
        // no alternative is decisively better; hand over otherwise. A dead
        // incumbent (no line of sight) is always abandoned.
        let keep_incumbent = match &incumbent {
            Some((_, v, _, cooling)) => {
                *cooling || best_alt_val <= params.hysteresis * v || best_alt_val <= 0.0
            }
            None => false,
        };
        let (target, value, idx) = if keep_incumbent {
            let (t, v, idx, _) = incumbent.expect("checked above");
            (t, v, idx)
        } else if relay_val > direct_val && first_hop != usize::MAX {
            (LinkTarget::Sat(nodes[first_hop].id.clone()), relay_val, first_hop)
        } else if direct_j != usize::MAX && direct_val > 0.0 {
            (LinkTarget::Ground(gs[direct_j].id.clone()), direct_val, direct_j)
        } else {
            memory.links.remove(&nodes[i].id);
            continue;
        };

        let acquiring = update_link_state(
            memory,
            &mut result.events,
            &nodes[i].id,
            target.clone(),
            params,
        );
        if acquiring > 0.0 || value <= 0.0 {
            // Terminal busy pointing, or link held with no allocatable
            // capacity right now (e.g. relay re-acquiring its own SGL):
            // visible on the map, no traffic.
            match target {
                LinkTarget::Ground(_) => result.sgl_links.push((i, idx, 0.0)),
                LinkTarget::Sat(_) => result.isl_links.push((i, idx, 0.0)),
            }
            continue;
        }

        match target {
            LinkTarget::Ground(_) => {
                gs_residual[idx] -= value;
                result.gs_throughputs[idx] += value;
                result.total_throughput += value;
                result.sat_ground_rate[i] = value;
                result.sgl_links.push((i, idx, value));
            }
            LinkTarget::Sat(_) => {
                // Walk the relay chain, consuming residuals hop by hop.
                let alloc = value;
                *isl_used.entry((i.min(idx), i.max(idx))).or_insert(0.0) += alloc;
                result.isl_links.push((i, idx, alloc));
                let mut b = idx;
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
            }
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
        // A zero-capacity entry marks a link still in acquisition.
        let link_total = alloc + exit_flow[i];
        if link_total > 0.0 || relay_acquiring[i] {
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
            id: String::new(), // assigned per-index by route()
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
        GroundNode { id: String::new(), r: [R_EARTH, 0.0, 0.0], k_value: 0.0, capacity, min_elev_rad: 0.0 }
    }

    fn params(prioritize_relay: bool) -> RouteParams {
        RouteParams { prioritize_relay, hysteresis: 1.3, acquisition_time_s: 0.0, min_dwell_s: 0.0 }
    }

    /// One-shot routing pass with fresh memory and zero acquisition time:
    /// equivalent to the pre-hysteresis behavior. Assigns unique ids.
    fn route(
        mut nodes: Vec<RouteNode>,
        mut gs: Vec<GroundNode>,
        prioritize_relay: bool,
        env: &SimEnvironment,
    ) -> RoutingResult {
        for (k, n) in nodes.iter_mut().enumerate() {
            n.id = format!("S{k}");
        }
        for (k, g) in gs.iter_mut().enumerate() {
            g.id = format!("G{k}");
        }
        let mut memory = LinkMemory::new();
        route_network(&nodes, &gs, &params(prioritize_relay), &mut memory, 0.0, env)
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
        let res = route(vec![low_sat], vec![gs], false, &env);
        assert_eq!(res.sat_ground_rate[0], 0.0, "link below the elevation mask must be unusable");

        // Same satellite straight overhead: well above the mask.
        let mut gs = gs_at_x(f64::INFINITY);
        gs.min_elev_rad = 10.0_f64.to_radians();
        let high_sat = node([R_EARTH + 300_000.0, 0.0, 0.0], false, 40.0);
        let res = route(vec![high_sat], vec![gs], false, &env);
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
        let res = route(nodes, gs, false, &env);
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
        let res = route(nodes, gs, true, &env);
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
        let res = route(nodes, gs, true, &env);
        assert!(res.sat_ground_rate[0] > 0.0, "LEO could not reach the ground through the chain");
        // The chain bottleneck can never exceed any relay payload on the path.
        assert!(res.sat_ground_rate[0] <= 25.0 + 1e-9);
    }

    #[test]
    fn pointing_loss_halves_link_capacity() {
        let env = test_env();
        let full = route(
            vec![node([R_EARTH + 1_000_000.0, 0.0, 0.0], false, 40.0)],
            vec![gs_at_x(f64::INFINITY)],
            false,
            &env,
        )
        .sat_ground_rate[0];
        assert!(full > 0.0);
        let mut degraded_node = node([R_EARTH + 1_000_000.0, 0.0, 0.0], false, 40.0);
        degraded_node.point_factor = 0.5;
        let half = route(vec![degraded_node], vec![gs_at_x(f64::INFINITY)], false, &env)
            .sat_ground_rate[0];
        assert!((half - full * 0.5).abs() < 1e-9, "full {full}, degraded {half}");
    }

    #[test]
    fn hysteresis_keeps_current_link_until_decisively_beaten() {
        let env = test_env();
        let p = params(false);
        let mut memory = LinkMemory::new();
        // Two stations 20° apart along the satellite's ground track. With a
        // 500 km reference distance the link capacity is distance-sensitive,
        // so moving along the arc shifts the balance between GA and GB.
        let gs_angle_b = 20.0_f64.to_radians();
        let mk_gs = || {
            vec![
                GroundNode { id: "GA".into(), r: [R_EARTH, 0.0, 0.0], k_value: 0.0, capacity: f64::INFINITY, min_elev_rad: 0.0 },
                GroundNode { id: "GB".into(), r: [R_EARTH * gs_angle_b.cos(), R_EARTH * gs_angle_b.sin(), 0.0], k_value: 0.0, capacity: f64::INFINITY, min_elev_rad: 0.0 },
            ]
        };
        let mk_sat = |arc_deg: f64| {
            let a = arc_deg.to_radians();
            let r_orb = R_EARTH + 800_000.0;
            let mut s = node([r_orb * a.cos(), r_orb * a.sin(), 0.0], false, 40.0);
            s.id = "LEO".into();
            s.sgl_ref_dist = 500.0;
            s
        };

        // Pass 1: overhead of GA → links GA.
        let res = route_network(&[mk_sat(0.0)], &mk_gs(), &p, &mut memory, 0.0, &env);
        assert_eq!(res.sgl_links[0].1, 0, "should link GA first");

        // Pass 2: slightly past the midpoint toward GB — GB is better, but by
        // less than the 30% hysteresis margin → stays on GA, no handover.
        let res = route_network(&[mk_sat(10.8)], &mk_gs(), &p, &mut memory, 10.0, &env);
        assert_eq!(res.sgl_links[0].1, 0, "hysteresis must keep GA");
        assert!(res.events.is_empty(), "no handover expected: {:?}", res.events);

        // Pass 3: overhead of GB → decisively better → handover.
        let res = route_network(&[mk_sat(20.0)], &mk_gs(), &p, &mut memory, 10.0, &env);
        assert_eq!(res.sgl_links[0].1, 1, "should hand over to GB");
        assert!(res.events.iter().any(|e| e.contains("handover")), "{:?}", res.events);
    }

    #[test]
    fn dwell_cooldown_blocks_pingpong() {
        let env = test_env();
        let p = RouteParams { prioritize_relay: false, hysteresis: 1.3, acquisition_time_s: 0.0, min_dwell_s: 30.0 };
        let mut memory = LinkMemory::new();
        // Two co-visible stations; which one is better is driven by weather (k_value).
        let mk_gs = |k_a: f64, k_b: f64| {
            vec![
                GroundNode { id: "GA".into(), r: [R_EARTH, 0.0, 0.0], k_value: k_a, capacity: f64::INFINITY, min_elev_rad: 0.0 },
                GroundNode { id: "GB".into(), r: [R_EARTH * 0.985, R_EARTH * 0.174, 0.0], k_value: k_b, capacity: f64::INFINITY, min_elev_rad: 0.0 },
            ]
        };
        let mk_sat = || {
            let mut s = node([R_EARTH + 800_000.0, 0.0, 0.0], false, 40.0);
            s.id = "LEO".into();
            s
        };

        // Pass 1: GA clear, GB stormy → links GA (dwell timer armed).
        let res = route_network(&[mk_sat()], &mk_gs(0.0, 5.0 / 1000.0), &p, &mut memory, 0.0, &env);
        assert_eq!(res.sgl_links[0].1, 0);

        // Weather flips: GA stormy, GB clear — decisively better, but the
        // dwell cooldown is running → the terminal must stay on GA.
        let res = route_network(&[mk_sat()], &mk_gs(5.0 / 1000.0, 0.0), &p, &mut memory, 5.0, &env);
        assert_eq!(res.sgl_links[0].1, 0, "cooldown must block the voluntary handover");
        assert!(res.events.is_empty(), "{:?}", res.events);

        // After the dwell expires the handover is allowed.
        let res = route_network(&[mk_sat()], &mk_gs(5.0 / 1000.0, 0.0), &p, &mut memory, 40.0, &env);
        assert_eq!(res.sgl_links[0].1, 1, "handover allowed after the dwell");
        assert!(res.events.iter().any(|e| e.contains("handover")));
    }

    #[test]
    fn held_isl_survives_relay_egress_outage() {
        let env = test_env();
        let p = RouteParams { prioritize_relay: true, hysteresis: 1.3, acquisition_time_s: 10.0, min_dwell_s: 60.0 };
        let mut memory = LinkMemory::new();
        let mk_world = |gs_visible: bool| {
            let mut leo = node([R_EARTH + 550_000.0, 0.0, 0.0], false, 15.0);
            leo.id = "LEO".into();
            let mut relay = node([R_EARTH + 10_000_000.0, 0.0, 0.0], true, 20.0);
            relay.id = "MEO".into();
            let gs_r = if gs_visible { [R_EARTH, 0.0, 0.0] } else { [-R_EARTH, 0.0, 0.0] };
            (vec![leo, relay], vec![GroundNode { id: "GA".into(), r: gs_r, k_value: 0.0, capacity: f64::INFINITY, min_elev_rad: 0.0 }])
        };

        // Establish LEO → MEO → GA. Acquisitions are sequential: first the
        // relay's SGL comes up, only then the LEO can start acquiring its ISL.
        let (nodes, gs) = mk_world(true);
        route_network(&nodes, &gs, &p, &mut memory, 0.0, &env);
        route_network(&nodes, &gs, &p, &mut memory, 15.0, &env);
        let res = route_network(&nodes, &gs, &p, &mut memory, 15.0, &env);
        assert!(res.sat_ground_rate[0] > 0.0, "chain should be up");

        // The relay loses its station (behind the planet): the LEO must HOLD
        // its ISL at zero traffic instead of dropping the link.
        let (nodes, gs) = mk_world(false);
        let res = route_network(&nodes, &gs, &p, &mut memory, 5.0, &env);
        assert_eq!(res.sat_ground_rate[0], 0.0);
        assert!(
            res.isl_links.iter().any(|&(a, b, c)| (a == 0 || b == 0) && c == 0.0),
            "held ISL should be reported at 0: {:?}", res.isl_links
        );
        assert!(res.events.is_empty(), "no handover events expected: {:?}", res.events);

        // Station returns: traffic resumes WITHOUT a new acquisition.
        let (nodes, gs) = mk_world(true);
        // (two passes: the relay's own SGL must re-acquire after its outage)
        route_network(&nodes, &gs, &p, &mut memory, 5.0, &env);
        let res = route_network(&nodes, &gs, &p, &mut memory, 15.0, &env);
        assert!(res.sat_ground_rate[0] > 0.0, "chain should resume");
        assert!(
            !res.events.iter().any(|e| e.starts_with("LEO")),
            "the LEO must not re-acquire: {:?}", res.events
        );
    }

    #[test]
    fn acquisition_delay_blocks_traffic_then_releases() {
        let env = test_env();
        let p = RouteParams { prioritize_relay: false, hysteresis: 1.3, acquisition_time_s: 30.0, min_dwell_s: 0.0 };
        let mut memory = LinkMemory::new();
        let mk_sat = || {
            let mut s = node([R_EARTH + 800_000.0, 0.0, 0.0], false, 40.0);
            s.id = "LEO".into();
            s
        };
        let mk_gs = || vec![GroundNode { id: "GA".into(), r: [R_EARTH, 0.0, 0.0], k_value: 0.0, capacity: f64::INFINITY, min_elev_rad: 0.0 }];

        // First contact: link created, still acquiring → no traffic.
        let res = route_network(&[mk_sat()], &mk_gs(), &p, &mut memory, 0.0, &env);
        assert_eq!(res.sat_ground_rate[0], 0.0);
        assert_eq!(res.sgl_links[0].2, 0.0, "acquiring link must be reported at 0");
        assert!(res.events.iter().any(|e| e.contains("acquiring")));

        // 20 s later: still acquiring.
        let res = route_network(&[mk_sat()], &mk_gs(), &p, &mut memory, 20.0, &env);
        assert_eq!(res.sat_ground_rate[0], 0.0);

        // Another 20 s: acquisition complete, traffic flows.
        let res = route_network(&[mk_sat()], &mk_gs(), &p, &mut memory, 20.0, &env);
        assert!(res.sat_ground_rate[0] > 0.0, "link must carry traffic after acquisition");
    }

    #[test]
    fn handover_on_link_loss_requires_reacquisition() {
        let env = test_env();
        let p = RouteParams { prioritize_relay: false, hysteresis: 1.3, acquisition_time_s: 10.0, min_dwell_s: 120.0 };
        let mut memory = LinkMemory::new();
        let mk_gs = |ga_visible: bool| {
            let ga_r = if ga_visible { [R_EARTH, 0.0, 0.0] } else { [-R_EARTH, 0.0, 0.0] };
            vec![
                GroundNode { id: "GA".into(), r: ga_r, k_value: 0.0, capacity: f64::INFINITY, min_elev_rad: 0.0 },
                GroundNode { id: "GB".into(), r: [R_EARTH * 0.9, R_EARTH * 0.436, 0.0], k_value: 0.0, capacity: f64::INFINITY, min_elev_rad: 0.0 },
            ]
        };
        let mk_sat = || {
            let mut s = node([R_EARTH + 800_000.0, 0.0, 0.0], false, 40.0);
            s.id = "LEO".into();
            s
        };

        // Establish and complete acquisition on GA.
        route_network(&[mk_sat()], &mk_gs(true), &p, &mut memory, 0.0, &env);
        let res = route_network(&[mk_sat()], &mk_gs(true), &p, &mut memory, 15.0, &env);
        assert!(res.sat_ground_rate[0] > 0.0);
        assert_eq!(res.sgl_links[0].1, 0);

        // GA disappears behind the planet: forced handover to GB, re-acquiring.
        let res = route_network(&[mk_sat()], &mk_gs(false), &p, &mut memory, 15.0, &env);
        assert_eq!(res.sgl_links[0].1, 1, "must fail over to GB");
        assert_eq!(res.sgl_links[0].2, 0.0, "failover link must re-acquire");
        assert!(res.events.iter().any(|e| e.contains("handover")));
    }

    #[test]
    #[ignore] // temporary diagnostic, run explicitly with --ignored --nocapture
    fn diag_full_constellation_over_time() {
        use crate::adcs::{compute_adcs_command, nadir_target_quaternion, AdcsGains, SensorNoise};
        use crate::math::{lla_to_ecef, mat_vec_mult, eci_to_ecef_matrix, rotate_vector_q, Lcg};
        use crate::physics::{dipole_field_eci, step_atmosphere, step_attitude, step_orbit, sun_direction};
        use crate::simulation::create_satellites_from_config;
        use crate::models::{AtmosphereModel, OrbitType};

        let config = crate::app::default_config();
        let mut constellation = create_satellites_from_config(&config);
        let mut stations = config.stations.clone();
        let mut atmos = AtmosphereModel {
            states: config.atmos_states.clone(),
            k_values: config.atmos_k.clone(),
            transition_matrix: config.transition_matrix.clone(),
            lcg: Lcg::new(42),
        };
        let mut sensor_rng = Lcg::new(2024);
        let mut memory = LinkMemory::new();
        let p = RouteParams {
            prioritize_relay: false,
            hysteresis: config.handover_hysteresis,
            acquisition_time_s: config.acquisition_time_s,
            min_dwell_s: config.min_dwell_s,
        };
        let gains = AdcsGains::default();
        let noise = SensorNoise::default();

        let mut t = 0.0_f64;
        let mut total_events = 0usize;
        let mut min_tp = f64::INFINITY;
        let mut max_tp = 0.0_f64;
        for step in 0..600 {
            // physics step (mirrors app update loop)
            for gs in stations.iter_mut() {
                step_atmosphere(gs, &mut atmos);
            }
            let sun = sun_direction(t);
            let gst = t * 7.292115e-5;
            for seg in &mut constellation.segments {
                for sat in &mut seg.satellites {
                    let b_eci = dipole_field_eci(sat.r, gst, config.env.r_earth);
                    let q_t = nadir_target_quaternion(sat.r, sat.v);
                    let b_body = rotate_vector_q(sat.q, b_eci);
                    let (rw, mtq) = compute_adcs_command(sat, q_t, b_body, &gains, &noise, &mut sensor_rng);
                    step_orbit(sat, 1.0, &config.env, sun);
                    step_attitude(sat, 1.0, b_eci, rw, mtq, [0.0; 3]);
                }
            }
            t += 1.0;

            // routing pass
            let rot = eci_to_ecef_matrix(gst);
            let rot_t = [[rot[0][0],rot[1][0],rot[2][0]],[rot[0][1],rot[1][1],rot[2][1]],[rot[0][2],rot[1][2],rot[2][2]]];
            let mut nodes = Vec::new();
            for seg in &constellation.segments {
                for sat in &seg.satellites {
                    let (max_cap, sgl_ref, isl_ref, is_relay) = match sat.orbit_type {
                        OrbitType::LEO => (100.0, config.ref_dist_sgl_km, config.ref_dist_isl_km, false),
                        OrbitType::MEO => (400.0, config.meo_alt_km, config.meo_alt_km, true),
                        OrbitType::GEO => (800.0, config.geo_alt_km, config.geo_alt_km, true),
                    };
                    nodes.push(RouteNode { id: sat.id.clone(), is_relay, max_cap, sgl_ref_dist: sgl_ref, isl_ref_dist: isl_ref, r: sat.r, point_factor: 1.0 });
                }
            }
            let gs_nodes: Vec<GroundNode> = stations.iter().map(|gs| {
                let ecef = lla_to_ecef(gs.lat_rad, gs.lon_rad, gs.alt_m);
                GroundNode { id: gs.id.clone(), r: mat_vec_mult(rot_t, ecef), k_value: gs.k_value, capacity: gs.downlink_nominal_gbps, min_elev_rad: config.min_elevation_deg.to_radians() }
            }).collect();
            let res = route_network(&nodes, &gs_nodes, &p, &mut memory, 1.0, &env_of(&config));
            total_events += res.events.len();
            if step >= 60 {
                min_tp = min_tp.min(res.total_throughput);
                max_tp = max_tp.max(res.total_throughput);
            }
            if step % 60 == 0 || !res.events.is_empty() {
                println!("t={step:4}s tp={:7.1} events={:?}", res.total_throughput, res.events);
            }
        }
        println!("=== total events: {total_events}, tp range after 60s: {min_tp:.1}..{max_tp:.1}");
    }

    fn env_of(config: &crate::config::Config) -> SimEnvironment {
        config.env.clone()
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
        let res = route(nodes, gs, true, &env);
        // Flow must traverse the 50 Gbps relay: its residual consumption shows up
        // as the LEO's first hop in isl_links.
        let leo_hop = res.isl_links.iter().find(|(a, b, _)| *a == 0 || *b == 0);
        let (a, b, flow) = leo_hop.expect("LEO got no ISL");
        let partner = if *a == 0 { *b } else { *a };
        assert_eq!(partner, 2, "LEO routed via the narrow relay");
        assert!(*flow > 5.0, "flow {flow} did not exceed the narrow relay's capacity");
    }
}
