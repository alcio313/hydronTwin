// Rewind support: a bounded ring buffer of exact simulation snapshots.
//
// Negative time warp restores previously recorded states instead of integrating
// the physics backward: reverse integration is wrong for this simulator (the
// ADCS feedback loop becomes positive feedback, the Markov weather chain is not
// reversible, drag is numerically unstable in reverse). Restoring snapshots —
// including both RNG states — replays the exact recorded history, and moving
// forward again after a rewind reproduces it deterministically.

use std::collections::VecDeque;

use crate::math::Lcg;
use crate::models::{Constellation, GroundStation};

/// Maximum snapshots kept (~30 min of simulated time at the 1 s step).
/// ~128 B per satellite per snapshot: ≈4 MB for a 17-satellite constellation.
pub const MAX_SNAPSHOTS: usize = 1800;

#[derive(Clone)]
struct SatSnapshot {
    r: [f64; 3],
    v: [f64; 3],
    q: [f64; 4],
    omega: [f64; 3],
    h_rw: [f64; 3],
}

struct SimSnapshot {
    time: f64,
    sats: Vec<SatSnapshot>,
    /// (atmos_state, k_value) per ground station.
    gs_atmos: Vec<(usize, f64)>,
    atmos_lcg: u64,
    sensor_lcg: u64,
}

/// Ring buffer of simulation states, invalidated automatically whenever the
/// constellation structure (satellite ids) or the station count changes.
pub struct RewindBuffer {
    snapshots: VecDeque<SimSnapshot>,
    sat_ids: Vec<String>,
    gs_count: usize,
}

fn current_sat_ids(constellation: &Constellation) -> Vec<String> {
    constellation
        .segments
        .iter()
        .flat_map(|seg| seg.satellites.iter().map(|s| s.id.clone()))
        .collect()
}

impl RewindBuffer {
    pub fn new() -> Self {
        Self {
            snapshots: VecDeque::new(),
            sat_ids: Vec::new(),
            gs_count: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    pub fn clear(&mut self) {
        self.snapshots.clear();
        self.sat_ids.clear();
        self.gs_count = 0;
    }

    /// Capture the full simulation state after a forward step. If the
    /// constellation structure or the station list changed since the last
    /// record, the whole buffer is discarded first: old snapshots no longer
    /// describe the current world.
    pub fn record(
        &mut self,
        time: f64,
        constellation: &Constellation,
        stations: &[GroundStation],
        atmos_lcg: &Lcg,
        sensor_lcg: &Lcg,
    ) {
        let ids = current_sat_ids(constellation);
        if ids != self.sat_ids || stations.len() != self.gs_count {
            self.snapshots.clear();
            self.sat_ids = ids;
            self.gs_count = stations.len();
        }

        let sats = constellation
            .segments
            .iter()
            .flat_map(|seg| seg.satellites.iter())
            .map(|s| SatSnapshot { r: s.r, v: s.v, q: s.q, omega: s.omega, h_rw: s.h_rw })
            .collect();
        let gs_atmos = stations.iter().map(|g| (g.atmos_state, g.k_value)).collect();

        self.snapshots.push_back(SimSnapshot {
            time,
            sats,
            gs_atmos,
            atmos_lcg: atmos_lcg.state(),
            sensor_lcg: sensor_lcg.state(),
        });
        if self.snapshots.len() > MAX_SNAPSHOTS {
            self.snapshots.pop_front();
        }
    }

    /// Step one snapshot back: drop the newest state (the present) and restore
    /// the previous one. Returns the restored simulation time, or None when the
    /// buffer is exhausted (or no longer matches the current world).
    pub fn rewind(
        &mut self,
        constellation: &mut Constellation,
        stations: &mut [GroundStation],
        atmos_lcg: &mut Lcg,
        sensor_lcg: &mut Lcg,
    ) -> Option<f64> {
        if current_sat_ids(constellation) != self.sat_ids || stations.len() != self.gs_count {
            self.clear();
            return None;
        }
        if self.snapshots.len() < 2 {
            return None;
        }
        self.snapshots.pop_back();
        let snap = self.snapshots.back().expect("checked above");

        let mut it = snap.sats.iter();
        for seg in &mut constellation.segments {
            for sat in &mut seg.satellites {
                let s = it.next().expect("sat count matches sat_ids");
                sat.r = s.r;
                sat.v = s.v;
                sat.q = s.q;
                sat.omega = s.omega;
                sat.h_rw = s.h_rw;
            }
        }
        for (gs, &(state, k)) in stations.iter_mut().zip(snap.gs_atmos.iter()) {
            gs.atmos_state = state;
            gs.k_value = k;
        }
        atmos_lcg.set_state(snap.atmos_lcg);
        sensor_lcg.set_state(snap.sensor_lcg);

        Some(snap.time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adcs::{compute_adcs_command, nadir_body_rate, nadir_target_quaternion, AdcsGains, SensorNoise};
    use crate::math::rotate_vector_q;
    use crate::models::{AtmosphereModel, OrbitType, Satellite, Segment, SimEnvironment};
    use crate::physics::{step_atmosphere, step_attitude, step_orbit};

    fn test_env() -> SimEnvironment {
        SimEnvironment {
            mu: 3.986004418e14,
            r_earth: 6378137.0,
            j2: 1.08262668e-3,
            rho0_500km: 3.8e-12,
            h0_km: 500.0,
            scale_height_km: 70.0,
            p_srp: 4.56e-6,
        }
    }

    fn make_sat(id: &str, r0: f64, phase: f64) -> Satellite {
        let v_mag = (3.986004418e14_f64 / r0).sqrt();
        let r = [r0 * phase.cos(), r0 * phase.sin(), 0.0];
        let v = [-v_mag * phase.sin(), v_mag * phase.cos(), 0.0];
        Satellite {
            id: id.to_string(),
            orbit_type: OrbitType::LEO,
            r,
            v,
            q: nadir_target_quaternion(r, v),
            omega: nadir_body_rate(r, v),
            mass: 20.0,
            area: 0.1,
            cd: 2.2,
            cr: 1.2,
            inertia: [0.4, 0.4, 0.5],
            h_rw: [0.0, 0.0, 0.0],
            is_custom: false,
            custom_color: None,
        }
    }

    fn make_world() -> (Constellation, Vec<GroundStation>, AtmosphereModel) {
        let constellation = Constellation {
            name: "TEST".to_string(),
            segments: vec![Segment {
                orbit_type: OrbitType::LEO,
                satellites: vec![make_sat("A", 6928137.0, 0.0), make_sat("B", 6928137.0, 1.0)],
            }],
        };
        let stations = vec![GroundStation {
            id: "GS".to_string(),
            name: "Test".to_string(),
            lat_rad: 0.0,
            lon_rad: 0.0,
            alt_m: 0.0,
            downlink_nominal_gbps: f64::INFINITY,
            atmos_state: 0,
            k_value: 0.05 / 1000.0,
        }];
        let atmos = AtmosphereModel {
            states: vec!["clear".into(), "thin".into(), "thick".into(), "heavy".into()],
            k_values: vec![0.05, 0.2, 1.5, 5.0],
            transition_matrix: vec![
                vec![0.85, 0.10, 0.04, 0.01],
                vec![0.15, 0.70, 0.10, 0.05],
                vec![0.05, 0.15, 0.65, 0.15],
                vec![0.02, 0.08, 0.20, 0.70],
            ],
            lcg: Lcg::new(42),
        };
        (constellation, stations, atmos)
    }

    /// One full simulation step, mirroring the app's update loop.
    fn sim_step(
        constellation: &mut Constellation,
        stations: &mut [GroundStation],
        atmos: &mut AtmosphereModel,
        sensor_rng: &mut Lcg,
        env: &SimEnvironment,
    ) {
        for gs in stations.iter_mut() {
            step_atmosphere(gs, atmos);
        }
        let gains = AdcsGains::default();
        let noise = SensorNoise::default();
        let b_eci = [1e-5, 2e-5, -3e-5];
        for seg in &mut constellation.segments {
            for sat in &mut seg.satellites {
                let q_t = nadir_target_quaternion(sat.r, sat.v);
                let b_body = rotate_vector_q(sat.q, b_eci);
                let (rw, mtq) = compute_adcs_command(sat, q_t, b_body, &gains, &noise, sensor_rng);
                step_orbit(sat, 1.0, env, [1.0, 0.0, 0.0]);
                step_attitude(sat, 1.0, b_eci, rw, mtq, [0.0; 3]);
            }
        }
    }

    fn world_fingerprint(constellation: &Constellation, stations: &[GroundStation]) -> Vec<f64> {
        let mut f = Vec::new();
        for seg in &constellation.segments {
            for s in &seg.satellites {
                f.extend_from_slice(&s.r);
                f.extend_from_slice(&s.v);
                f.extend_from_slice(&s.q);
                f.extend_from_slice(&s.omega);
                f.extend_from_slice(&s.h_rw);
            }
        }
        for g in stations {
            f.push(g.atmos_state as f64);
            f.push(g.k_value);
        }
        f
    }

    #[test]
    fn rewind_restores_exact_state() {
        let env = test_env();
        let (mut con, mut gs, mut atmos) = make_world();
        let mut sensor_rng = Lcg::new(2024);
        let mut buf = RewindBuffer::new();
        let mut time = 0.0;
        buf.record(time, &con, &gs, &atmos.lcg, &sensor_rng);

        let mut fingerprint_at_60 = Vec::new();
        for k in 0..100 {
            sim_step(&mut con, &mut gs, &mut atmos, &mut sensor_rng, &env);
            time += 1.0;
            buf.record(time, &con, &gs, &atmos.lcg, &sensor_rng);
            if k == 59 {
                fingerprint_at_60 = world_fingerprint(&con, &gs);
            }
        }

        let mut restored_time = time;
        for _ in 0..40 {
            restored_time = buf
                .rewind(&mut con, &mut gs, &mut atmos.lcg, &mut sensor_rng)
                .expect("buffer holds 101 snapshots");
        }
        assert_eq!(restored_time, 60.0);
        assert_eq!(world_fingerprint(&con, &gs), fingerprint_at_60);
    }

    #[test]
    fn forward_replay_is_deterministic() {
        let env = test_env();
        let (mut con, mut gs, mut atmos) = make_world();
        let mut sensor_rng = Lcg::new(2024);
        let mut buf = RewindBuffer::new();
        buf.record(0.0, &con, &gs, &atmos.lcg, &sensor_rng);

        for k in 0..100 {
            sim_step(&mut con, &mut gs, &mut atmos, &mut sensor_rng, &env);
            buf.record((k + 1) as f64, &con, &gs, &atmos.lcg, &sensor_rng);
        }
        let final_original = world_fingerprint(&con, &gs);

        for _ in 0..40 {
            buf.rewind(&mut con, &mut gs, &mut atmos.lcg, &mut sensor_rng).unwrap();
        }
        for _ in 0..40 {
            sim_step(&mut con, &mut gs, &mut atmos, &mut sensor_rng, &env);
        }
        assert_eq!(world_fingerprint(&con, &gs), final_original);
    }

    #[test]
    fn buffer_eviction_and_exhaustion() {
        let (mut con, mut gs, mut atmos) = make_world();
        let mut sensor_rng = Lcg::new(1);
        let mut buf = RewindBuffer::new();

        for k in 0..(MAX_SNAPSHOTS + 10) {
            buf.record(k as f64, &con, &gs, &atmos.lcg, &sensor_rng);
        }
        assert_eq!(buf.len(), MAX_SNAPSHOTS);

        let mut rewinds = 0;
        while buf
            .rewind(&mut con, &mut gs, &mut atmos.lcg, &mut sensor_rng)
            .is_some()
        {
            rewinds += 1;
        }
        // One snapshot always remains (the oldest recorded state).
        assert_eq!(rewinds, MAX_SNAPSHOTS - 1);
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn structure_change_clears_buffer() {
        let (mut con, gs, atmos) = make_world();
        let sensor_rng = Lcg::new(1);
        let mut buf = RewindBuffer::new();

        buf.record(0.0, &con, &gs, &atmos.lcg, &sensor_rng);
        buf.record(1.0, &con, &gs, &atmos.lcg, &sensor_rng);
        assert_eq!(buf.len(), 2);

        con.segments[0].satellites.push(make_sat("C", 6928137.0, 2.0));
        buf.record(2.0, &con, &gs, &atmos.lcg, &sensor_rng);
        assert_eq!(buf.len(), 1, "buffer must restart after a structure change");
    }
}
