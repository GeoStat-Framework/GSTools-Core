//! Rust-owned Direct Sampling simulation engine.
//!
//! Python supplies path and random-number generation. This module owns the per-node computation and a
//! deterministic level scheduler: neighbour selection, search-window
//! construction, candidate scanning, retrieval, variation adjustment, and SG
//! writes. Nodes in one dependency level are independent and may be evaluated
//! by one simulation-local Rayon pool; their results are committed in path
//! order after the level completes.

use crate::mps::scan_node;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ArrayView3, Axis};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use rayon::{ThreadPool, ThreadPoolBuilder};

/// Result and scheduler diagnostics from one Rust-owned simulation.
pub struct SimulationOutput {
    /// Simulated variables as C-order rows, shape `(n_variables, sg_size)`.
    pub fields: Array2<f64>,
    /// Number of strict windows that required the established partial fallback.
    pub strict_fallback_count: usize,
    /// Number of deterministic dependency levels executed.
    pub level_count: usize,
    /// Largest number of mutually independent nodes in one level.
    pub max_ready_width: usize,
    /// Effective Rayon worker count after the measured small-path cutoff.
    pub used_threads: usize,
    /// Transformed event lags removed because multiple SG lags collapsed.
    pub collapsed_lag_count: usize,
}

#[derive(Clone)]
struct Neighbor {
    flat: usize,
    lag: Vec<i64>,
}

struct CachedNode {
    by_variable: Vec<Vec<Neighbor>>,
    level: usize,
}

struct Event {
    values: Vec<f64>,
    conditioned: Vec<bool>,
    lags: Vec<Vec<i64>>,
    lag_norms: Vec<f64>,
}

impl Event {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn truncate(&mut self, keep: usize) {
        self.values.truncate(keep);
        self.conditioned.truncate(keep);
        self.lags.truncate(keep);
        self.lag_norms.truncate(keep);
    }
}

struct NodeResult {
    flat: usize,
    values: Vec<(usize, f64)>,
    strict_fallback_count: usize,
    collapsed_lag_count: usize,
}

type WindowBounds = Option<(Vec<i64>, Vec<i64>, usize)>;

fn c_strides(shape: ArrayView1<'_, i64>) -> Vec<i64> {
    let mut strides = vec![1_i64; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

fn flat_index(coords: &[i64], strides: &[i64]) -> usize {
    flat_index_iter(coords.iter().copied(), strides)
}

fn flat_index_iter(coords: impl Iterator<Item = i64>, strides: &[i64]) -> usize {
    coords
        .zip(strides.iter())
        .map(|(coordinate, &stride)| coordinate * stride)
        .sum::<i64>() as usize
}

#[allow(clippy::too_many_arguments)]
fn build_neighbor_cache(
    path: ArrayView2<'_, i64>,
    sim_shape: ArrayView1<'_, i64>,
    offsets: ArrayView2<'_, i64>,
    conditioned: ArrayView2<'_, u8>,
    n_neighbors: ArrayView1<'_, i64>,
    max_radius: ArrayView1<'_, f64>,
    active_mask: Option<ArrayView1<'_, u8>>,
) -> Vec<CachedNode> {
    let variable_count = conditioned.nrows();
    let sg_size = sim_shape.iter().product::<i64>() as usize;
    let strides = c_strides(sim_shape);
    let path_flat: Vec<usize> = path
        .rows()
        .into_iter()
        .map(|coords| flat_index_iter(coords.iter().copied(), &strides))
        .collect();

    let mut path_maps = vec![vec![-1_i64; sg_size]; variable_count];
    for variable in 0..variable_count {
        for (index, &flat) in path_flat.iter().enumerate() {
            path_maps[variable][flat] = index as i64;
        }
        for flat in 0..sg_size {
            if conditioned[(variable, flat)] != 0 {
                path_maps[variable][flat] = -1;
            }
        }
    }

    let mut levels = vec![0_usize; path.nrows()];
    let mut cache = Vec::with_capacity(path.nrows());
    for node in 0..path.nrows() {
        let coords = path.row(node);
        let mut by_variable = Vec::with_capacity(variable_count);
        let mut level = 0_usize;
        for variable in 0..variable_count {
            let radius_sq = if max_radius[variable].is_finite() {
                Some(max_radius[variable] * max_radius[variable])
            } else {
                None
            };
            let mut selected =
                Vec::with_capacity((n_neighbors[variable] as usize).min(offsets.nrows()));
            for offset in offsets.rows() {
                let distance_sq: f64 = offset
                    .iter()
                    .map(|&component| (component as f64) * (component as f64))
                    .sum();
                if radius_sq.is_some_and(|limit| distance_sq > limit) {
                    break;
                }
                let mut candidate = Vec::with_capacity(sim_shape.len());
                let mut in_bounds = true;
                for axis in 0..sim_shape.len() {
                    let coordinate = coords[axis] + offset[axis];
                    if coordinate < 0 || coordinate >= sim_shape[axis] {
                        in_bounds = false;
                        break;
                    }
                    candidate.push(coordinate);
                }
                if !in_bounds {
                    continue;
                }
                let flat = flat_index(&candidate, &strides);
                if active_mask.is_some_and(|mask| mask[flat] == 0) {
                    continue;
                }
                let dependency = path_maps[variable][flat];
                if dependency == -1 || dependency < node as i64 {
                    if dependency >= 0 {
                        level = level.max(levels[dependency as usize] + 1);
                    }
                    selected.push(Neighbor {
                        flat,
                        lag: offset.to_vec(),
                    });
                    if selected.len() == n_neighbors[variable] as usize {
                        break;
                    }
                }
            }
            by_variable.push(selected);
        }
        levels[node] = level;
        cache.push(CachedNode { by_variable, level });
    }
    cache
}

/// Neighbour cache for a post-processing (cleanup) pass.
///
/// Unlike [`build_neighbor_cache`], every in-bounds cell within radius is an
/// eligible neighbour: a cleanup pass runs over an already-complete field, so
/// the neighbourhood is fully informed (Meerschman et al. 2013, Sect. 4) and
/// is independent of visit order. Each node keeps the nearest
/// `n_neighbors[variable]` offsets. A separate position-oriented dependency DAG
/// preserves Gauss-Seidel read/write order during parallel execution.
fn build_cleanup_neighbor_cache(
    path: ArrayView2<'_, i64>,
    sim_shape: ArrayView1<'_, i64>,
    offsets: ArrayView2<'_, i64>,
    n_neighbors: ArrayView1<'_, i64>,
    max_radius: ArrayView1<'_, f64>,
    variable_count: usize,
    active_mask: Option<ArrayView1<'_, u8>>,
) -> Vec<CachedNode> {
    let strides = c_strides(sim_shape);
    let mut cache = Vec::with_capacity(path.nrows());
    for node in 0..path.nrows() {
        let coords = path.row(node);
        let mut by_variable = Vec::with_capacity(variable_count);
        for variable in 0..variable_count {
            let radius_sq = if max_radius[variable].is_finite() {
                Some(max_radius[variable] * max_radius[variable])
            } else {
                None
            };
            let mut selected =
                Vec::with_capacity((n_neighbors[variable] as usize).min(offsets.nrows()));
            for offset in offsets.rows() {
                let distance_sq: f64 = offset
                    .iter()
                    .map(|&component| (component as f64) * (component as f64))
                    .sum();
                if radius_sq.is_some_and(|limit| distance_sq > limit) {
                    break;
                }
                let mut candidate = Vec::with_capacity(sim_shape.len());
                let mut in_bounds = true;
                for axis in 0..sim_shape.len() {
                    let coordinate = coords[axis] + offset[axis];
                    if coordinate < 0 || coordinate >= sim_shape[axis] {
                        in_bounds = false;
                        break;
                    }
                    candidate.push(coordinate);
                }
                if !in_bounds {
                    continue;
                }
                let flat = flat_index(&candidate, &strides);
                if active_mask.is_some_and(|mask| mask[flat] == 0) {
                    continue;
                }
                selected.push(Neighbor {
                    flat,
                    lag: offset.to_vec(),
                });
                if selected.len() == n_neighbors[variable] as usize {
                    break;
                }
            }
            by_variable.push(selected);
        }
        cache.push(CachedNode {
            by_variable,
            level: 0,
        });
    }
    cache
}

/// Deterministic dependency levels for a parallel cleanup pass.
///
/// Edges are oriented by *visit position*: for every eligible-node neighbour
/// pair the lower position precedes the higher, so the visit order is itself a
/// topological order (Meerschman et al. 2013, Sect. 4, realised as a DAG). An
/// earlier-position neighbour is a read-after-write true dependency (the node
/// reads its refreshed value); a later-position neighbour is a write-after-read
/// anti-dependency (the node reads its stale value, so it must run before that
/// neighbour overwrites). Because `n_neighbors` truncation makes the read
/// relation asymmetric, a pair may be found from either endpoint — taking the
/// union (any sighting creates the edge) yields the same levels regardless.
///
/// `level[node]` is one past the deepest lower-position neighbour. Nodes in one
/// level are mutually non-adjacent (an edge forces distinct levels), so a
/// read-the-snapshot/commit-after-level scheduler reproduces serial
/// Gauss-Seidel at any thread count. Returns the per-level node groups, the
/// level count, and the widest level.
fn build_cleanup_levels(
    cache: &[CachedNode],
    path: ArrayView2<'_, i64>,
    sim_shape: ArrayView1<'_, i64>,
) -> (Vec<Vec<usize>>, usize, usize) {
    let n = path.nrows();
    let strides = c_strides(sim_shape);
    let sg_size = sim_shape.iter().product::<i64>() as usize;
    // Visit position of each eligible node by flat cell (-1 = frozen / absent).
    let mut pos_map = vec![-1_i64; sg_size];
    for (i, coords) in path.rows().into_iter().enumerate() {
        pos_map[flat_index_iter(coords.iter().copied(), &strides)] = i as i64;
    }
    // Lower-position neighbours per node (duplicates are harmless for the max).
    let mut lower_adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, cached) in cache.iter().enumerate() {
        for by_var in &cached.by_variable {
            for neighbor in by_var {
                let other = pos_map[neighbor.flat];
                if other < 0 || other as usize == i {
                    continue;
                }
                let other = other as usize;
                let (lo, hi) = if other < i { (other, i) } else { (i, other) };
                lower_adj[hi].push(lo);
            }
        }
    }
    let mut node_level = vec![0_usize; n];
    for hi in 0..n {
        let level = lower_adj[hi]
            .iter()
            .map(|&lo| node_level[lo] + 1)
            .max()
            .unwrap_or(0);
        node_level[hi] = level;
    }
    let level_count = node_level.iter().copied().max().map_or(0, |l| l + 1);
    let mut levels = vec![Vec::new(); level_count];
    for (node, &level) in node_level.iter().enumerate() {
        levels[level].push(node);
    }
    let max_width = levels.iter().map(Vec::len).max().unwrap_or(0);
    (levels, level_count, max_width)
}

fn compute_node_weights(
    lag_norms: &[f64],
    conditioned: &[bool],
    distance_power: f64,
    cond_weight: f64,
) -> Vec<f64> {
    let mut weights = vec![1.0_f64; lag_norms.len()];
    if distance_power != 0.0 {
        for (weight, &norm) in weights.iter_mut().zip(lag_norms.iter()) {
            if norm != 0.0 {
                *weight = norm.powf(-distance_power);
            }
        }
    }
    for (weight, &is_conditioned) in weights.iter_mut().zip(conditioned.iter()) {
        if is_conditioned {
            *weight *= cond_weight;
        }
    }
    let total: f64 = weights.iter().sum();
    if total.is_finite() && total > 0.0 {
        // Direct normalization preserves the ordinary-case arithmetic order.
        for weight in &mut weights {
            *weight /= total;
        }
        return weights;
    }

    // Degenerate direct sum: raw weights overflowed to +inf (large
    // finite cond_weight) or every positive weight underflowed to 0 (large
    // distance_power). Rebuild in the log domain so neither extreme collapses
    // to a uniform or all-zero event. A node is a structural zero only by
    // intent: a conditioning node with cond_weight == 0.
    let count = weights.len();
    let structural_zero = |index: usize| cond_weight == 0.0 && conditioned[index];
    if (0..count).all(structural_zero) {
        weights.fill(0.0);
        return weights;
    }
    // log weight_i = -power * (log r_i - min log r) + log(cond_weight?), with
    // r_i = 1 for a zero-lag baseline; then subtract the max before exp.
    let log_lag: Vec<f64> = lag_norms
        .iter()
        .map(|&norm| {
            if distance_power != 0.0 && norm != 0.0 {
                norm.ln()
            } else {
                0.0
            }
        })
        .collect();
    let min_log_lag = (0..count)
        .filter(|&i| !structural_zero(i))
        .map(|i| log_lag[i])
        .fold(f64::INFINITY, f64::min);
    let log_cond = if cond_weight > 0.0 {
        cond_weight.ln()
    } else {
        0.0
    };
    let mut log_w: Vec<f64> = (0..count)
        .map(|i| {
            if structural_zero(i) {
                f64::NEG_INFINITY
            } else {
                let mut value = -distance_power * (log_lag[i] - min_log_lag);
                if conditioned[i] {
                    value += log_cond;
                }
                value
            }
        })
        .collect();
    let max_log_w = log_w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for value in &mut log_w {
        *value -= max_log_w;
    }
    for (weight, &value) in weights.iter_mut().zip(log_w.iter()) {
        *weight = if value.is_finite() { value.exp() } else { 0.0 };
    }
    let total: f64 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= total;
    }
    weights
}

fn bounds_for_prefix(
    lags: &[Vec<i64>],
    keep: usize,
    ti_shape: ArrayView1<'_, i64>,
) -> Option<(Vec<i64>, Vec<i64>)> {
    let dim = ti_shape.len();
    let nonzero: Vec<&Vec<i64>> = lags[..keep]
        .iter()
        .filter(|lag| lag.iter().any(|&component| component != 0))
        .collect();
    if nonzero.is_empty() {
        return Some((
            vec![0_i64; dim],
            ti_shape.iter().map(|&size| size - 1).collect(),
        ));
    }
    let mut lo = vec![0_i64; dim];
    let mut hi: Vec<i64> = ti_shape.iter().map(|&size| size - 1).collect();
    for axis in 0..dim {
        let min_lag = nonzero.iter().map(|lag| lag[axis]).min().unwrap_or(0);
        let max_lag = nonzero.iter().map(|lag| lag[axis]).max().unwrap_or(0);
        lo[axis] = 0_i64.max(-min_lag);
        hi[axis] = (ti_shape[axis] - 1).min(ti_shape[axis] - 1 - max_lag);
    }
    lo.iter()
        .zip(hi.iter())
        .all(|(&lower, &upper)| lower <= upper)
        .then_some((lo, hi))
}

fn window_bounds(
    lags: &[Vec<i64>],
    ti_shape: ArrayView1<'_, i64>,
    partial_boundary: bool,
) -> (WindowBounds, bool) {
    if let Some((lo, hi)) = bounds_for_prefix(lags, lags.len(), ti_shape) {
        return (Some((lo, hi, lags.len())), false);
    }
    let strict_fallback = !partial_boundary;
    for keep in (1..lags.len()).rev() {
        if let Some((lo, hi)) = bounds_for_prefix(lags, keep, ti_shape) {
            return (Some((lo, hi, keep)), strict_fallback);
        }
    }
    (None, strict_fallback)
}

fn transform_and_reduce_event(
    event: &mut Event,
    matrix: ArrayView2<'_, f64>,
    ti_shape: ArrayView1<'_, i64>,
) -> usize {
    let mut transformed = Vec::with_capacity(event.len());
    for lag in &event.lags {
        let mut mapped = vec![0_i64; lag.len()];
        for output_axis in 0..lag.len() {
            let value: f64 = (0..lag.len())
                .map(|input_axis| lag[input_axis] as f64 * matrix[(output_axis, input_axis)])
                .sum();
            mapped[output_axis] = value.round_ties_even() as i64;
        }
        transformed.push(mapped);
    }

    let mut keep = Vec::with_capacity(transformed.len());
    for index in 0..transformed.len() {
        if !keep
            .iter()
            .any(|&prior| transformed[prior] == transformed[index])
        {
            keep.push(index);
        }
    }
    let collapsed = transformed.len() - keep.len();
    if collapsed > 0 {
        event.values = keep.iter().map(|&index| event.values[index]).collect();
        event.conditioned = keep.iter().map(|&index| event.conditioned[index]).collect();
        event.lag_norms = keep.iter().map(|&index| event.lag_norms[index]).collect();
        transformed = keep
            .iter()
            .map(|&index| transformed[index].clone())
            .collect();
    }
    event.lags = transformed;

    loop {
        let nonzero: Vec<usize> = event
            .lags
            .iter()
            .enumerate()
            .filter_map(|(index, lag)| lag.iter().any(|&component| component != 0).then_some(index))
            .collect();
        if nonzero.is_empty() {
            break;
        }
        let mut over = vec![0_i64; ti_shape.len()];
        for axis in 0..ti_shape.len() {
            let min_lag = nonzero
                .iter()
                .map(|&index| event.lags[index][axis])
                .min()
                .unwrap_or(0)
                .min(0);
            let max_lag = nonzero
                .iter()
                .map(|&index| event.lags[index][axis])
                .max()
                .unwrap_or(0)
                .max(0);
            over[axis] = max_lag - min_lag - (ti_shape[axis] - 1);
        }
        if over.iter().all(|&amount| amount <= 0) {
            break;
        }
        let axis = over
            .iter()
            .enumerate()
            .max_by_key(|&(index, &amount)| (amount, std::cmp::Reverse(index)))
            .map(|(index, _)| index)
            .unwrap_or(0);
        let drop = nonzero
            .iter()
            .copied()
            .max_by_key(|&index| (event.lags[index][axis].abs(), std::cmp::Reverse(index)))
            .unwrap_or(0);
        event.values.remove(drop);
        event.conditioned.remove(drop);
        event.lags.remove(drop);
        event.lag_norms.remove(drop);
    }
    collapsed
}

fn fallback_flat(
    node: usize,
    u_fallback: ArrayView2<'_, f64>,
    ti_shape: ArrayView1<'_, i64>,
    ti_strides: &[i64],
    finite_all: &[usize],
    ti_has_nan: bool,
) -> usize {
    if ti_has_nan {
        let index = (u_fallback[(node, 0)] * finite_all.len() as f64) as usize;
        finite_all[index]
    } else {
        let coords: Vec<i64> = (0..ti_shape.len())
            .map(|axis| (u_fallback[(node, axis)] * ti_shape[axis] as f64) as i64)
            .collect();
        flat_index(&coords, ti_strides)
    }
}

#[allow(clippy::too_many_arguments)]
fn fallback_result(
    node: usize,
    node_flat: usize,
    targets: &[usize],
    ti: ArrayView2<'_, f64>,
    u_fallback: ArrayView2<'_, f64>,
    ti_shape: ArrayView1<'_, i64>,
    ti_strides: &[i64],
    finite_all: &[usize],
    ti_has_nan: bool,
    strict_fallback_count: usize,
) -> NodeResult {
    let source = fallback_flat(
        node, u_fallback, ti_shape, ti_strides, finite_all, ti_has_nan,
    );
    NodeResult {
        flat: node_flat,
        values: targets
            .iter()
            .map(|&variable| (variable, ti[(variable, source)]))
            .collect(),
        strict_fallback_count,
        collapsed_lag_count: 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn simulate_node(
    node: usize,
    cached: &CachedNode,
    fields: ArrayView2<'_, f64>,
    ti: ArrayView2<'_, f64>,
    ti_shape: ArrayView1<'_, i64>,
    sim_shape: ArrayView1<'_, i64>,
    path: ArrayView2<'_, i64>,
    lag_matrices: ArrayView3<'_, f64>,
    u_start: ArrayView1<'_, f64>,
    u_fallback: ArrayView2<'_, f64>,
    conditioned: ArrayView2<'_, u8>,
    variable_weights: ArrayView1<'_, f64>,
    metric_kinds: ArrayView1<'_, i64>,
    has_nan: ArrayView1<'_, u8>,
    d_max: ArrayView1<'_, f64>,
    p_norm: ArrayView1<'_, f64>,
    penalty_flat: ArrayView1<'_, f64>,
    penalty_offsets: ArrayView1<'_, i64>,
    penalty_ncat: ArrayView1<'_, i64>,
    threshold: f64,
    scan_fraction: f64,
    random_scan: bool,
    distance_power: ArrayView1<'_, f64>,
    cond_weight: ArrayView1<'_, f64>,
    partial_boundary: bool,
    finite_all: &[usize],
    ti_has_nan: bool,
    cleanup: bool,
) -> NodeResult {
    let ti_strides = c_strides(ti_shape);
    let sim_strides = c_strides(sim_shape);
    let node_coords = path.row(node);
    let node_flat = flat_index_iter(node_coords.iter().copied(), &sim_strides);
    // Cleanup reopens an already-complete field: a non-conditioned variable is
    // always a target (its stale value is suppressed, never a self-constraint),
    // and only a hard-conditioned variable stays as a collocated h=0 row. The
    // main pass instead keys both off the uninformed (NaN) state.
    let is_target = |variable: usize| {
        if cleanup {
            conditioned[(variable, node_flat)] == 0
        } else {
            fields[(variable, node_flat)].is_nan()
        }
    };
    let targets: Vec<usize> = (0..fields.nrows()).filter(|&v| is_target(v)).collect();

    let mut events = Vec::with_capacity(fields.nrows());
    let mut collapsed_lag_count = 0_usize;
    for variable in 0..fields.nrows() {
        // Zero-weight variables are excluded before search geometry.
        // An empty event places no constraint on the TI-window
        // intersection and is never scored, so an ignored variable cannot
        // discard a valid positive-weight match. Its centre is still pasted
        // from the chosen anchor (targets are independent of weight).
        if variable_weights[variable] == 0.0 {
            events.push(Event {
                values: Vec::new(),
                conditioned: Vec::new(),
                lags: Vec::new(),
                lag_norms: Vec::new(),
            });
            continue;
        }
        let collocated = !is_target(variable);
        let capacity = cached.by_variable[variable].len() + usize::from(collocated);
        let mut event = Event {
            values: Vec::with_capacity(capacity),
            conditioned: Vec::with_capacity(capacity),
            lags: Vec::with_capacity(capacity),
            lag_norms: Vec::with_capacity(capacity),
        };
        if collocated {
            event.values.push(fields[(variable, node_flat)]);
            event
                .conditioned
                .push(conditioned[(variable, node_flat)] != 0);
            event.lags.push(vec![0_i64; sim_shape.len()]);
            event.lag_norms.push(0.0);
        }
        for neighbor in &cached.by_variable[variable] {
            event.values.push(fields[(variable, neighbor.flat)]);
            event
                .conditioned
                .push(conditioned[(variable, neighbor.flat)] != 0);
            event.lags.push(neighbor.lag.clone());
            event.lag_norms.push(
                neighbor
                    .lag
                    .iter()
                    .map(|&component| (component as f64) * (component as f64))
                    .sum::<f64>()
                    .sqrt(),
            );
        }
        if !lag_matrices.is_empty() {
            collapsed_lag_count += transform_and_reduce_event(
                &mut event,
                lag_matrices.slice(ndarray::s![node, .., ..]),
                ti_shape,
            );
        }
        events.push(event);
    }

    if events.iter().all(|event| event.len() == 0) {
        let mut result = fallback_result(
            node,
            node_flat,
            &targets,
            ti,
            u_fallback,
            ti_shape,
            &ti_strides,
            finite_all,
            ti_has_nan,
            0,
        );
        result.collapsed_lag_count = collapsed_lag_count;
        return result;
    }

    let mut win_lo = vec![0_i64; ti_shape.len()];
    let mut win_hi: Vec<i64> = ti_shape.iter().map(|&size| size - 1).collect();
    let mut strict_fallback_count = 0_usize;
    for event in &mut events {
        let (bounds, strict_fallback) = window_bounds(&event.lags, ti_shape, partial_boundary);
        if strict_fallback {
            strict_fallback_count += 1;
        }
        let Some((lo, hi, keep)) = bounds else {
            let mut result = fallback_result(
                node,
                node_flat,
                &targets,
                ti,
                u_fallback,
                ti_shape,
                &ti_strides,
                finite_all,
                ti_has_nan,
                strict_fallback_count,
            );
            result.collapsed_lag_count = collapsed_lag_count;
            return result;
        };
        event.truncate(keep);
        for axis in 0..ti_shape.len() {
            win_lo[axis] = win_lo[axis].max(lo[axis]);
            win_hi[axis] = win_hi[axis].min(hi[axis]);
        }
    }
    if win_lo.iter().zip(win_hi.iter()).any(|(&lo, &hi)| lo > hi) {
        let mut result = fallback_result(
            node,
            node_flat,
            &targets,
            ti,
            u_fallback,
            ti_shape,
            &ti_strides,
            finite_all,
            ti_has_nan,
            strict_fallback_count,
        );
        result.collapsed_lag_count = collapsed_lag_count;
        return result;
    }

    let active: Vec<usize> = (0..events.len())
        .filter(|&variable| events[variable].len() > 0)
        .collect();
    let mut event_offsets = Vec::with_capacity(active.len() + 1);
    let mut de_sim = Vec::new();
    let mut lag_flat = Vec::new();
    let mut node_weights = Vec::new();
    event_offsets.push(0_i64);
    for &variable in &active {
        let event = &events[variable];
        de_sim.extend_from_slice(&event.values);
        lag_flat.extend(event.lags.iter().map(|lag| {
            lag.iter()
                .zip(ti_strides.iter())
                .map(|(&component, &stride)| component * stride)
                .sum::<i64>()
        }));
        node_weights.extend(compute_node_weights(
            &event.lag_norms,
            &event.conditioned,
            distance_power[variable],
            cond_weight[variable],
        ));
        event_offsets.push(de_sim.len() as i64);
    }

    let win_shape: Vec<i64> = win_lo
        .iter()
        .zip(win_hi.iter())
        .map(|(&lo, &hi)| hi - lo + 1)
        .collect();
    let win_size = win_shape.iter().product::<i64>() as usize;
    let ti_size = ti_shape.iter().product::<i64>() as usize;
    let max_scan = 1_usize.max(win_size.min((scan_fraction * ti_size as f64) as usize));
    let start = (u_start[node] * win_size as f64) as usize;
    // Node-local random-scan seed (reference derives the same from u_start_i).
    let scan_seed = (u_start[node] * 9007199254740992.0) as u64; // 2^53

    let active_rows = Array1::from_iter(active.iter().map(|&value| value as i64));
    let active_variable_weights =
        Array1::from_iter(active.iter().map(|&value| variable_weights[value]));
    let active_metrics = Array1::from_iter(active.iter().map(|&value| metric_kinds[value]));
    let active_has_nan = Array1::from_iter(active.iter().map(|&value| has_nan[value]));
    let active_d_max = Array1::from_iter(active.iter().map(|&value| d_max[value]));
    let active_p_norm = Array1::from_iter(active.iter().map(|&value| p_norm[value]));
    let active_penalty_offsets =
        Array1::from_iter(active.iter().map(|&value| penalty_offsets[value]));
    let active_penalty_ncat = Array1::from_iter(active.iter().map(|&value| penalty_ncat[value]));
    let target_rows = Array1::from_iter(targets.iter().map(|&value| value as i64));

    let anchor = scan_node(
        Array1::from_vec(win_lo).view(),
        Array1::from_vec(win_shape).view(),
        start,
        max_scan,
        threshold,
        ti,
        Array1::from_vec(ti_strides.clone()).view(),
        active_rows.view(),
        Array1::from_vec(event_offsets).view(),
        Array1::from_vec(de_sim).view(),
        Array1::from_vec(lag_flat).view(),
        Array1::from_vec(node_weights).view(),
        active_variable_weights.view(),
        active_metrics.view(),
        active_has_nan.view(),
        active_d_max.view(),
        active_p_norm.view(),
        random_scan,
        scan_seed,
        penalty_flat,
        active_penalty_offsets.view(),
        active_penalty_ncat.view(),
        target_rows.view(),
        ti_has_nan,
    );
    let Some(anchor) = anchor else {
        let mut result = fallback_result(
            node,
            node_flat,
            &targets,
            ti,
            u_fallback,
            ti_shape,
            &ti_strides,
            finite_all,
            ti_has_nan,
            strict_fallback_count,
        );
        result.collapsed_lag_count = collapsed_lag_count;
        return result;
    };
    let anchor_flat = flat_index(anchor.as_slice().expect("contiguous anchor"), &ti_strides);

    let values = targets
        .iter()
        .map(|&variable| {
            let center = ti[(variable, anchor_flat)];
            if metric_kinds[variable] != 2 || events[variable].len() == 0 {
                return (variable, center);
            }
            // Common-support mean shift: average SG - TI over only
            // the lags where BOTH the SG event value and the anchored TI value
            // are finite — the exact support the variation score used. The
            // means computed independently would shift the paste using values
            // outside the event support used for scoring.
            let event = &events[variable];
            let mut shift_sum = 0.0_f64;
            let mut common = 0_usize;
            for (index, lag) in event.lags.iter().enumerate() {
                let sg_value = event.values[index];
                if !sg_value.is_finite() {
                    continue;
                }
                let lag_flat: i64 = lag
                    .iter()
                    .zip(ti_strides.iter())
                    .map(|(&component, &stride)| component * stride)
                    .sum();
                let ti_value = ti[(variable, (anchor_flat as i64 + lag_flat) as usize)];
                if !ti_value.is_finite() {
                    continue;
                }
                shift_sum += sg_value - ti_value;
                common += 1;
            }
            let shift = if common > 0 {
                shift_sum / common as f64
            } else {
                0.0
            };
            (variable, center + shift)
        })
        .collect();

    NodeResult {
        flat: node_flat,
        values,
        strict_fallback_count,
        collapsed_lag_count,
    }
}

fn checked_shape_size(shape: ArrayView1<'_, i64>) -> Result<usize, String> {
    if shape.is_empty() || shape.iter().any(|&s| s <= 0) {
        return Err("MPS dimensions must be nonempty and positive".into());
    }
    shape.iter().try_fold(1_usize, |size, &s| {
        size.checked_mul(s as usize)
            .filter(|&n| n <= i64::MAX as usize)
            .ok_or_else(|| "MPS shape product overflow".into())
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_engine_configuration(
    ti_shape: ArrayView1<'_, i64>,
    sim_shape: ArrayView1<'_, i64>,
    fields: ArrayView2<'_, f64>,
    conditioned: ArrayView2<'_, u8>,
    path: ArrayView2<'_, i64>,
    offsets: ArrayView2<'_, i64>,
    matrices: ArrayView3<'_, f64>,
    neighbors: ArrayView1<'_, i64>,
    radius: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    metrics: ArrayView1<'_, i64>,
    ranges: ArrayView1<'_, f64>,
    exponents: ArrayView1<'_, f64>,
    penalty: ArrayView1<'_, f64>,
    penalty_offsets: ArrayView1<'_, i64>,
    penalty_ncat: ArrayView1<'_, i64>,
    distance_power: ArrayView1<'_, f64>,
    cond_weight: ArrayView1<'_, f64>,
    threshold: f64,
    scan_fraction: f64,
    cleanup: bool,
    active_mask: Option<ArrayView1<'_, u8>>,
) -> Result<(), String> {
    if ti_shape.len() != sim_shape.len() {
        return Err("MPS TI and simulation dimensionality must agree".into());
    }
    if neighbors.iter().any(|&n| n <= 0) {
        return Err("MPS neighbor counts must be positive".into());
    }
    if radius
        .iter()
        .any(|r| !r.is_nan() && (!r.is_finite() || *r <= 0.0))
    {
        return Err("MPS radii must be positive finite values or NaN (unlimited)".into());
    }
    crate::mps::validate_weights(weights)?;
    crate::mps::validate_metric_descriptors(
        metrics,
        ranges,
        exponents,
        penalty,
        penalty_offsets,
        penalty_ncat,
    )?;
    if distance_power
        .iter()
        .chain(cond_weight.iter())
        .any(|&v| !v.is_finite() || v < 0.0)
        || !threshold.is_finite()
        || threshold < 0.0
        || !scan_fraction.is_finite()
        || scan_fraction <= 0.0
    {
        return Err("MPS threshold/weighting parameters and scan fraction are invalid".into());
    }
    if let Some(mask) = active_mask {
        if mask.len() != fields.ncols() || mask.iter().any(|&v| v > 1) {
            return Err("MPS active mask must be boolean and match the flattened grid".into());
        }
        let strides = c_strides(sim_shape);
        for coords in path.rows() {
            let flat = flat_index_iter(coords.iter().copied(), &strides);
            if mask[flat] == 0 {
                return Err("MPS path must not visit inactive cells".into());
            }
        }
    }
    for ((variable, flat), value) in fields.indexed_iter() {
        let active = active_mask.is_none_or(|mask| mask[flat] != 0);
        if !active && (!value.is_nan() || conditioned[(variable, flat)] != 0) {
            return Err("MPS inactive fields must be NaN and cannot be conditioned".into());
        }
        if value.is_infinite() || (cleanup && active && !value.is_finite()) {
            return Err("MPS initial fields contain invalid nonfinite values".into());
        }
    }
    if conditioned
        .iter()
        .zip(fields.iter())
        .any(|(&hard, &value)| hard > 1 || (hard != 0 && !value.is_finite()))
    {
        return Err("MPS hard mask must be boolean and hard values finite".into());
    }
    let mut seen = std::collections::HashSet::with_capacity(path.nrows());
    for row in path.rows() {
        if !seen.insert(row.to_vec()) {
            return Err("MPS path must not contain duplicate nodes".into());
        }
    }
    if !cleanup {
        let strides = c_strides(sim_shape);
        let scheduled: std::collections::HashSet<usize> = path
            .rows()
            .into_iter()
            .map(|row| flat_index_iter(row.iter().copied(), &strides))
            .collect();
        for (variable, row) in fields.rows().into_iter().enumerate() {
            for (flat, value) in row.iter().enumerate() {
                if value.is_nan()
                    && active_mask.is_none_or(|mask| mask[flat] != 0)
                    && !scheduled.contains(&flat)
                {
                    return Err(format!("MPS undefined initial value for variable {variable} is absent from the path"));
                }
            }
        }
    }
    let max_shape = sim_shape.iter().copied().max().unwrap_or(1) as i128;
    let mut max_offsets = vec![0_f64; sim_shape.len()];
    let mut previous_norm = -1_f64;
    for row in offsets.rows() {
        let mut norm = 0_f64;
        if row.iter().all(|&x| x == 0) {
            return Err("MPS neighbor offsets must be nonzero".into());
        }
        for (axis, &x) in row.iter().enumerate() {
            if (x as i128).abs() + max_shape > i64::MAX as i128 / 4 {
                return Err("MPS neighbor offset arithmetic overflow".into());
            }
            max_offsets[axis] = max_offsets[axis].max((x as f64).abs());
            norm += (x as f64) * (x as f64);
        }
        if norm < previous_norm {
            return Err("MPS neighbor offsets must be ordered by distance".into());
        }
        previous_norm = norm;
    }
    if matrices.iter().any(|v| !v.is_finite()) {
        return Err("MPS lag transforms must be finite".into());
    }
    // Bounds keep rounded transformed offsets, their spread and subsequent
    // signed flattened arithmetic representable, before any worker runs.
    let stride_sum: f64 = c_strides(ti_shape).iter().map(|&s| s as f64).sum();
    let stationary_bound: f64 = max_offsets
        .iter()
        .zip(c_strides(ti_shape).iter())
        .map(|(&offset, &stride)| offset * stride as f64)
        .sum();
    if stationary_bound > i64::MAX as f64 / 4.0 {
        return Err("MPS flattened lag arithmetic overflow".into());
    }
    for matrix in matrices.outer_iter() {
        for row in matrix.rows() {
            let bound: f64 = row
                .iter()
                .zip(max_offsets.iter())
                .map(|(&a, &b)| a.abs() * b)
                .sum();
            if !bound.is_finite() || bound * stride_sum > i64::MAX as f64 / 4.0 {
                return Err("MPS transformed lag arithmetic overflow".into());
            }
        }
    }
    Ok(())
}

/// Validate the *contents* (not just the shapes) that the engines will index
/// or that must be finite, returning a controlled error instead of letting a
/// later out-of-bounds access can panic. NaN is the legitimate missing-value
/// marker and is allowed; only
/// infinities are rejected in the TI data.
fn validate_engine_inputs(
    path: ArrayView2<'_, i64>,
    sim_shape: ArrayView1<'_, i64>,
    u_start: ArrayView1<'_, f64>,
    u_fallback: ArrayView2<'_, f64>,
    ti: ArrayView2<'_, f64>,
    zone_ti: ArrayView2<'_, f64>,
) -> Result<(), String> {
    for coords in path.rows() {
        for axis in 0..sim_shape.len() {
            if coords[axis] < 0 || coords[axis] >= sim_shape[axis] {
                return Err("MPS path coordinate out of simulation-grid bounds".to_owned());
            }
        }
    }
    if u_start
        .iter()
        .chain(u_fallback.iter())
        .any(|&u| !(0.0..1.0).contains(&u))
    {
        return Err("MPS random uniforms must lie in [0, 1)".to_owned());
    }
    if ti.iter().chain(zone_ti.iter()).any(|&v| v.is_infinite()) {
        return Err("MPS training image contains a non-finite (infinite) value".to_owned());
    }
    Ok(())
}

/// Per-domain scan views shared by the main and cleanup engines.
///
/// Domain 0 is the primary TI; domains `1..=n_zones` are the zone TIs
/// (concatenated column-wise in `zone_ti`). For each domain this returns the
/// TI view, its shape, per-variable `d_max`, the jointly-defined-cell list used
/// by the unconditional fallback, and whether the TI carries any `NaN`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn build_domains<'a>(
    ti: ArrayView2<'a, f64>,
    ti_shape: ArrayView1<'a, i64>,
    d_max: ArrayView1<'a, f64>,
    _has_nan: ArrayView1<'_, u8>,
    zone_ti: ArrayView2<'a, f64>,
    zone_offsets: ArrayView1<'_, i64>,
    zone_shapes: ArrayView2<'a, i64>,
    zone_d_max: ArrayView2<'a, f64>,
    variable_count: usize,
) -> Result<
    (
        Vec<ArrayView2<'a, f64>>,
        Vec<ArrayView1<'a, i64>>,
        Vec<ArrayView1<'a, f64>>,
        Vec<Vec<usize>>,
        Vec<bool>,
        Vec<Array1<u8>>,
    ),
    String,
> {
    if zone_offsets.is_empty()
        || zone_offsets[0] != 0
        || zone_offsets
            .iter()
            .any(|&x| x < 0 || x as usize > zone_ti.ncols())
        || zone_offsets[zone_offsets.len() - 1] as usize != zone_ti.ncols()
        || zone_d_max.iter().any(|x| !x.is_finite() || *x <= 0.0)
    {
        return Err("invalid MPS zone offsets or ranges".into());
    }
    let n_zones = zone_offsets.len().saturating_sub(1);
    if zone_shapes.dim() != (n_zones, ti_shape.len())
        || zone_d_max.dim() != (n_zones, variable_count)
        || (n_zones > 0 && zone_ti.nrows() != variable_count)
    {
        return Err("inconsistent MPS zone array shapes".to_owned());
    }
    let mut domain_tis: Vec<ArrayView2<'a, f64>> = Vec::with_capacity(n_zones + 1);
    let mut domain_shapes: Vec<ArrayView1<'a, i64>> = Vec::with_capacity(n_zones + 1);
    let mut domain_d_max: Vec<ArrayView1<'a, f64>> = Vec::with_capacity(n_zones + 1);
    domain_tis.push(ti);
    domain_shapes.push(ti_shape);
    domain_d_max.push(d_max);
    for zone in 0..n_zones {
        let lo = zone_offsets[zone] as usize;
        let hi = zone_offsets[zone + 1] as usize;
        // `*_move` consumes a `Copy` of the `'a` view, so the sub-views keep the
        // input data's lifetime rather than borrowing the local binding.
        let shape = zone_shapes.index_axis_move(Axis(0), zone);
        if hi < lo || hi > zone_ti.ncols() || (hi - lo) != checked_shape_size(shape)? {
            return Err("inconsistent MPS zone TI offsets/shape".to_owned());
        }
        domain_tis.push(zone_ti.slice_move(s![.., lo..hi]));
        domain_shapes.push(shape);
        domain_d_max.push(zone_d_max.index_axis_move(Axis(0), zone));
    }
    let mut domain_finite_all: Vec<Vec<usize>> = Vec::with_capacity(n_zones + 1);
    let mut domain_ti_has_nan: Vec<bool> = Vec::with_capacity(n_zones + 1);
    // Per-domain, per-variable missingness: the masked/unmasked
    // kernel choice must follow the SELECTED domain's data, not the primary's.
    // All flags are recomputed from the selected TI, including the primary,
    // so raw callers cannot accidentally select an unmasked kernel for NaN data.
    let mut domain_has_nan: Vec<Array1<u8>> = Vec::with_capacity(n_zones + 1);
    for dti in &domain_tis {
        let has_nan_d = dti.iter().any(|value| value.is_nan());
        let finite_d: Vec<usize> = (0..dti.ncols())
            .filter(|&flat| (0..variable_count).all(|v| !dti[(v, flat)].is_nan()))
            .collect();
        if has_nan_d && finite_d.is_empty() {
            return Err("training image has no jointly defined cell".to_owned());
        }
        domain_ti_has_nan.push(has_nan_d);
        domain_finite_all.push(finite_d);
        domain_has_nan
            .push(Array1::from_iter((0..variable_count).map(|v| {
                u8::from((0..dti.ncols()).any(|flat| dti[(v, flat)].is_nan()))
            })));
    }
    Ok((
        domain_tis,
        domain_shapes,
        domain_d_max,
        domain_finite_all,
        domain_ti_has_nan,
        domain_has_nan,
    ))
}

/// Run one complete MPS simulation with a deterministic scheduler.
///
/// Python supplies the simulation path and per-node random values so seeded
/// behavior follows the frontend random-stream contract. Rust owns all
/// per-node work. The
/// path-derived neighbour graph is converted to fixed dependency levels;
/// nodes inside one level are independent, calculated in parallel, and then
/// committed in path order. Candidate/lag reductions remain sequential.
///
/// `max_radius` uses `NaN` to represent Python's `None`. `metric_kinds` uses
/// `0` for categorical, `1` for continuous Lp, `2` for variation, and `3` for
/// categorical with a per-variable penalty matrix.
/// `partial_boundary == false` means strict mode with the established partial
/// fallback when the complete event cannot fit.
///
/// # Arguments
///
/// * `ti` – Float64 TI rows, shape `(n_variables, ti_size)`.
/// * `ti_shape` – Shared TI shape.
/// * `initial_fields` – Initial SG rows with unknown cells as `NaN`.
/// * `conditioned` – Conditioning mask, shape matching `initial_fields`.
/// * `sim_shape` – Shared SG shape.
/// * `path` – Pre-generated unknown-node path, shape `(n_nodes, dim)`.
/// * `lag_matrices` – Optional per-node SG-to-TI matrices, shape `(n_nodes, dim, dim)`;
///   an empty first axis selects the stationary identity path.
/// * `u_start` – Per-node scan-start uniforms.
/// * `u_fallback` – Per-node fallback uniforms, shape `(n_nodes, dim)`.
/// * `offsets` – Canonically ordered nonzero SG offsets.
/// * `n_neighbors` – Maximum event size per variable.
/// * `max_radius` – Radius per variable (`NaN` means unlimited).
/// * `variable_weights` – Normalized multivariate weights.
/// * `metric_kinds` – Distance kind per variable.
/// * `has_nan` – Per-variable masked-distance flags.
/// * `d_max` – Continuous normalization ranges.
/// * `p_norm` – Lp or variation exponent per variable.
/// * `threshold` – Best within the capped candidate budget when zero;
///   strict early acceptance for a positive DS threshold.
/// * `scan_fraction` – Candidate budget relative to selected TI size, capped
///   to the valid window; cleanup effort may produce a value above one.
/// * `distance_power` – Spatial event-weight exponent.
/// * `cond_weight` – Conditioning-event weight multiplier.
/// * `partial_boundary` – Select partial instead of strict-first geometry.
/// * `num_threads` – Size of the one simulation-local Rayon pool.
///
/// # Returns
///
/// Simulated fields plus deterministic scheduler diagnostics, or an error for
/// inconsistent internal inputs or a generated `NaN` value.
#[allow(clippy::too_many_arguments)]
pub fn simulate_engine(
    ti: ArrayView2<'_, f64>,
    ti_shape: ArrayView1<'_, i64>,
    initial_fields: ArrayView2<'_, f64>,
    conditioned: ArrayView2<'_, u8>,
    sim_shape: ArrayView1<'_, i64>,
    path: ArrayView2<'_, i64>,
    lag_matrices: ArrayView3<'_, f64>,
    u_start: ArrayView1<'_, f64>,
    u_fallback: ArrayView2<'_, f64>,
    offsets: ArrayView2<'_, i64>,
    n_neighbors: ArrayView1<'_, i64>,
    max_radius: ArrayView1<'_, f64>,
    variable_weights: ArrayView1<'_, f64>,
    metric_kinds: ArrayView1<'_, i64>,
    has_nan: ArrayView1<'_, u8>,
    d_max: ArrayView1<'_, f64>,
    p_norm: ArrayView1<'_, f64>,
    penalty_flat: ArrayView1<'_, f64>,
    penalty_offsets: ArrayView1<'_, i64>,
    penalty_ncat: ArrayView1<'_, i64>,
    threshold: f64,
    scan_fraction: f64,
    random_scan: bool,
    distance_power: ArrayView1<'_, f64>,
    cond_weight: ArrayView1<'_, f64>,
    partial_boundary: bool,
    num_threads: usize,
    zone_ti: ArrayView2<'_, f64>,
    zone_offsets: ArrayView1<'_, i64>,
    zone_shapes: ArrayView2<'_, i64>,
    zone_d_max: ArrayView2<'_, f64>,
    node_domain: ArrayView1<'_, i64>,
    active_mask: Option<ArrayView1<'_, u8>>,
) -> Result<SimulationOutput, String> {
    let variable_count = ti.nrows();
    let ti_size = checked_shape_size(ti_shape)?;
    let sg_size = checked_shape_size(sim_shape)?;
    if variable_count == 0
        || ti.ncols() != ti_size
        || initial_fields.dim() != (variable_count, sg_size)
        || conditioned.dim() != initial_fields.dim()
        || path.ncols() != sim_shape.len()
        || (!lag_matrices.is_empty()
            && lag_matrices.dim() != (path.nrows(), sim_shape.len(), sim_shape.len()))
        || u_start.len() != path.nrows()
        || u_fallback.dim() != (path.nrows(), sim_shape.len())
        || offsets.ncols() != sim_shape.len()
        || n_neighbors.len() != variable_count
        || max_radius.len() != variable_count
        || variable_weights.len() != variable_count
        || metric_kinds.len() != variable_count
        || has_nan.len() != variable_count
        || d_max.len() != variable_count
        || p_norm.len() != variable_count
        || distance_power.len() != variable_count
        || cond_weight.len() != variable_count
        || penalty_offsets.len() != variable_count
        || penalty_ncat.len() != variable_count
        || node_domain.len() != path.nrows()
    {
        return Err("inconsistent MPS simulation-engine array shapes".to_owned());
    }
    validate_engine_inputs(path, sim_shape, u_start, u_fallback, ti, zone_ti)?;
    validate_engine_configuration(
        ti_shape,
        sim_shape,
        initial_fields,
        conditioned,
        path,
        offsets,
        lag_matrices,
        n_neighbors,
        max_radius,
        variable_weights,
        metric_kinds,
        d_max,
        p_norm,
        penalty_flat,
        penalty_offsets,
        penalty_ncat,
        distance_power,
        cond_weight,
        threshold,
        scan_fraction,
        false,
        active_mask,
    )?;

    // Domain 0 is the primary TI; domains 1.. are the zone TIs. `node_domain`
    // selects a domain per path node. A zero-length `node_domain`/`zone_*`
    // configuration (no zones) leaves every node on the primary TI, so the
    // whole run is identical to the single-TI path.
    let n_zones = zone_offsets.len().saturating_sub(1);
    if node_domain.iter().any(|&d| d < 0 || d as usize > n_zones) {
        return Err("inconsistent MPS zone array shapes".to_owned());
    }
    let (
        domain_tis,
        domain_shapes,
        domain_d_max,
        domain_finite_all,
        domain_ti_has_nan,
        domain_has_nan,
    ) = build_domains(
        ti,
        ti_shape,
        d_max,
        has_nan,
        zone_ti,
        zone_offsets,
        zone_shapes,
        zone_d_max,
        variable_count,
    )?;

    let cache = build_neighbor_cache(
        path,
        sim_shape,
        offsets,
        conditioned,
        n_neighbors,
        max_radius,
        active_mask,
    );
    let level_count = cache
        .iter()
        .map(|node| node.level)
        .max()
        .map_or(0, |level| level + 1);
    let mut levels = vec![Vec::new(); level_count];
    for (node, cached) in cache.iter().enumerate() {
        levels[cached.level].push(node);
    }
    let max_ready_width = levels.iter().map(Vec::len).max().unwrap_or(0);
    // Building/waking a pool regressed the measured 256-node small case. Keep
    // paths below 512 nodes serial to avoid pool setup overhead.
    let used_threads = if num_threads > 1 && path.nrows() >= 512 {
        num_threads
    } else {
        1
    };
    let pool: Option<ThreadPool> = if used_threads > 1 {
        Some(
            ThreadPoolBuilder::new()
                .num_threads(used_threads)
                .build()
                .map_err(|error| format!("could not build MPS Rayon pool: {error}"))?,
        )
    } else {
        None
    };

    let mut fields = initial_fields.to_owned();
    let mut strict_fallback_count = 0_usize;
    let mut collapsed_lag_count = 0_usize;
    for ready in &levels {
        let calculate = |node: &usize| {
            // Pick this node's domain (0 = primary TI, k = zones[k-1]); only the
            // scanned TI, its shape, range, missingness and defined-cell list
            // change. Neighbour gathering (build_neighbor_cache) is grid-based
            // and zone-agnostic, so data events cross zone boundaries.
            let domain = node_domain[*node] as usize;
            simulate_node(
                *node,
                &cache[*node],
                fields.view(),
                domain_tis[domain],
                domain_shapes[domain],
                sim_shape,
                path,
                lag_matrices,
                u_start,
                u_fallback,
                conditioned,
                variable_weights,
                metric_kinds,
                domain_has_nan[domain].view(),
                domain_d_max[domain],
                p_norm,
                penalty_flat,
                penalty_offsets,
                penalty_ncat,
                threshold,
                scan_fraction,
                random_scan,
                distance_power,
                cond_weight,
                partial_boundary,
                &domain_finite_all[domain],
                domain_ti_has_nan[domain],
                false,
            )
        };
        let results: Vec<NodeResult> = if let Some(pool) = &pool {
            pool.install(|| ready.par_iter().map(calculate).collect())
        } else {
            ready.iter().map(calculate).collect()
        };
        for result in results {
            strict_fallback_count += result.strict_fallback_count;
            collapsed_lag_count += result.collapsed_lag_count;
            for (variable, value) in result.values {
                if !value.is_finite() {
                    return Err(format!(
                        "simulation produced a non-finite value for variable {variable} at flat node {}",
                        result.flat
                    ));
                }
                fields[(variable, result.flat)] = value;
            }
        }
    }

    Ok(SimulationOutput {
        fields,
        strict_fallback_count,
        level_count,
        max_ready_width,
        used_threads,
        collapsed_lag_count,
    })
}

/// Run one deterministic post-processing (cleanup) pass (Meerschman et al.
/// 2013, Sect. 4).
///
/// Re-simulates every node of `path` against an already-complete field as a
/// Gauss-Seidel sweep: each node reads the refreshed values of earlier-visited
/// neighbours and the stale values of later ones, so one pass refines the
/// realization instead of redrawing it. The sweep is expressed as the
/// position-oriented dependency DAG of [`build_cleanup_levels`] and run with
/// the same level scheduler as [`simulate_engine`], so the output is
/// **bit-identical for any `num_threads`** (serial == parallel).
///
/// Each node's non-conditioned variables are its scan targets (their stale
/// value suppressed as a self-constraint via [`simulate_node`]'s `cleanup`
/// mode); conditioned (hard) variables are never rewritten and still act as
/// `h = 0` collocated constraints. The search config (TI, metrics, weights,
/// `n_neighbors`, `scan_fraction`, penalties) is the caller's *effective*
/// post-pass config, so a `metric_kinds`/paste-rule override (e.g.
/// variation↔Lp) takes effect here, including the variation mean-shift which
/// [`simulate_node`] keys off `metric_kinds`.
///
/// `fields_in` is the complete current field; the returned array is the updated
/// field. Arguments mirror [`simulate_engine`]; `path` is this pass's
/// eligible-node visit order and `u_start`/`u_fallback` are its per-node draws.
/// Returns the field plus the strict-fallback count, collapsed-lag count, and
/// the effective thread count after the small-path cutoff.
#[allow(clippy::too_many_arguments)]
pub fn cleanup_pass_engine(
    ti: ArrayView2<'_, f64>,
    ti_shape: ArrayView1<'_, i64>,
    fields_in: ArrayView2<'_, f64>,
    conditioned: ArrayView2<'_, u8>,
    sim_shape: ArrayView1<'_, i64>,
    path: ArrayView2<'_, i64>,
    lag_matrices: ArrayView3<'_, f64>,
    u_start: ArrayView1<'_, f64>,
    u_fallback: ArrayView2<'_, f64>,
    offsets: ArrayView2<'_, i64>,
    n_neighbors: ArrayView1<'_, i64>,
    max_radius: ArrayView1<'_, f64>,
    variable_weights: ArrayView1<'_, f64>,
    metric_kinds: ArrayView1<'_, i64>,
    has_nan: ArrayView1<'_, u8>,
    d_max: ArrayView1<'_, f64>,
    p_norm: ArrayView1<'_, f64>,
    penalty_flat: ArrayView1<'_, f64>,
    penalty_offsets: ArrayView1<'_, i64>,
    penalty_ncat: ArrayView1<'_, i64>,
    threshold: f64,
    scan_fraction: f64,
    random_scan: bool,
    distance_power: ArrayView1<'_, f64>,
    cond_weight: ArrayView1<'_, f64>,
    partial_boundary: bool,
    num_threads: usize,
    zone_ti: ArrayView2<'_, f64>,
    zone_offsets: ArrayView1<'_, i64>,
    zone_shapes: ArrayView2<'_, i64>,
    zone_d_max: ArrayView2<'_, f64>,
    node_domain: ArrayView1<'_, i64>,
    active_mask: Option<ArrayView1<'_, u8>>,
) -> Result<(Array2<f64>, usize, usize, usize), String> {
    let variable_count = ti.nrows();
    let ti_size = checked_shape_size(ti_shape)?;
    let sg_size = checked_shape_size(sim_shape)?;
    if variable_count == 0
        || ti.ncols() != ti_size
        || fields_in.dim() != (variable_count, sg_size)
        || conditioned.dim() != fields_in.dim()
        || path.ncols() != sim_shape.len()
        || (!lag_matrices.is_empty()
            && lag_matrices.dim() != (path.nrows(), sim_shape.len(), sim_shape.len()))
        || u_start.len() != path.nrows()
        || u_fallback.dim() != (path.nrows(), sim_shape.len())
        || offsets.ncols() != sim_shape.len()
        || n_neighbors.len() != variable_count
        || max_radius.len() != variable_count
        || variable_weights.len() != variable_count
        || metric_kinds.len() != variable_count
        || has_nan.len() != variable_count
        || d_max.len() != variable_count
        || p_norm.len() != variable_count
        || distance_power.len() != variable_count
        || cond_weight.len() != variable_count
        || penalty_offsets.len() != variable_count
        || penalty_ncat.len() != variable_count
        || node_domain.len() != path.nrows()
    {
        return Err("inconsistent MPS cleanup-engine array shapes".to_owned());
    }
    validate_engine_inputs(path, sim_shape, u_start, u_fallback, ti, zone_ti)?;
    validate_engine_configuration(
        ti_shape,
        sim_shape,
        fields_in,
        conditioned,
        path,
        offsets,
        lag_matrices,
        n_neighbors,
        max_radius,
        variable_weights,
        metric_kinds,
        d_max,
        p_norm,
        penalty_flat,
        penalty_offsets,
        penalty_ncat,
        distance_power,
        cond_weight,
        threshold,
        scan_fraction,
        true,
        active_mask,
    )?;
    let n_zones = zone_offsets.len().saturating_sub(1);
    if node_domain.iter().any(|&d| d < 0 || d as usize > n_zones) {
        return Err("inconsistent MPS zone array shapes".to_owned());
    }
    let (
        domain_tis,
        domain_shapes,
        domain_d_max,
        domain_finite_all,
        domain_ti_has_nan,
        domain_has_nan,
    ) = build_domains(
        ti,
        ti_shape,
        d_max,
        has_nan,
        zone_ti,
        zone_offsets,
        zone_shapes,
        zone_d_max,
        variable_count,
    )?;

    // One neighbour cache feeds both the dependency DAG and the per-node scan.
    let cache = build_cleanup_neighbor_cache(
        path,
        sim_shape,
        offsets,
        n_neighbors,
        max_radius,
        variable_count,
        active_mask,
    );
    let (levels, _level_count, _max_width) = build_cleanup_levels(&cache, path, sim_shape);

    // Same small-path cutoff and one-pool policy as simulate_engine.
    let used_threads = if num_threads > 1 && path.nrows() >= 512 {
        num_threads
    } else {
        1
    };
    let pool: Option<ThreadPool> = if used_threads > 1 {
        Some(
            ThreadPoolBuilder::new()
                .num_threads(used_threads)
                .build()
                .map_err(|error| format!("could not build MPS Rayon pool: {error}"))?,
        )
    } else {
        None
    };

    let mut fields = fields_in.to_owned();
    let mut strict_fallback_count = 0_usize;
    let mut collapsed_lag_count = 0_usize;
    for ready in &levels {
        let calculate = |node: &usize| {
            let domain = node_domain[*node] as usize;
            simulate_node(
                *node,
                &cache[*node],
                fields.view(),
                domain_tis[domain],
                domain_shapes[domain],
                sim_shape,
                path,
                lag_matrices,
                u_start,
                u_fallback,
                conditioned,
                variable_weights,
                metric_kinds,
                domain_has_nan[domain].view(),
                domain_d_max[domain],
                p_norm,
                penalty_flat,
                penalty_offsets,
                penalty_ncat,
                threshold,
                scan_fraction,
                random_scan,
                distance_power,
                cond_weight,
                partial_boundary,
                &domain_finite_all[domain],
                domain_ti_has_nan[domain],
                true,
            )
        };
        let results: Vec<NodeResult> = if let Some(pool) = &pool {
            pool.install(|| ready.par_iter().map(calculate).collect())
        } else {
            ready.iter().map(calculate).collect()
        };
        // Commit after the level: within a level no node reads another (an edge
        // would force distinct levels), so the snapshot the closure read is the
        // correct Gauss-Seidel state and writes are to disjoint cells.
        for result in results {
            strict_fallback_count += result.strict_fallback_count;
            collapsed_lag_count += result.collapsed_lag_count;
            for (variable, value) in result.values {
                if !value.is_finite() {
                    return Err(format!(
                        "cleanup produced a non-finite value for variable {variable} at flat node {}",
                        result.flat
                    ));
                }
                fields[(variable, result.flat)] = value;
            }
        }
    }

    Ok((
        fields,
        strict_fallback_count,
        collapsed_lag_count,
        used_threads,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array3};

    fn categorical_fixture(num_threads: usize) -> SimulationOutput {
        simulate_engine(
            array![[0.0, 1.0, 1.0, 0.0]].view(),
            array![4_i64].view(),
            array![[0.0, f64::NAN, f64::NAN]].view(),
            array![[1_u8, 0, 0]].view(),
            array![3_i64].view(),
            array![[1_i64], [2_i64]].view(),
            Array3::<f64>::zeros((0, 1, 1)).view(),
            array![0.0, 0.5].view(),
            array![[0.25], [0.75]].view(),
            array![[-1_i64], [1_i64]].view(),
            array![1_i64].view(),
            array![f64::NAN].view(),
            array![1.0].view(),
            array![0_i64].view(),
            array![0_u8].view(),
            array![1.0].view(),
            array![1.0].view(),
            Array1::<f64>::zeros(0).view(),
            array![0_i64].view(),
            array![0_i64].view(),
            0.0,
            1.0,
            false,
            array![0.0].view(),
            array![1.0].view(),
            false,
            num_threads,
            Array2::<f64>::zeros((1, 0)).view(),
            array![0_i64].view(),
            Array2::<i64>::zeros((0, 1)).view(),
            Array2::<f64>::zeros((0, 1)).view(),
            array![0_i64, 0_i64].view(),
            None,
        )
        .expect("valid fixture")
    }

    #[test]
    fn active_mask_excludes_holes_from_main_and_cleanup_neighbors() {
        let path = array![[0_i64], [2_i64]];
        let shape = array![3_i64];
        let offsets = array![[-1_i64], [1_i64], [-2_i64], [2_i64]];
        let conditioned = array![[0_u8, 0, 0]];
        let neighbors = array![2_i64];
        let radius = array![f64::NAN];
        let mask = array![1_u8, 0, 1];
        let cache = build_neighbor_cache(
            path.view(),
            shape.view(),
            offsets.view(),
            conditioned.view(),
            neighbors.view(),
            radius.view(),
            Some(mask.view()),
        );
        assert!(cache[0].by_variable[0].is_empty());
        assert_eq!(cache[1].by_variable[0].len(), 1);
        assert_eq!(cache[1].by_variable[0][0].flat, 0);
        let cleanup = build_cleanup_neighbor_cache(
            path.view(),
            shape.view(),
            offsets.view(),
            neighbors.view(),
            radius.view(),
            1,
            Some(mask.view()),
        );
        assert_eq!(cleanup[0].by_variable[0].len(), 1);
        assert_eq!(cleanup[0].by_variable[0][0].flat, 2);
        assert_eq!(cleanup[1].by_variable[0][0].flat, 0);
    }

    #[test]
    fn stationary_engine_preserves_conditioning_and_is_thread_independent() {
        let serial = categorical_fixture(1);
        let parallel = categorical_fixture(4);
        assert_eq!(serial.fields, parallel.fields);
        assert_eq!(serial.fields[(0, 0)], 0.0);
        assert!(serial.fields.iter().all(|value| value.is_finite()));
        assert_eq!(serial.level_count, parallel.level_count);
        assert_eq!(parallel.used_threads, 1);
    }

    #[test]
    fn transformed_event_keeps_first_duplicate_and_reduces_to_fit() {
        let mut event = Event {
            values: vec![5.0, 9.0, 12.0],
            conditioned: vec![true, false, false],
            lags: vec![vec![0, 1], vec![0, 2], vec![10, 0]],
            lag_norms: vec![1.0, 2.0, 10.0],
        };
        let collapsed = transform_and_reduce_event(
            &mut event,
            array![[1.0, 0.0], [0.0, 0.1]].view(),
            array![4_i64, 4_i64].view(),
        );
        assert_eq!(collapsed, 1);
        assert_eq!(event.lags, vec![vec![0, 0]]);
        assert_eq!(event.values, vec![5.0]);
        assert_eq!(event.conditioned, vec![true]);
    }

    #[test]
    fn strict_window_reports_fallback_even_when_no_prefix_fits() {
        let lags = vec![vec![10_i64]];
        let (bounds, strict_fallback) = window_bounds(&lags, array![4_i64].view(), false);
        assert!(bounds.is_none());
        assert!(strict_fallback);

        let (bounds, strict_fallback) = window_bounds(&lags, array![4_i64].view(), true);
        assert!(bounds.is_none());
        assert!(!strict_fallback);
    }

    #[allow(clippy::too_many_arguments)]
    fn cleanup_fixture(u_start: Array1<f64>, num_threads: usize) -> (Array2<f64>, usize, usize) {
        let (fields, strict, collapsed, _used) = cleanup_pass_engine(
            array![[0.0, 1.0, 1.0, 0.0]].view(),
            array![4_i64].view(),
            array![[0.0, 0.0, 1.0]].view(),
            array![[1_u8, 0, 0]].view(),
            array![3_i64].view(),
            array![[1_i64], [2_i64]].view(),
            Array3::<f64>::zeros((0, 1, 1)).view(),
            u_start.view(),
            array![[0.25], [0.75]].view(),
            array![[-1_i64], [1_i64]].view(),
            array![1_i64].view(),
            array![f64::NAN].view(),
            array![1.0].view(),
            array![0_i64].view(),
            array![0_u8].view(),
            array![1.0].view(),
            array![1.0].view(),
            Array1::<f64>::zeros(0).view(),
            array![0_i64].view(),
            array![0_i64].view(),
            0.0,
            1.0,
            false,
            array![0.0].view(),
            array![1.0].view(),
            false,
            num_threads,
            Array2::<f64>::zeros((1, 0)).view(),
            array![0_i64].view(),
            Array2::<i64>::zeros((0, 1)).view(),
            Array2::<f64>::zeros((0, 1)).view(),
            array![0_i64, 0_i64].view(),
            None,
        )
        .expect("valid cleanup fixture");
        (fields, strict, collapsed)
    }

    #[test]
    fn cleanup_pass_preserves_hard_data_and_is_gauss_seidel() {
        let (fields, strict, collapsed) = cleanup_fixture(array![0.25, 0.5], 1);
        // Node 0 is hard-conditioned and must be untouched.
        assert_eq!(fields[(0, 0)], 0.0);
        // Node 1 reopens, its only neighbour is the hard 0 at lag -1; the unique
        // distance-0 TI anchor is y=1 (TI[0]==0), pasting TI[1]==1. Node 2 then
        // reads node 1's refreshed 1 (Gauss-Seidel) and pastes 1 as well.
        assert_eq!(fields, array![[0.0, 1.0, 1.0]]);
        assert_eq!(strict, 0);
        assert_eq!(collapsed, 0);
    }

    #[test]
    fn cleanup_pass_is_deterministic() {
        let a = cleanup_fixture(array![0.25, 0.5], 1).0;
        let b = cleanup_fixture(array![0.25, 0.5], 1).0;
        assert_eq!(a, b);
    }

    #[test]
    fn compute_node_weights_handles_overflow_and_underflow() {
        // F1: a large finite cond_weight overflows the raw sum; the two hard
        // nodes must still normalize to 0.5 each, not a uniform third.
        let over = compute_node_weights(&[1.0, 1.0, 1.0], &[true, true, false], 0.0, 1e308);
        assert!((over[0] - 0.5).abs() < 1e-9);
        assert!((over[1] - 0.5).abs() < 1e-9);
        assert!(over[2] > 0.0 && over[2] < 1e-300);
        // A large distance_power underflows the far node to 0; the near node
        // must keep weight 1, not collapse the whole event to zero.
        let under = compute_node_weights(&[2.0, 3.0], &[false, false], 2048.0, 1.0);
        assert_eq!(under, vec![1.0, 0.0]);
        // Ordinary inputs stay on the direct path and sum to one.
        let ordinary = compute_node_weights(&[1.0, 2.0], &[false, false], 1.0, 1.0);
        assert!((ordinary.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // Intentional all-conditioning zero-weight event stays all-zero.
        let zeroed = compute_node_weights(&[0.0, 1.0], &[true, true], 0.0, 0.0);
        assert_eq!(zeroed, vec![0.0, 0.0]);
    }

    #[test]
    fn cleanup_levels_order_mutual_neighbours_by_visit_position() {
        // Three colinear eligible nodes 0,1,2 at positions 0,1,2 with nearest
        // neighbours: each adjacent pair is an edge, so levels are 0,1,2 (a
        // chain) -- the Gauss-Seidel sweep cannot parallelise colinear reads.
        let path = array![[0_i64], [1_i64], [2_i64]];
        let cache = build_cleanup_neighbor_cache(
            path.view(),
            array![3_i64].view(),
            array![[-1_i64], [1_i64]].view(),
            array![2_i64].view(),
            array![f64::NAN].view(),
            1,
            None,
        );
        let (levels, level_count, max_width) =
            build_cleanup_levels(&cache, path.view(), array![3_i64].view());
        assert_eq!(level_count, 3);
        assert_eq!(max_width, 1);
        assert_eq!(levels, vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn variation_retrieval_applies_mean_shift() {
        let output = simulate_engine(
            array![[10.0, 11.0, 20.0, 21.0]].view(),
            array![4_i64].view(),
            array![[5.0, f64::NAN]].view(),
            array![[1_u8, 0]].view(),
            array![2_i64].view(),
            array![[1_i64]].view(),
            Array3::<f64>::zeros((0, 1, 1)).view(),
            array![0.0].view(),
            array![[0.0]].view(),
            array![[-1_i64], [1_i64]].view(),
            array![1_i64].view(),
            array![f64::NAN].view(),
            array![1.0].view(),
            array![2_i64].view(),
            array![0_u8].view(),
            array![11.0].view(),
            array![2.0].view(),
            Array1::<f64>::zeros(0).view(),
            array![0_i64].view(),
            array![0_i64].view(),
            1.0,
            1.0,
            false,
            array![0.0].view(),
            array![1.0].view(),
            false,
            1,
            Array2::<f64>::zeros((1, 0)).view(),
            array![0_i64].view(),
            Array2::<i64>::zeros((0, 1)).view(),
            Array2::<f64>::zeros((0, 1)).view(),
            array![0_i64].view(),
            None,
        )
        .expect("valid variation fixture");
        assert_eq!(output.fields[(0, 1)], 6.0);
    }
}
