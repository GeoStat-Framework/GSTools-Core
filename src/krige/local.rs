//! Local (moving-neighborhood) kriging of one or many fields.
//!
//! Every target point is kriged from the conditioning points within
//! `local_radius`. Positions are assumed already isometrized (anisotropy and
//! rotation stay on the Python side), so every distance is a plain Euclidean
//! norm. Unlike global kriging (see [`crate::krige::global`]) every
//! neighborhood is a different small system that needs its own solve.
//!
//! The conditioning values are a matrix `(n, T)`: one column per field (e.g.
//! one time step of measurements at the conditioning points), NaN marks a
//! missing value in that field. A
//! single field is the case `T = 1`.
//!
//! For one target point the kriging weights depend on positions, covariance
//! model, drift and *which neighbors have data* -- never on the values. So
//! they are computed once per neighbor mask and reused for all fields with
//! that mask:
//!
//! ```text
//! for each target j (in parallel):
//!     neighbors, full covariance matrix, drift      <- once per target
//!     cache = {}                                    <- per target only
//!     for each field t:
//!         key = which neighbors have data at t      <- bitset
//!         weights = cache[key] or solve(sub-system of the full matrix)
//!         field[j, t] = weights . values
//! ```
//!
//! Each target is handled by one task from start to end and writes only
//! its own output row, so there is no synchronization between threads,
//! and the cache only lives while that target is processed.
//!
//! # Layout of a local system
//!
//! A system with `k` neighbors and `p` constraints is
//!
//! ```text
//! [ C   F ] [ w  ]   [ r ]      C: (k, k) covariance between neighbors (+ cond_err)
//! [ F^T 0 ] [ mu ] = [ f ]      F: (k, p) constraint columns
//!                               r: (k,)   covariance between neighbors and target
//!                               f: (p,)   constraint values at the target
//! ```
//!
//! Constraints are always ordered **unbiasedness first** (a column of ones,
//! `f = 1`), **then the drift terms** (drift values at the neighbors, `f` =
//! drift values at the target). It is solved by [`solve_saddle_point`]
//! (Cholesky + Schur complement, dense fallback; see
//! [`crate::krige::solve`]).
//!
//! # Small neighborhoods
//!
//! ```text
//! n = neighbors with data in this field,  p = unbiased + drift terms
//!
//! n < min_neighbors  ->  NaN  (n = 0: always NaN, n_used = 0, no error)
//! n <= p             ->  ALL DRIFT TERMS DROPPED: ordinary kriging
//!                        (simple kriging if not unbiased)
//! otherwise          ->  kriging as requested
//! ```
//!
//! Which case applies depends only on `n`, so it is part of the cached
//! solution of a neighbor mask. The rule lives in [`drift_terms_used`].
//!
//! # Too few neighbors for the drift: why the drift is dropped
//!
//! With `n <= p` the constraints alone fix the weights and the covariance
//! model has no influence. For example external drift kriging with two
//! neighbors (`p = 2`) solves `w1 + w2 = 1`, `w1 h1 + w2 h2 = h0`, so
//! `w1 = (h0 - h2) / (h1 - h2)`: a straight line through two points,
//! singular for `h1 = h2` and extrapolating (negative weights) whenever
//! `h0` lies outside `[h1, h2]`. With one neighbor there is no solution at
//! all. Ordinary kriging is well-defined for any `n >= 1`. The returned
//! kriging variance is the one of the system actually solved.

use std::collections::HashMap;

use ndarray::{Array2, ArrayView1, ArrayView2, ArrayViewMut1, Axis, Zip};
use rayon::prelude::*;

use crate::covmodel::CovModel;
use crate::krige::neighbors::Neighbors;
use crate::krige::solve::solve_saddle_point;
use crate::krige::KrigeError;

/// Number of drift terms to use for a neighborhood of `n_neighbors`
/// neighbors with data.
///
/// **All drift terms are dropped** (returns 0) if there are not more
/// neighbors than constraints, i.e. `n_neighbors <= unbiased + drift_no`.
/// The system is then ordinary kriging (or simple kriging if `unbiased` is
/// false). Otherwise all `drift_no` terms are used. Without drift terms
/// nothing changes.
///
/// The unbiasedness condition is never dropped. Constraints are always
/// ordered unbiasedness first, then drift, so the reduced system is the
/// full one without its last `drift_no` rows/columns.
pub fn drift_terms_used(n_neighbors: usize, unbiased: bool, drift_no: usize) -> usize {
    if n_neighbors <= unbiased as usize + drift_no {
        0
    } else {
        drift_no
    }
}

/// Euclidean distance between two points (columns of a position array).
pub(crate) fn euclidean_dist(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> f64 {
    Zip::from(a)
        .and(b)
        .fold(0.0, |acc, &x, &y| acc + (x - y) * (x - y))
        .sqrt()
}

/// Counters describing the work done (how much the cache saved, how many
/// values were left NaN or kriged without drift).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LocalKrigeStats {
    /// Number of kriging systems actually solved.
    pub n_solves: usize,
    /// Fields that reused the previous field's weights (same neighbor mask).
    pub n_hits_last: usize,
    /// Fields that reused weights of an earlier mask from the per-target
    /// hash map.
    pub n_hits_hash: usize,
    /// (target, field) pairs left as NaN (fewer than `min_neighbors`,
    /// including targets without neighbors).
    pub n_skipped: usize,
    /// (target, field) pairs kriged **without drift** because there were
    /// not more neighbors with data than constraints (see
    /// [`drift_terms_used`]). Always 0 without drift terms.
    pub n_without_drift: usize,
    /// Largest number of distinct masks cached for one target.
    pub max_masks_per_target: usize,
}

impl LocalKrigeStats {
    fn merge(self, other: Self) -> Self {
        Self {
            n_solves: self.n_solves + other.n_solves,
            n_hits_last: self.n_hits_last + other.n_hits_last,
            n_hits_hash: self.n_hits_hash + other.n_hits_hash,
            n_skipped: self.n_skipped + other.n_skipped,
            n_without_drift: self.n_without_drift + other.n_without_drift,
            max_masks_per_target: self.max_masks_per_target.max(other.max_masks_per_target),
        }
    }
}

/// Output of [`calc_field_krige_local`]. All arrays have the shape
/// `(m, T)`: one row per target point, one column per field.
#[derive(Debug, Clone)]
pub struct LocalKrigeResult {
    /// Raw kriging estimate, NaN where fewer than `min_neighbors` were
    /// available (always NaN without neighbors). Kriged without drift where
    /// `n_used <= unbiased + drift terms` (see [`drift_terms_used`]).
    pub field: Array2<f64>,
    /// Raw `rhs . weights` (not yet `sill - error`), NaN like `field`.
    pub error: Array2<f64>,
    /// Number of available neighbors per target and field. Tells which
    /// rule of the module docs applied (NaN, without drift, as requested).
    pub n_used: Array2<u32>,
    /// Statistics.
    pub stats: LocalKrigeStats,
}

/// Evaluate local kriging at every target point for one or many fields
/// sharing the same conditioning positions, reusing the kriging weights per
/// target point and neighbor mask.
///
/// # Arguments
/// * `cond_pos` - isometrized conditioning positions, shape `(dim, n)`
/// * `cond_val` - centered conditioning values, shape `(n, T)` (one column
///   per field, `T = 1` for a single field); non-finite values (NaN) mark a
///   missing value for that field only
/// * `target_pos` - isometrized target positions, shape `(dim, m)`
/// * `model` - covariance model
/// * `cond_err` - measurement error per conditioning point, shape `(n,)`
/// * `drift_cond` - drift terms at the conditioning points, shape `(p, n)`
///   (`p` may be 0)
/// * `drift_target` - drift terms at the target points, shape `(p, m)`
/// * `unbiased` - whether to add the unbiasedness constraint
/// * `exact` - whether to use `model.cov_nugget` on the RHS, so a target
///   point that coincides with a neighbor reproduces its value exactly
/// * `local_radius` - search radius (isometrized distance); every
///   conditioning point within this distance enters the local system
/// * `min_neighbors` - fields with fewer available neighbors are left NaN
///   (values below 1 are treated as 1)
/// * `num_threads` - number of threads, `None` for rayon's global pool
///
/// # Small neighborhoods
/// Per target and field, with `n` neighbors with data and
/// `p = unbiased + drift terms` constraints:
/// * `n < min_neighbors`: NaN. A target without any neighbor is NaN with
///   `n_used = 0`, not an error.
/// * `n <= p`: **all drift terms are dropped**, the value is ordinary
///   kriging (simple kriging if `unbiased` is false). Counted in
///   [`LocalKrigeStats::n_without_drift`]. See [`drift_terms_used`].
/// * otherwise: kriging as requested.
///
/// # Returns
/// A [`LocalKrigeResult`]; errors only for invalid input or a system that
/// even the least-squares fallback cannot solve.
#[allow(clippy::too_many_arguments)]
pub fn calc_field_krige_local(
    cond_pos: ArrayView2<'_, f64>,
    cond_val: ArrayView2<'_, f64>,
    target_pos: ArrayView2<'_, f64>,
    model: &CovModel,
    cond_err: ArrayView1<'_, f64>,
    drift_cond: ArrayView2<'_, f64>,
    drift_target: ArrayView2<'_, f64>,
    unbiased: bool,
    exact: bool,
    local_radius: f64,
    min_neighbors: usize,
    num_threads: Option<usize>,
) -> Result<LocalKrigeResult, KrigeError> {
    check_input(
        cond_pos,
        cond_val,
        target_pos,
        cond_err,
        drift_cond,
        drift_target,
        local_radius,
    )?;
    let pnt_cnt = target_pos.ncols();
    let field_cnt = cond_val.ncols();

    // neighborhoods are searched among *all* conditioning points
    let neighbors = Neighbors::new(cond_pos, local_radius);
    // bundle all read-only data for the target tasks
    let ctx = Context {
        cond_pos,
        cond_val,
        target_pos,
        model,
        cond_err,
        drift_cond,
        drift_target,
        unbiased,
        exact,
        min_neighbors: min_neighbors.max(1),
        neighbors: &neighbors,
    };

    let mut field = Array2::from_elem((pnt_cnt, field_cnt), f64::NAN);
    let mut error = Array2::from_elem((pnt_cnt, field_cnt), f64::NAN);
    let mut n_used = Array2::<u32>::zeros((pnt_cnt, field_cnt));

    let stats = with_threads(num_threads, || {
        field
            // yields `ArrayViewMut1` for each row
            // row = one target over all fields
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            // zip the three output arrays so that each task gets one row of each
            .zip(error.axis_iter_mut(Axis(0)))
            .zip(n_used.axis_iter_mut(Axis(0)))
            // -> ((field_row, error_row), n_used_row)
            .enumerate()
            // adds the target index `j` to each task
            .map(|(j, ((field_row, error_row), n_used_row))| {
                // process the target point for all fields
                // returns stats but writes the output rows directly
                krige_target(&ctx, j, field_row, error_row, n_used_row)
            })
            .try_reduce(LocalKrigeStats::default, |a, b| Ok(a.merge(b)))
    })?;

    Ok(LocalKrigeResult {
        field,
        error,
        n_used,
        stats,
    })
}

/// Read-only data shared by all target tasks.
struct Context<'a> {
    cond_pos: ArrayView2<'a, f64>,
    cond_val: ArrayView2<'a, f64>,
    target_pos: ArrayView2<'a, f64>,
    model: &'a CovModel,
    cond_err: ArrayView1<'a, f64>,
    drift_cond: ArrayView2<'a, f64>,
    drift_target: ArrayView2<'a, f64>,
    unbiased: bool,
    exact: bool,
    min_neighbors: usize,
    neighbors: &'a Neighbors,
}

/// Cached solution for one availability mask of one target.
struct CacheEntry {
    /// conditioning point indices of the neighbors with data, same order
    /// as `weights`
    neighbors: Vec<usize>,
    /// kriging weights of these neighbors (Lagrange multipliers dropped)
    weights: Vec<f64>,
    /// `rhs . weights` of the full system (incl. Lagrange multipliers)
    error: f64,
}

impl CacheEntry {
    /// `weights . values` for one field.
    fn apply(&self, values: ArrayView1<'_, f64>) -> f64 {
        self.neighbors
            .iter()
            .zip(&self.weights)
            .map(|(&g, &w)| w * values[g])
            .sum()
    }
}

/// Process one target point for all fields, writing its output rows.
fn krige_target(
    ctx: &Context<'_>,
    j: usize,
    mut field_row: ArrayViewMut1<'_, f64>,
    mut error_row: ArrayViewMut1<'_, f64>,
    mut n_used_row: ArrayViewMut1<'_, u32>,
) -> Result<LocalKrigeStats, KrigeError> {
    let mut stats = LocalKrigeStats::default();
    let field_cnt = field_row.len();

    // --- once per target: neighbors (sorted by distance) -----------------
    let (nbrs, dists) = ctx.neighbors.within(ctx.target_pos.column(j));
    if nbrs.is_empty() {
        // no neighbors: the whole row stays NaN, n_used stays 0
        stats.n_skipped = field_cnt;
        return Ok(stats);
    }
    let k = nbrs.len();

    // --- once per target: full neighborhood system ----------------------
    let full = FullSystem::new(ctx, &nbrs, &dists);

    // --- per field: key, cache lookup, apply -----------------------------
    let words = k.div_ceil(64);
    let mut key = vec![0u64; words];
    let mut last_key = vec![0u64; words];
    let mut last_idx: Option<usize> = None;
    // entries live in a Vec, the map and `last_idx` only hold indices
    let mut entries: Vec<CacheEntry> = Vec::new();
    // map from neighbor mask to index in `entries` (per target)
    let mut index: HashMap<Vec<u64>, usize> = HashMap::new();

    // Loop over all fields, compute the availability key and either reuse weights or solve a new system.
    for t in 0..field_cnt {
        let values = ctx.cond_val.column(t);
        let count = availability_key(values, &nbrs, &mut key);
        n_used_row[t] = count as u32;
        if count < ctx.min_neighbors {
            stats.n_skipped += 1;
            continue;
        }
        // too few neighbors for the drift: solved without drift (see
        // `drift_terms_used`; done in `solve_subsystem`, counted here)
        if drift_terms_used(count, ctx.unbiased, ctx.drift_cond.nrows()) < ctx.drift_cond.nrows() {
            stats.n_without_drift += 1;
        }

        // cache lookup: first the previous field's mask (cheap, the common
        // case of consecutive fields without changes), then the hash map of
        // all masks seen for this target, else solve and store
        let idx = match last_idx {
            Some(idx) if key == last_key => {
                stats.n_hits_last += 1;
                idx
            }
            _ => match index.get(&key) {
                Some(&idx) => {
                    stats.n_hits_hash += 1;
                    idx
                }
                None => {
                    let entry = full.solve_subsystem(ctx, j, &nbrs, &key).ok_or(
                        KrigeError::SingularSystem {
                            target_index: j,
                            field_index: t,
                        },
                    )?;
                    stats.n_solves += 1;
                    entries.push(entry);
                    let idx = entries.len() - 1;
                    index.insert(key.clone(), idx);
                    idx
                }
            },
        };
        last_key.copy_from_slice(&key);
        last_idx = Some(idx);

        let entry = &entries[idx];
        field_row[t] = entry.apply(values);
        error_row[t] = entry.error;
    }
    stats.max_masks_per_target = index.len();
    Ok(stats)
}

/// Set bit `i` of `key` if neighbor `i` has a finite value in this field.
///
/// `key` has `ceil(k / 64)` words; bit `i % 64` of word `i / 64` belongs to
/// local neighbor `i` (the bit order only has to be consistent, keys are
/// never compared across targets). Returns the number of set bits.
fn availability_key(values: ArrayView1<'_, f64>, nbrs: &[usize], key: &mut [u64]) -> usize {
    key.fill(0);
    let mut count = 0;
    for (i, &g) in nbrs.iter().enumerate() {
        if values[g].is_finite() {
            key[i / 64] |= 1u64 << (i % 64);
            count += 1;
        }
    }
    count
}

/// Local indices (into the neighbor list) of the set bits of `key`.
fn selected(key: &[u64], k: usize) -> Vec<usize> {
    (0..k)
        .filter(|&i| (key[i / 64] >> (i % 64)) & 1 == 1)
        .collect()
}

/// Covariances of the full neighborhood of one target, evaluated once.
struct FullSystem {
    /// `(k, k)` covariance between all neighbors, `cond_err` on the diagonal
    cov: Array2<f64>,
    /// `(k,)` covariance between each neighbor and the target
    rhs: Vec<f64>,
}

impl FullSystem {
    fn new(ctx: &Context<'_>, nbrs: &[usize], dists: &[f64]) -> Self {
        let k = nbrs.len();
        let mut cov = Array2::<f64>::zeros((k, k));
        for a in 0..k {
            let pa = ctx.cond_pos.column(nbrs[a]);
            for b in a..k {
                let c = ctx
                    .model
                    .covariance(euclidean_dist(pa, ctx.cond_pos.column(nbrs[b])));
                cov[[a, b]] = c;
                cov[[b, a]] = c;
            }
            cov[[a, a]] += ctx.cond_err[nbrs[a]];
        }
        let rhs = dists
            .iter()
            .map(|&d| {
                if ctx.exact {
                    ctx.model.cov_nugget(d)
                } else {
                    ctx.model.covariance(d)
                }
            })
            .collect();
        Self { cov, rhs }
    }

    /// Solve the system of the neighbors marked in `key` by selecting
    /// rows/columns of the full matrices (no covariance evaluations).
    /// Layout as in the module docs: neighbors, then the unbiasedness
    /// row/column, then the drift rows/columns.
    ///
    /// **Too few neighbors for the drift** (`kt <= unbiased + drift terms`):
    /// all drift rows/columns are left out, see [`drift_terms_used`].
    fn solve_subsystem(
        &self,
        ctx: &Context<'_>,
        j: usize,
        nbrs: &[usize],
        key: &[u64],
    ) -> Option<CacheEntry> {
        let sel = selected(key, nbrs.len());
        let kt = sel.len();
        let unb = ctx.unbiased as usize;
        // !!! too few neighbors for the drift -> drop ALL drift terms !!!
        let drift_no = drift_terms_used(kt, ctx.unbiased, ctx.drift_cond.nrows());
        let p = unb + drift_no;

        // right-hand side [r; f]
        let mut rhs = vec![0.0; kt + p];
        for (a, &ia) in sel.iter().enumerate() {
            rhs[a] = self.rhs[ia];
        }
        if ctx.unbiased {
            rhs[kt] = 1.0;
        }
        for d in 0..drift_no {
            rhs[kt + unb + d] = ctx.drift_target[[d, j]];
        }
        // constraint column q of neighbor a: 1 (unbiasedness) or drift value
        let con = |a: usize, q: usize| {
            if q < unb {
                1.0
            } else {
                ctx.drift_cond[[q - unb, nbrs[sel[a]]]]
            }
        };

        // solve on the selected rows/columns (Cholesky + Schur complement,
        // dense fallback; see `crate::krige::solve`)
        let mut x = Vec::with_capacity(kt + p);
        solve_saddle_point(kt, p, |a, b| self.cov[[sel[a], sel[b]]], con, &rhs, &mut x).ok()?;
        Some(CacheEntry {
            neighbors: sel.iter().map(|&i| nbrs[i]).collect(),
            weights: x[..kt].to_vec(),
            error: rhs.iter().zip(&x).map(|(r, w)| r * w).sum(),
        })
    }
}

/// Run `op` in a pool with `num_threads` threads, or in rayon's global
/// pool if `None` (avoids building a new pool per call).
fn with_threads<R, F>(num_threads: Option<usize>, op: F) -> R
// `op` must not capture any non-`Send` data otherwise it will panic at runtime when rayon tries to move it to another thread
// R must be `Send` because it will be returned from a rayon thread
where
    R: Send,
    F: FnOnce() -> R + Send,
{
    match num_threads {
        Some(n) => rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("failed to build rayon thread pool")
            .install(op),
        None => op(),
    }
}

#[allow(clippy::too_many_arguments)]
fn check_input(
    cond_pos: ArrayView2<'_, f64>,
    cond_val: ArrayView2<'_, f64>,
    target_pos: ArrayView2<'_, f64>,
    cond_err: ArrayView1<'_, f64>,
    drift_cond: ArrayView2<'_, f64>,
    drift_target: ArrayView2<'_, f64>,
    local_radius: f64,
) -> Result<(), KrigeError> {
    let err = |msg: String| Err(KrigeError::InvalidInput(msg));
    let (dim, n, m) = (cond_pos.nrows(), cond_pos.ncols(), target_pos.ncols());
    if target_pos.nrows() != dim {
        return err(format!(
            "target_pos has {} dimensions, cond_pos has {dim}",
            target_pos.nrows()
        ));
    }
    if cond_val.nrows() != n {
        return err(format!(
            "cond_val has {} rows, expected one per conditioning point ({n})",
            cond_val.nrows()
        ));
    }
    if cond_err.len() != n {
        return err(format!(
            "cond_err has length {}, expected {n}",
            cond_err.len()
        ));
    }
    if drift_cond.ncols() != n {
        return err(format!(
            "drift_cond has {} columns, expected {n}",
            drift_cond.ncols()
        ));
    }
    if drift_target.ncols() != m || drift_target.nrows() != drift_cond.nrows() {
        return err(format!(
            "drift_target has shape {:?}, expected ({}, {m})",
            drift_target.dim(),
            drift_cond.nrows()
        ));
    }
    if !(local_radius.is_finite() && local_radius >= 0.0) {
        return err(format!(
            "local_radius must be finite and >= 0, got {local_radius}"
        ));
    }
    if !cond_pos
        .iter()
        .chain(target_pos.iter())
        .all(|v| v.is_finite())
    {
        return err("cond_pos and target_pos must be finite".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::krige::solve::{solve_dense, solve_kriging};
    use crate::krige::test_util::{self, build_lhs, build_rhs, gaussian, Lcg};
    use ndarray::{array, s, Array1};

    const FULL_RADIUS: f64 = 1e6;
    const RADIUS: f64 = 4.0;

    fn exponential() -> CovModel {
        test_util::exponential(1.5, 3.0, 0.1)
    }

    fn assert_close(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-10 * (1.0 + b.abs()), "{a} != {b}");
    }

    fn same(a: &Array2<f64>, b: &Array2<f64>) -> bool {
        a.iter()
            .zip(b.iter())
            .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
    }

    struct Data {
        cond_pos: Array2<f64>,
        cond_val: Array2<f64>,
        target_pos: Array2<f64>,
        cond_err: Array1<f64>,
        drift_cond: Array2<f64>,
        drift_target: Array2<f64>,
    }

    /// `n` conditioning points and `m` target points in [0, 10]^2, `t`
    /// fields with a share
    /// `missing` of NaN, a linear external drift.
    fn make_data(n: usize, m: usize, t: usize, missing: f64, seed: u64) -> Data {
        let mut rng = Lcg(seed);
        let cond_pos = Array2::from_shape_fn((2, n), |_| rng.uniform(0.0, 10.0));
        let target_pos = Array2::from_shape_fn((2, m), |_| rng.uniform(0.0, 10.0));
        let cond_val = Array2::from_shape_fn((n, t), |_| {
            let v = rng.uniform(-1.0, 1.0);
            if rng.uniform(0.0, 1.0) < missing {
                f64::NAN
            } else {
                v
            }
        });
        let drift = |p: &Array2<f64>| {
            Array2::from_shape_fn((1, p.ncols()), |(_, i)| p[[0, i]] + 0.5 * p[[1, i]])
        };
        Data {
            drift_cond: drift(&cond_pos),
            drift_target: drift(&target_pos),
            cond_err: Array1::from_elem(n, 0.1),
            cond_pos,
            cond_val,
            target_pos,
        }
    }

    impl Data {
        fn without_drift(mut self) -> Self {
            self.drift_cond = Array2::zeros((0, self.cond_pos.ncols()));
            self.drift_target = Array2::zeros((0, self.target_pos.ncols()));
            self
        }

        fn run(
            &self,
            unbiased: bool,
            radius: f64,
            min_neighbors: usize,
            num_threads: Option<usize>,
        ) -> LocalKrigeResult {
            calc_field_krige_local(
                self.cond_pos.view(),
                self.cond_val.view(),
                self.target_pos.view(),
                &exponential(),
                self.cond_err.view(),
                self.drift_cond.view(),
                self.drift_target.view(),
                unbiased,
                false,
                radius,
                min_neighbors,
                num_threads,
            )
            .unwrap()
        }

        /// Conditioning points with a value in field `t` within `radius` of
        /// target point `j`
        /// (brute force).
        fn available(&self, j: usize, t: usize, radius: f64) -> Vec<usize> {
            (0..self.cond_pos.ncols())
                .filter(|&i| {
                    self.cond_val[[i, t]].is_finite()
                        && euclidean_dist(self.cond_pos.column(i), self.target_pos.column(j))
                            <= radius
                })
                .collect()
        }

        /// Independent reference for target `j` and field `t`: the system
        /// assembled the straightforward way with `build_lhs` /
        /// `build_rhs` and solved with `solve_kriging` (no cache, no
        /// sub-matrix selection). `None` without neighbors.
        fn reference(&self, j: usize, t: usize, unbiased: bool, radius: f64) -> Option<(f64, f64)> {
            let avail = self.available(j, t, radius);
            if avail.is_empty() {
                return None;
            }
            let k = avail.len();
            let model = exponential();
            let drift_no = drift_terms_used(k, unbiased, self.drift_cond.nrows());
            let lhs = build_lhs(
                self.cond_pos.select(Axis(1), &avail).view(),
                &model,
                self.cond_err.select(Axis(0), &avail).view(),
                unbiased,
                self.drift_cond
                    .slice(s![..drift_no, ..])
                    .select(Axis(1), &avail)
                    .view(),
            );
            let dists: Vec<f64> = avail
                .iter()
                .map(|&i| euclidean_dist(self.cond_pos.column(i), self.target_pos.column(j)))
                .collect();
            let rhs = build_rhs(
                &dists,
                &model,
                false,
                unbiased,
                self.drift_target.slice(s![..drift_no, j]),
            );
            let w = solve_kriging(lhs.view(), rhs.view(), k).unwrap();
            let field = (0..k).map(|a| w[a] * self.cond_val[[avail[a], t]]).sum();
            Some((field, rhs.dot(&w)))
        }
    }

    #[test]
    fn drift_is_dropped_only_with_too_few_neighbors() {
        // external drift kriging: unbiased + 1 drift term, p = 2
        assert_eq!(drift_terms_used(1, true, 1), 0);
        assert_eq!(drift_terms_used(2, true, 1), 0);
        assert_eq!(drift_terms_used(3, true, 1), 1);
        // universal kriging, linear drift in 2D: p = 3
        assert_eq!(drift_terms_used(3, true, 2), 0);
        assert_eq!(drift_terms_used(4, true, 2), 2);
        // drift without unbiasedness: p = 1
        assert_eq!(drift_terms_used(1, false, 1), 0);
        assert_eq!(drift_terms_used(2, false, 1), 1);
        // no drift: nothing to drop
        assert_eq!(drift_terms_used(1, true, 0), 0);
        assert_eq!(drift_terms_used(1, false, 0), 0);
    }

    #[test]
    fn matches_reference_per_field() {
        // simple, ordinary and external drift kriging, 20 % missing values
        for (unbiased, drift) in [(false, false), (true, false), (true, true)] {
            let data = make_data(40, 25, 30, 0.2, 1);
            let data = if drift { data } else { data.without_drift() };
            let res = data.run(unbiased, RADIUS, 1, None);
            for t in 0..data.cond_val.ncols() {
                for j in 0..data.target_pos.ncols() {
                    let count = data.available(j, t, RADIUS).len() as u32;
                    assert_eq!(res.n_used[[j, t]], count);
                    match data.reference(j, t, unbiased, RADIUS) {
                        Some((field, error)) => {
                            assert_close(res.field[[j, t]], field);
                            assert_close(res.error[[j, t]], error);
                        }
                        None => assert!(res.field[[j, t]].is_nan()),
                    }
                }
            }
        }
    }

    #[test]
    fn single_field_equals_its_column_of_a_multi_field_call() {
        // a single field is just T = 1: identical to the bit
        let data = make_data(40, 25, 6, 0.2, 10);
        let all = data.run(true, RADIUS, 1, None);
        for t in 0..6 {
            let one = calc_field_krige_local(
                data.cond_pos.view(),
                data.cond_val.slice(s![.., t..t + 1]),
                data.target_pos.view(),
                &exponential(),
                data.cond_err.view(),
                data.drift_cond.view(),
                data.drift_target.view(),
                true,
                false,
                RADIUS,
                1,
                None,
            )
            .unwrap();
            let column = |a: &Array2<f64>| a.slice(s![.., t..t + 1]).to_owned();
            assert!(same(&one.field, &column(&all.field)));
            assert!(same(&one.error, &column(&all.error)));
            assert_eq!(one.n_used, all.n_used.slice(s![.., t..t + 1]));
        }
    }

    #[test]
    fn too_few_neighbors_for_the_drift_drops_all_drift_terms() {
        // small radius: many target points with only 1 or 2 neighbors
        const SMALL: f64 = 1.5;
        let data = make_data(40, 60, 30, 0.2, 5);
        let no_drift = make_data(40, 60, 30, 0.2, 5).without_drift();
        // p = unbiased + 1 drift term
        for unbiased in [true, false] {
            let p = unbiased as u32 + 1;
            let res = data.run(unbiased, SMALL, 1, None);
            let reference = no_drift.run(unbiased, SMALL, 1, None);
            let mut dropped = 0;
            for ((&n, (&f, &e)), (&f_ref, &e_ref)) in res
                .n_used
                .iter()
                .zip(res.field.iter().zip(&res.error))
                .zip(reference.field.iter().zip(&reference.error))
            {
                if n >= 1 && n <= p {
                    // without drift: identical to ordinary / simple kriging
                    assert_close(f, f_ref);
                    assert_close(e, e_ref);
                    dropped += 1;
                }
            }
            assert!(dropped > 0, "test data has no small neighborhoods");
            assert_eq!(res.stats.n_without_drift, dropped);
            assert_eq!(reference.stats.n_without_drift, 0);
        }
    }

    #[test]
    fn one_neighbor_with_drift_gives_its_value() {
        // external drift kriging with a single neighbor: the result is its
        // value (ordinary kriging, weight 1), not an error
        let data = Data {
            cond_pos: Array2::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap(),
            cond_val: Array2::from_shape_vec((1, 2), vec![3.5, -2.0]).unwrap(),
            target_pos: Array2::from_shape_vec((2, 1), vec![1.5, 1.0]).unwrap(),
            cond_err: Array1::from_elem(1, 0.0),
            drift_cond: Array2::from_shape_vec((1, 1), vec![200.0]).unwrap(),
            drift_target: Array2::from_shape_vec((1, 1), vec![450.0]).unwrap(),
        };
        let res = data.run(true, FULL_RADIUS, 1, None);
        assert_close(res.field[[0, 0]], 3.5);
        assert_close(res.field[[0, 1]], -2.0);
        assert_eq!(res.stats.n_without_drift, 2);
    }

    #[test]
    fn one_solve_per_target_without_missing_data() {
        let data = make_data(40, 25, 30, 0.0, 3);
        let stats = data.run(true, FULL_RADIUS, 1, None).stats;
        assert_eq!(stats.n_solves, 25);
        assert_eq!(stats.n_hits_last, 29 * 25);
    }

    #[test]
    fn cache_reuses_recurring_masks() {
        // outages alternate A, B, A, B, ...: the previous field's mask never
        // matches, but the hash map finds the mask of two fields before
        let mut data = make_data(40, 25, 30, 0.0, 4);
        for t in 0..30 {
            data.cond_val[[t % 2, t]] = f64::NAN;
        }
        let stats = data.run(true, FULL_RADIUS, 1, None).stats;
        assert_eq!(stats.n_solves, 2 * 25);
        assert_eq!(stats.n_hits_hash, 28 * 25);
        assert_eq!(stats.n_hits_last, 0);
        assert_eq!(stats.max_masks_per_target, 2);
    }

    #[test]
    fn keys_distinguish_neighbors_beyond_the_first_64_bits() {
        // 100 conditioning points on a line, target point at the start:
        // local neighbor index == conditioning point index, so point 70
        // lives in the second key word
        let n = 100;
        let mut data = make_data(n, 1, 3, 0.0, 5);
        for i in 0..n {
            data.cond_pos[[0, i]] = 1.0 + i as f64;
            data.cond_pos[[1, i]] = 0.0;
        }
        data.target_pos = array![[0.0], [0.0]];
        data.cond_val[[70, 1]] = f64::NAN; // fields 0 and 2 equal, 1 differs
        let data = data.without_drift();
        let res = data.run(true, FULL_RADIUS, 1, None);
        assert_eq!(res.stats.n_solves, 2);
        assert_eq!(res.stats.n_hits_hash, 1);
        let (field, _) = data.reference(0, 1, true, FULL_RADIUS).unwrap();
        assert_close(res.field[[0, 1]], field);
    }

    #[test]
    fn too_few_neighbors_give_nan() {
        let mut data = make_data(40, 25, 5, 0.0, 6);
        data.cond_val.slice_mut(s![2.., 0]).fill(f64::NAN); // only 2 points with a value in field 0
        let res = data.run(true, FULL_RADIUS, 3, None);
        assert!(res.n_used.column(0).iter().all(|&c| c == 2));
        assert!(res.field.column(0).iter().all(|v| v.is_nan()));
        assert!(res.error.column(0).iter().all(|v| v.is_nan()));
        assert!(res.field.slice(s![.., 1..]).iter().all(|v| v.is_finite()));
        assert_eq!(res.stats.n_skipped, 25);
    }

    #[test]
    fn target_without_neighbors_is_nan_not_an_error() {
        let mut data = make_data(10, 1, 4, 0.0, 7);
        data.target_pos = array![[1000.0], [1000.0]];
        let res = data.run(true, 1.0, 1, None);
        assert!(res.field.iter().all(|v| v.is_nan()));
        assert!(res.error.iter().all(|v| v.is_nan()));
        assert!(res.n_used.iter().all(|&c| c == 0));
        assert_eq!(res.stats.n_skipped, 4);
    }

    #[test]
    fn thread_count_does_not_change_result() {
        let data = make_data(40, 25, 30, 0.2, 8);
        let one = data.run(true, RADIUS, 1, Some(1));
        let three = data.run(true, RADIUS, 1, Some(3));
        assert!(same(&one.field, &three.field));
        assert_eq!(one.stats, three.stats);
    }

    #[test]
    fn invalid_input_is_an_error_not_a_panic() {
        let data = make_data(10, 3, 4, 0.0, 9);
        let wrong_rows = data.cond_val.slice(s![..5, ..]);
        let res = calc_field_krige_local(
            data.cond_pos.view(),
            wrong_rows,
            data.target_pos.view(),
            &exponential(),
            data.cond_err.view(),
            data.drift_cond.view(),
            data.drift_target.view(),
            true,
            false,
            RADIUS,
            1,
            None,
        );
        assert!(matches!(res, Err(KrigeError::InvalidInput(_))));
    }

    // --- one field, small hand-made data -----------------------------------

    fn scattered_cond() -> (Array2<f64>, Array2<f64>) {
        let cond_pos = array![[0.0, 4.0, 8.0, 2.0, 6.0], [0.0, 3.0, 1.0, 7.0, 5.0]];
        let cond_val = array![[0.3], [-0.5], [1.2], [0.1], [-0.8]]; // (n, 1)
        (cond_pos, cond_val)
    }

    /// One field without drift, ordinary kriging.
    fn krige_one(
        model: &CovModel,
        cond_pos: &Array2<f64>,
        cond_val: &Array2<f64>,
        target_pos: &Array2<f64>,
        cond_err: f64,
        exact: bool,
        radius: f64,
    ) -> LocalKrigeResult {
        let n = cond_pos.ncols();
        calc_field_krige_local(
            cond_pos.view(),
            cond_val.view(),
            target_pos.view(),
            model,
            Array1::from_elem(n, cond_err).view(),
            Array2::zeros((0, n)).view(),
            Array2::zeros((0, target_pos.ncols())).view(),
            true,
            exact,
            radius,
            1,
            None,
        )
        .unwrap()
    }

    #[test]
    fn exact_mode_reproduces_conditioning_value_at_conditioning_point() {
        let model = gaussian(2.0, 3.0, 0.0);
        let (cond_pos, cond_val) = scattered_cond();
        // target == the first conditioning point exactly
        let target_pos = cond_pos.column(0).to_owned().insert_axis(Axis(1));
        let res = krige_one(&model, &cond_pos, &cond_val, &target_pos, 0.0, true, 100.0);
        assert!((res.field[[0, 0]] - cond_val[[0, 0]]).abs() < 1e-6);
    }

    #[test]
    fn matches_manual_lhs_rhs_solve_when_radius_covers_everyone() {
        let model = gaussian(1.5, 3.0, 0.1);
        let (cond_pos, cond_val) = scattered_cond();
        let cond_no = cond_pos.ncols();
        let target_pos = array![[3.0], [3.0]];
        let res = krige_one(&model, &cond_pos, &cond_val, &target_pos, 0.1, false, 1e6);

        // Manually assemble the very same (full-neighborhood) system with
        // build_lhs/build_rhs/solve_dense directly, bypassing
        // the k-d tree entirely, and check the two paths agree.
        let target = target_pos.column(0);
        let dists: Vec<f64> = (0..cond_no)
            .map(|i| euclidean_dist(cond_pos.column(i), target))
            .collect();
        let lhs = build_lhs(
            cond_pos.view(),
            &model,
            Array1::from_elem(cond_no, 0.1).view(),
            true,
            Array2::zeros((0, cond_no)).view(),
        );
        let rhs = build_rhs(&dists, &model, false, true, Array1::zeros(0).view());
        let weights = solve_dense(lhs.view(), rhs.view()).unwrap();
        let expected_field: f64 = (0..cond_no).map(|i| weights[i] * cond_val[[i, 0]]).sum();
        let expected_error = rhs.dot(&weights);

        assert!((res.field[[0, 0]] - expected_field).abs() < 1e-8);
        assert!((res.error[[0, 0]] - expected_error).abs() < 1e-8);
    }

    // `cond_pos`/`target_pos` being `ArrayView2` only fixes the array's
    // *rank* (a matrix: axis 0 = spatial dimension, axis 1 = point index).
    // The actual spatial dimensionality is `cond_pos.nrows()`, read at
    // runtime -- so the exact same function works unchanged for 1D and 3D
    // positions, not just the 2D data used in every other test here.

    #[test]
    fn works_with_1d_positions() {
        let model = gaussian(1.0, 3.0, 0.1);
        let cond_pos = array![[0.0, 2.0, 5.0, 8.0]]; // shape (1, 4)
        let cond_val = array![[0.1], [0.4], [-0.2], [0.9]];
        let target_pos = array![[1.0, 6.0]]; // shape (1, 2)
        let res = krige_one(&model, &cond_pos, &cond_val, &target_pos, 0.1, false, 100.0);
        assert_eq!(res.field.dim(), (2, 1));
        assert!(res.field.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn works_with_3d_positions() {
        let model = gaussian(1.0, 3.0, 0.1);
        let cond_pos = array![
            [0.0, 2.0, 5.0, 8.0],
            [1.0, 0.0, 3.0, 2.0],
            [4.0, 1.0, 0.0, 5.0],
        ]; // shape (3, 4)
        let cond_val = array![[0.1], [0.4], [-0.2], [0.9]];
        let target_pos = array![[1.0, 6.0], [0.5, 1.0], [2.0, 3.0]]; // shape (3, 2)
        let res = krige_one(&model, &cond_pos, &cond_val, &target_pos, 0.1, false, 100.0);
        assert_eq!(res.field.dim(), (2, 1));
        assert!(res.field.iter().all(|v| v.is_finite()));
    }
}
