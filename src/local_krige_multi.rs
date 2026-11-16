//! Local kriging of many fields at once, reusing the kriging weights.
//!
//! Rust port of `calc_field_krige_local_multi_python` from GSTools'
//! `dev/local_kriging_multi_numpy.py`.
//!
//! Typical use: daily station data (one field per day) interpolated onto
//! a fixed target grid. For one target point the kriging weights depend
//! on positions, covariance model, drift and *which neighbors have data*
//! -- never on the values. So the loop order of the single-field
//! [`calc_field_krige_local`](crate::local_krige::calc_field_krige_local)
//! is swapped:
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

use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use kdtree::{distance::squared_euclidean, KdTree};
use ndarray::{Array2, ArrayView1, ArrayView2, ArrayViewMut1, Axis};
use rayon::prelude::*;

use crate::covmodel::CovModel;
use crate::kriging_solve::{solve_cholesky_schur, with_workspace};
use crate::local_krige::{euclidean_dist, solve_dense};

/// How kriging weights are reused between the fields of one target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheMode {
    /// Solve every system again (reference / benchmark).
    None,
    /// Reuse the weights only if the mask equals the previous field's
    /// mask (what EDK does).
    Last,
    /// Reuse the weights of any mask seen before for this target.
    Hash,
}

impl FromStr for CacheMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" => Ok(Self::None),
            "last" => Ok(Self::Last),
            "hash" => Ok(Self::Hash),
            other => Err(format!(
                "unknown cache mode '{other}', expected 'none', 'last' or 'hash'"
            )),
        }
    }
}

/// Counters describing how much work the cache saved.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MultiStats {
    /// Number of kriging systems actually solved.
    pub n_solves: usize,
    /// Fields that reused the previous field's weights.
    pub n_hits_last: usize,
    /// Fields that reused weights from the per-target hash map.
    pub n_hits_hash: usize,
    /// (target, field) pairs left as NaN (fewer than `min_neighbors`).
    pub n_skipped: usize,
    /// Largest number of distinct masks cached for one target.
    pub max_masks_per_target: usize,
}

impl MultiStats {
    fn merge(self, other: Self) -> Self {
        Self {
            n_solves: self.n_solves + other.n_solves,
            n_hits_last: self.n_hits_last + other.n_hits_last,
            n_hits_hash: self.n_hits_hash + other.n_hits_hash,
            n_skipped: self.n_skipped + other.n_skipped,
            max_masks_per_target: self.max_masks_per_target.max(other.max_masks_per_target),
        }
    }
}

/// Errors of [`calc_field_krige_local_multi`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalKrigeMultiError {
    /// Inconsistent array shapes or non-finite positions / radius.
    InvalidInput(String),
    /// A local system could not be solved, neither by the Cholesky solver
    /// nor by the LU / least-squares fallback of [`solve_dense`].
    SingularSystem {
        /// index of the target point
        target_index: usize,
        /// index of the field (e.g. day)
        field_index: usize,
    },
}

impl fmt::Display for LocalKrigeMultiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::SingularSystem {
                target_index,
                field_index,
            } => write!(
                f,
                "local kriging matrix for target point {target_index} and field \
                 {field_index} is singular even for the least-squares fallback"
            ),
        }
    }
}

impl std::error::Error for LocalKrigeMultiError {}

/// Output of [`calc_field_krige_local_multi`]. All arrays have the shape
/// `(m, T)`: one row per target point, one column per field.
#[derive(Debug, Clone)]
pub struct LocalKrigeMultiResult {
    /// Raw kriging estimate, NaN where fewer than `min_neighbors` were available.
    pub field: Array2<f64>,
    /// Raw `rhs . weights` (not yet `sill - error`), NaN like `field`.
    pub error: Array2<f64>,
    /// Number of available neighbors per target and field.
    pub n_used: Array2<u32>,
    /// Cache statistics.
    pub stats: MultiStats,
}

/// Evaluate local kriging for many fields sharing the same conditioning
/// positions, reusing the kriging weights per target point.
///
/// # Arguments
/// * `cond_pos` - isometrized conditioning positions, shape `(dim, n)`
/// * `cond_val` - centered conditioning values, shape `(n, T)`;
///   non-finite values (NaN) mark a missing value for that field only
/// * `target_pos` - isometrized target positions, shape `(dim, m)`
/// * `model` - covariance model
/// * `cond_err` - measurement error per conditioning point, shape `(n,)`
/// * `drift_cond` - drift terms at the conditioning points, shape `(p, n)`
/// * `drift_target` - drift terms at the target points, shape `(p, m)`
/// * `unbiased` - whether to add the unbiasedness constraint
/// * `exact` - whether to use `model.cov_nugget` on the RHS
/// * `local_radius` - search radius (isometrized distance)
/// * `min_neighbors` - fields with fewer available neighbors are left NaN
///   (values below 1 are treated as 1)
/// * `cache` - see [`CacheMode`]
/// * `num_threads` - number of threads, `None` for rayon's global pool
///
/// # Returns
/// A [`LocalKrigeMultiResult`]. Unlike the single-field function, a target
/// without neighbors is not an error: its row stays NaN.
#[allow(clippy::too_many_arguments)]
pub fn calc_field_krige_local_multi(
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
    cache: CacheMode,
    num_threads: Option<usize>,
) -> Result<LocalKrigeMultiResult, LocalKrigeMultiError> {
    check_input(
        cond_pos,
        cond_val,
        target_pos,
        cond_err,
        drift_cond,
        drift_target,
        local_radius,
    )?;
    let dim = cond_pos.nrows();
    let pnt_cnt = target_pos.ncols();
    let field_cnt = cond_val.ncols();

    // neighborhoods are searched among *all* conditioning points
    let mut tree: KdTree<f64, usize, Vec<f64>> = KdTree::new(dim);
    for i in 0..cond_pos.ncols() {
        tree.add(cond_pos.column(i).to_vec(), i)
            .expect("cond_pos columns are finite and match `dim`");
    }
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
        radius_sq: local_radius * local_radius,
        min_neighbors: min_neighbors.max(1),
        cache,
        tree: &tree,
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
            .try_reduce(MultiStats::default, |a, b| Ok(a.merge(b)))
    })?;

    Ok(LocalKrigeMultiResult {
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
    radius_sq: f64,
    min_neighbors: usize,
    cache: CacheMode,
    tree: &'a KdTree<f64, usize, Vec<f64>>,
}

/// Cached solution for one availability mask of one target.
struct CacheEntry {
    /// global indices of the available neighbors, same order as `weights`
    stations: Vec<usize>,
    /// kriging weights of these stations (Lagrange multipliers dropped)
    weights: Vec<f64>,
    /// `rhs . weights` of the full system (incl. Lagrange multipliers)
    error: f64,
}

impl CacheEntry {
    /// `weights . values` for one field.
    fn apply(&self, values: ArrayView1<'_, f64>) -> f64 {
        self.stations
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
) -> Result<MultiStats, LocalKrigeMultiError> {
    let mut stats = MultiStats::default();
    let field_cnt = field_row.len();

    // --- once per target: neighbors (sorted by distance) -----------------
    let target = ctx.target_pos.column(j).to_vec();
    let found = ctx
        .tree
        .within(&target, ctx.radius_sq, &squared_euclidean)
        .expect("target point is finite and matches `dim`");
    if found.is_empty() {
        stats.n_skipped = field_cnt;
        return Ok(stats);
    }
    let nbrs: Vec<usize> = found.iter().map(|&(_, &idx)| idx).collect();
    let dists: Vec<f64> = found.iter().map(|&(sq, _)| sq.sqrt()).collect();
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
    let mut index: HashMap<Vec<u64>, usize> = HashMap::new();

    for t in 0..field_cnt {
        let values = ctx.cond_val.column(t);
        let count = availability_key(values, &nbrs, &mut key);
        n_used_row[t] = count as u32;
        if count < ctx.min_neighbors {
            stats.n_skipped += 1;
            continue;
        }

        let cached = if ctx.cache != CacheMode::None && last_idx.is_some() && key == last_key {
            stats.n_hits_last += 1;
            last_idx
        } else if ctx.cache == CacheMode::Hash {
            let hit = index.get(&key).copied();
            if hit.is_some() {
                stats.n_hits_hash += 1;
            }
            hit
        } else {
            None
        };

        let idx = match cached {
            Some(idx) => idx,
            None => {
                let entry = full.solve_subsystem(ctx, j, &nbrs, &key).ok_or(
                    LocalKrigeMultiError::SingularSystem {
                        target_index: j,
                        field_index: t,
                    },
                )?;
                stats.n_solves += 1;
                if ctx.cache != CacheMode::Hash {
                    entries.clear(); // only the latest entry is ever used
                }
                entries.push(entry);
                let idx = entries.len() - 1;
                if ctx.cache == CacheMode::Hash {
                    index.insert(key.clone(), idx);
                }
                idx
            }
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
    /// Layout as in `build_local_lhs`/`build_local_rhs`: neighbors, then
    /// the unbiasedness row/column, then the drift rows/columns.
    fn solve_subsystem(
        &self,
        ctx: &Context<'_>,
        j: usize,
        nbrs: &[usize],
        key: &[u64],
    ) -> Option<CacheEntry> {
        let sel = selected(key, nbrs.len());
        let kt = sel.len();
        let drift_no = ctx.drift_cond.nrows();
        let unb = ctx.unbiased as usize;
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

        // Cholesky + Schur complement on the selected rows/columns
        let mut x = Vec::with_capacity(kt + p);
        let ok = with_workspace(|ws| {
            solve_cholesky_schur(
                ws,
                kt,
                p,
                |a, b| self.cov[[sel[a], sel[b]]],
                con,
                &rhs,
                &mut x,
            )
        });
        if !ok {
            // fallback: full system with LU, then SVD least squares
            let size = kt + p;
            let mut lhs = Array2::<f64>::zeros((size, size));
            for a in 0..kt {
                for b in 0..kt {
                    lhs[[a, b]] = self.cov[[sel[a], sel[b]]];
                }
                for q in 0..p {
                    lhs[[a, kt + q]] = con(a, q);
                    lhs[[kt + q, a]] = con(a, q);
                }
            }
            x = solve_dense(lhs.view(), ArrayView1::from(&rhs))
                .ok()?
                .to_vec();
        }
        Some(CacheEntry {
            stations: sel.iter().map(|&i| nbrs[i]).collect(),
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
) -> Result<(), LocalKrigeMultiError> {
    let err = |msg: String| Err(LocalKrigeMultiError::InvalidInput(msg));
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
    use crate::covmodel_spec::CovModelSpec;
    use crate::local_krige::calc_field_krige_local;
    use ndarray::Array1;
    use ndarray::{array, s};

    const FULL_RADIUS: f64 = 1e6;
    const RADIUS: f64 = 4.0;

    fn exponential() -> CovModel {
        CovModel::build(CovModelSpec::Exponential {
            var: 1.5,
            len_scale: 3.0,
            nugget: 0.1,
        })
        .unwrap()
    }

    /// Small deterministic random number generator (no extra dependency).
    struct Lcg(u64);

    impl Lcg {
        fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            lo + (hi - lo) * ((self.0 >> 11) as f64 / (1u64 << 53) as f64)
        }
    }

    struct Data {
        cond_pos: Array2<f64>,
        cond_val: Array2<f64>,
        target_pos: Array2<f64>,
        cond_err: Array1<f64>,
        drift_cond: Array2<f64>,
        drift_target: Array2<f64>,
    }

    /// `n` stations and `m` targets in [0, 10]^2, `t` fields with a share
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
            cache: CacheMode,
            num_threads: Option<usize>,
        ) -> LocalKrigeMultiResult {
            calc_field_krige_local_multi(
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
                cache,
                num_threads,
            )
            .unwrap()
        }

        /// Single-field reference: only the stations available in field `t`
        /// and only the given targets.
        fn single(
            &self,
            t: usize,
            targets: &[usize],
            unbiased: bool,
            radius: f64,
        ) -> (Array1<f64>, Array1<f64>) {
            let avail: Vec<usize> = (0..self.cond_pos.ncols())
                .filter(|&i| self.cond_val[[i, t]].is_finite())
                .collect();
            calc_field_krige_local(
                self.cond_pos.select(Axis(1), &avail).view(),
                self.cond_val.column(t).select(Axis(0), &avail).view(),
                self.target_pos.select(Axis(1), targets).view(),
                &exponential(),
                self.cond_err.select(Axis(0), &avail).view(),
                self.drift_cond.select(Axis(1), &avail).view(),
                self.drift_target.select(Axis(1), targets).view(),
                unbiased,
                false,
                radius,
                None,
            )
            .unwrap()
        }

        /// Brute-force count of available stations within `radius`.
        fn count(&self, j: usize, t: usize, radius: f64) -> u32 {
            (0..self.cond_pos.ncols())
                .filter(|&i| {
                    self.cond_val[[i, t]].is_finite()
                        && euclidean_dist(self.cond_pos.column(i), self.target_pos.column(j))
                            <= radius
                })
                .count() as u32
        }
    }

    fn assert_close(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-10 * (1.0 + b.abs()), "{a} != {b}");
    }

    fn same(a: &Array2<f64>, b: &Array2<f64>) -> bool {
        a.iter()
            .zip(b.iter())
            .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
    }

    #[test]
    fn matches_single_field_function_per_field() {
        // simple, ordinary and external drift kriging
        for (unbiased, drift) in [(false, false), (true, false), (true, true)] {
            let data = make_data(40, 25, 30, 0.2, 1);
            let data = if drift { data } else { data.without_drift() };
            let res = data.run(unbiased, RADIUS, 1, CacheMode::Hash, None);
            for t in 0..data.cond_val.ncols() {
                let mut targets = Vec::new();
                for j in 0..data.target_pos.ncols() {
                    let count = data.count(j, t, RADIUS);
                    assert_eq!(res.n_used[[j, t]], count);
                    if count > 0 {
                        targets.push(j);
                    } else {
                        assert!(res.field[[j, t]].is_nan());
                    }
                }
                let (field, error) = data.single(t, &targets, unbiased, RADIUS);
                for (a, &j) in targets.iter().enumerate() {
                    assert_close(res.field[[j, t]], field[a]);
                    assert_close(res.error[[j, t]], error[a]);
                }
            }
        }
    }

    #[test]
    fn cache_modes_give_identical_results() {
        let data = make_data(40, 25, 30, 0.2, 2);
        let none = data.run(true, RADIUS, 1, CacheMode::None, None);
        let last = data.run(true, RADIUS, 1, CacheMode::Last, None);
        let hash = data.run(true, RADIUS, 1, CacheMode::Hash, None);
        for res in [&last, &hash] {
            assert!(same(&res.field, &none.field));
            assert!(same(&res.error, &none.error));
            assert_eq!(res.n_used, none.n_used);
        }
        assert!(none.stats.n_solves >= last.stats.n_solves);
        assert!(last.stats.n_solves >= hash.stats.n_solves);
        assert_eq!(none.stats.n_solves + none.stats.n_skipped, 25 * 30);
    }

    #[test]
    fn one_solve_per_target_without_missing_data() {
        let data = make_data(40, 25, 30, 0.0, 3);
        let solves = |cache| data.run(true, FULL_RADIUS, 1, cache, None).stats.n_solves;
        assert_eq!(solves(CacheMode::Hash), 25);
        assert_eq!(solves(CacheMode::Last), 25);
        assert_eq!(solves(CacheMode::None), 25 * 30);
    }

    #[test]
    fn hash_cache_reuses_recurring_masks() {
        // outages alternate A, B, A, B, ...: "last" misses every field
        let mut data = make_data(40, 25, 30, 0.0, 4);
        for t in 0..30 {
            data.cond_val[[t % 2, t]] = f64::NAN;
        }
        let hash = data.run(true, FULL_RADIUS, 1, CacheMode::Hash, None).stats;
        let last = data.run(true, FULL_RADIUS, 1, CacheMode::Last, None).stats;
        assert_eq!(hash.n_solves, 2 * 25);
        assert_eq!(hash.n_hits_hash, 28 * 25);
        assert_eq!(hash.max_masks_per_target, 2);
        assert_eq!(last.n_solves, 30 * 25);
    }

    #[test]
    fn keys_distinguish_neighbors_beyond_the_first_64_bits() {
        // 100 stations on a line, target at the start: local neighbor index
        // == station index, so station 70 lives in the second key word
        let n = 100;
        let mut data = make_data(n, 1, 3, 0.0, 5);
        for i in 0..n {
            data.cond_pos[[0, i]] = 1.0 + i as f64;
            data.cond_pos[[1, i]] = 0.0;
        }
        data.target_pos = array![[0.0], [0.0]];
        data.cond_val[[70, 1]] = f64::NAN; // fields 0 and 2 equal, 1 differs
        let data = data.without_drift();
        let res = data.run(true, FULL_RADIUS, 1, CacheMode::Hash, None);
        assert_eq!(res.stats.n_solves, 2);
        assert_eq!(res.stats.n_hits_hash, 1);
        let (field, _) = data.single(1, &[0], true, FULL_RADIUS);
        assert_close(res.field[[0, 1]], field[0]);
    }

    #[test]
    fn too_few_neighbors_give_nan() {
        let mut data = make_data(40, 25, 5, 0.0, 6);
        data.cond_val.slice_mut(s![2.., 0]).fill(f64::NAN); // only 2 stations in field 0
        let res = data.run(true, FULL_RADIUS, 3, CacheMode::Hash, None);
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
        let res = data.run(true, 1.0, 1, CacheMode::Hash, None);
        assert!(res.field.iter().all(|v| v.is_nan()));
        assert!(res.n_used.iter().all(|&c| c == 0));
        assert_eq!(res.stats.n_skipped, 4);
    }

    #[test]
    fn thread_count_does_not_change_result() {
        let data = make_data(40, 25, 30, 0.2, 8);
        let one = data.run(true, RADIUS, 1, CacheMode::Hash, Some(1));
        let three = data.run(true, RADIUS, 1, CacheMode::Hash, Some(3));
        assert!(same(&one.field, &three.field));
        assert_eq!(one.stats, three.stats);
    }

    #[test]
    fn invalid_input_is_an_error_not_a_panic() {
        let data = make_data(10, 3, 4, 0.0, 9);
        let wrong_rows = data.cond_val.slice(s![..5, ..]);
        let res = calc_field_krige_local_multi(
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
            CacheMode::Hash,
            None,
        );
        assert!(matches!(res, Err(LocalKrigeMultiError::InvalidInput(_))));
    }

    #[test]
    fn cache_mode_parses_from_str() {
        assert_eq!("hash".parse::<CacheMode>(), Ok(CacheMode::Hash));
        assert_eq!("last".parse::<CacheMode>(), Ok(CacheMode::Last));
        assert_eq!("none".parse::<CacheMode>(), Ok(CacheMode::None));
        assert!("lru".parse::<CacheMode>().is_err());
    }
}
