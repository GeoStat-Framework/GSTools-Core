//! Local (moving-neighborhood) kriging.
//!
//! Rust port of `calc_field_krige_local` from GSTools'
//! `dev/local_kriging/local_kriging_numpy.py`. That Python function is
//! itself already written as a stand-in for the planned
//! `gstools_core::krige::calc_field_krige_local` -- see its docstring: it
//! only ever receives plain arrays and scalars, no Python callables, no
//! `Krige`/`CovModel` objects with behaviour beyond `covariance`/
//! `cov_nugget`. Anisotropy/rotation stay on the Python side: positions
//! here are assumed already isometrized (isotropic), so every distance is
//! a plain Euclidean norm.
//!
//! The per-target-point loop in [`calc_field_krige_local`] is deliberately
//! *not* parallelized yet: it is written as a `.map()` over an iterator of
//! independent, side-effect-free target-point computations (each only
//! borrows shared, immutable data), so switching to
//! `rayon`'s `.into_par_iter()` later -- as `krige.rs` does for the global
//! case -- is a one-line change, not a restructuring.
//!
//! Unlike global kriging (see `krige.rs`), local kriging needs an actual
//! dense solve per target point (each neighborhood is a different small
//! system) -- there is no pre-inverted matrix to just contract against.
//! [`solve_dense`] uses `nalgebra`'s partial-pivoted `LU` for the common
//! (well-conditioned) case, falling back to an `SVD`-based least-squares
//! solve -- analogous to NumPy's `lstsq` in the Python prototype -- for
//! singular systems.

use std::fmt;

use kdtree::{distance::squared_euclidean, KdTree};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;

use crate::covmodel::CovModel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalKrigeError {
    /// No conditioning point fell within `local_radius` of this target
    /// point (mirrors the `ValueError` raised in the Python prototype).
    NoNeighbors { target_index: usize },
    /// The local kriging matrix was singular even for [`solve_dense`]'s
    /// least-squares fallback -- in practice essentially unreachable, see
    /// its docs.
    SingularSystem { target_index: usize },
}

impl fmt::Display for LocalKrigeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoNeighbors { target_index } => write!(
                f,
                "no conditioning points within local_radius for target point {target_index}; \
                 increase local_radius"
            ),
            Self::SingularSystem { target_index } => write!(
                f,
                "local kriging matrix for target point {target_index} is singular even for the \
                 least-squares fallback"
            ),
        }
    }
}

impl std::error::Error for LocalKrigeError {}

/// Assemble the local kriging matrix (LHS) for one target point's
/// neighborhood: pairwise covariances between the `k` neighbors, plus
/// measurement error on the diagonal, plus (optionally) the
/// unbiasedness row/column and drift rows/columns. Mirrors the `lhs`
/// block in `local_kriging_numpy.calc_field_krige_local`.
///
/// # Arguments
/// * `nbr_pos` - neighbor positions, shape `(dim, k)`
/// * `model` - covariance model
/// * `cond_err` - measurement error per neighbor, shape `(k,)`
/// * `unbiased` - whether to add the Ordinary-Kriging Lagrange row/column
/// * `drift_nbrs` - drift terms evaluated at the neighbor positions,
///   shape `(p, k)` (`p` may be 0)
///
/// # Returns
/// the `(k + p + unbiased) x (k + p + unbiased)` LHS matrix
pub fn build_local_lhs(
    nbr_pos: ArrayView2<'_, f64>,
    model: &CovModel,
    cond_err: ArrayView1<'_, f64>,
    unbiased: bool,
    drift_nbrs: ArrayView2<'_, f64>,
) -> Array2<f64> {
    let k = nbr_pos.ncols();
    let drift_no = drift_nbrs.nrows();
    let pad = drift_no + unbiased as usize;
    let size = k + pad;

    let mut lhs = Array2::<f64>::zeros((size, size));
    for i in 0..k {
        let pi = nbr_pos.column(i);
        for j in 0..k {
            let pj = nbr_pos.column(j);
            let dist = euclidean_dist(pi, pj);
            lhs[[i, j]] = model.covariance(dist);
        }
        lhs[[i, i]] += cond_err[i];
    }
    if unbiased {
        let u = k;
        for i in 0..k {
            lhs[[u, i]] = 1.0;
            lhs[[i, u]] = 1.0;
        }
    }
    if drift_no > 0 {
        let offset = k + unbiased as usize;
        for d in 0..drift_no {
            for i in 0..k {
                lhs[[offset + d, i]] = drift_nbrs[[d, i]];
                lhs[[i, offset + d]] = drift_nbrs[[d, i]];
            }
        }
    }
    lhs
}

/// Assemble the local kriging vector (RHS) for one target point: the
/// covariance (or, in `exact` mode, the nugget-aware covariance) between
/// each neighbor and the target point, plus the unbiasedness/drift
/// entries matching [`build_local_lhs`]'s padding. Mirrors the `rhs`
/// block in `local_kriging_numpy.calc_field_krige_local`.
///
/// # Arguments
/// * `nbr_dists` - distance from each neighbor to the target point,
///   shape `(k,)`
/// * `model` - covariance model
/// * `exact` - use `model.cov_nugget` instead of `model.covariance`, so a
///   target point that coincides with a neighbor reproduces its value
///   exactly
/// * `unbiased` - must match [`build_local_lhs`]
/// * `drift_target` - drift terms evaluated at the target position,
///   shape `(p,)`, `p` must match [`build_local_lhs`]'s `drift_nbrs`
///
/// # Returns
/// the RHS vector, same length as [`build_local_lhs`]'s matrix dimension
pub fn build_local_rhs(
    nbr_dists: &[f64],
    model: &CovModel,
    exact: bool,
    unbiased: bool,
    drift_target: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let k = nbr_dists.len();
    let drift_no = drift_target.len();
    let pad = drift_no + unbiased as usize;

    let mut rhs = Array1::<f64>::zeros(k + pad);
    for (i, &dist) in nbr_dists.iter().enumerate() {
        rhs[i] = if exact {
            model.cov_nugget(dist)
        } else {
            model.covariance(dist)
        };
    }
    if unbiased {
        rhs[k] = 1.0;
    }
    if drift_no > 0 {
        let offset = k + unbiased as usize;
        for d in 0..drift_no {
            rhs[offset + d] = drift_target[d];
        }
    }
    rhs
}

/// Tolerance below which an SVD singular value is treated as zero in the
/// least-squares fallback -- analogous to NumPy `lstsq`'s default `rcond`
/// cutoff.
const LSTSQ_EPS: f64 = 1e-12;

/// Solve the dense linear system `a @ x = b`. Pure function (takes views,
/// returns a fresh `Array1`), so it is independently testable with plain
/// literal matrices, unrelated to kriging.
///
/// Local kriging systems are small (a handful to a few hundred neighbors),
/// so this is not a performance bottleneck compared to the covariance
/// evaluations that build `a`.
///
/// Tries `nalgebra::linalg::LU` (partial-pivoted Gaussian elimination)
/// first -- cheap, and sufficient for the well-conditioned case (nugget >
/// 0 keeps the diagonal away from zero). `LU::solve` only treats an
/// *exactly* zero pivot as singular (no tolerance), so it can, in
/// principle, still miss a numerically near-singular system; that
/// tradeoff is accepted here in favor of the fallback below actually
/// covering the case NumPy's `lstsq` was covering in the Python
/// prototype: if `LU` reports singular, this falls back to
/// `nalgebra::linalg::SVD::solve`, i.e. a least-squares solve via the
/// Moore-Penrose pseudo-inverse (singular values below [`LSTSQ_EPS`] are
/// treated as zero). That fallback finds *a* solution for consistent
/// rank-deficient systems (e.g. two neighbors at the same position) and
/// the residual-minimizing solution for inconsistent ones, rather than
/// erroring out -- so `Err(SingularSystem)` is now reserved for cases
/// where even that fails (in practice: essentially never, since both `u`
/// and `v` are requested from the SVD).
pub fn solve_dense(
    a: ArrayView2<'_, f64>,
    b: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, SingularSystem> {
    let n = b.len();
    assert_eq!(a.nrows(), n, "matrix/vector size mismatch");
    assert_eq!(a.ncols(), n, "matrix must be square");

    // Convert `ndarray` views to `nalgebra` types, required by
    // `nalgebra::linalg::LU`/`SVD`.
    let a_na = DMatrix::from_fn(n, n, |r, c| a[[r, c]]);
    let b_na = DVector::from_fn(n, |r, _| b[r]);

    // try cheap LU first, then fall back to SVD least-squares if it reports singular
    if let Some(x) = a_na.clone().lu().solve(&b_na) {
        return Ok(Array1::from_iter(x.iter().copied()));
    }

    a_na.svd(true, true)
        .solve(&b_na, LSTSQ_EPS)
        .map(|x| Array1::from_iter(x.iter().copied()))
        .map_err(|_| SingularSystem)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SingularSystem;

fn euclidean_dist(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> f64 {
    a.iter()
        .zip(b.iter()) // (&f64, &f64)
        .map(|(x, y)| (x - y) * (x - y)) // |(x, y)| is type for closure
        .sum::<f64>() // sum over all dimensions
        .sqrt()
}

/// Evaluate local (moving-neighborhood) kriging at every target point.
///
/// Rust port of `local_kriging_numpy.calc_field_krige_local`. Builds a
/// k-d tree over the conditioning positions once, then for each target
/// point: finds every conditioning point within `local_radius`,
/// assembles that neighborhood's local system via [`build_local_lhs`]/
/// [`build_local_rhs`], solves it with [`solve_dense`], and accumulates
/// the field value and (raw, not yet `sill - error`) error variance.
///
/// # Arguments
/// * `cond_pos` - isometrized conditioning positions, shape `(dim, n)`
/// * `cond_val` - centered conditioning values, shape `(n,)`
/// * `target_pos` - isometrized target positions, shape `(dim, m)`
/// * `model` - covariance model
/// * `cond_err` - measurement error per conditioning point, shape `(n,)`
/// * `drift_cond` - drift terms at conditioning positions, shape `(p, n)`
///   (`p` may be 0)
/// * `drift_target` - drift terms at target positions, shape `(p, m)`
/// * `unbiased` - whether to enforce the Ordinary-Kriging unbiasedness
///   condition
/// * `exact` - whether to use `model.cov_nugget` on the RHS diagonal case
///   (target coincides with a conditioning point)
/// * `local_radius` - search radius (isometrized distance); every
///   conditioning point within this distance of a target point enters
///   its local system -- the only neighborhood-selection criterion
/// * `num_threads` - reserved for a future `rayon` parallelization of the
///   per-target-point loop below; unused for now (see module docs)
///
/// # Returns
/// `(field, error)`, both shape `(m,)`; `field` is the raw local-kriging
/// estimate, `error` is the raw `rhs . weights` (not yet post-processed
/// into a variance)
// 11 parameters, matching the Python signature it mirrors 1:1 (see module
// docs) -- not a case of a function that grew organically and should be
// split up.
#[allow(clippy::too_many_arguments)]
pub fn calc_field_krige_local(
    cond_pos: ArrayView2<'_, f64>,
    cond_val: ArrayView1<'_, f64>,
    target_pos: ArrayView2<'_, f64>,
    model: &CovModel,
    cond_err: ArrayView1<'_, f64>,
    drift_cond: ArrayView2<'_, f64>,
    drift_target: ArrayView2<'_, f64>,
    unbiased: bool,
    exact: bool,
    local_radius: f64,
    num_threads: Option<usize>,
) -> Result<(Array1<f64>, Array1<f64>), LocalKrigeError> {
    // Unused until the loop below becomes a `par_iter` (see module docs).
    // let _ = num_threads;

    let dim = cond_pos.nrows();
    let cond_no = cond_pos.ncols();
    let pnt_cnt = target_pos.ncols();

    // Create Kdtree for fast neighbor search. Each point is a column of `cond_pos`,
    // need to transpose it to get a Vec<f64> for each point.
    let mut tree: KdTree<f64, usize, Vec<f64>> = KdTree::new(dim);
    for i in 0..cond_no {
        tree.add(cond_pos.column(i).to_vec(), i)
            .expect("cond_pos columns are finite and match `dim`");
    }
    // Using squared_euclidean distance function with squared distance
    // to avoid computing square roots unnecessarily.
    // Kdtree crate docs recommend this
    let radius_sq = local_radius * local_radius;

    // Closure
    // unlike functions closures can access variables from the surrounding scope
    // this makes parallelization easier and faster because all threads can access the same data without having to copy it
    // j is the index of the target point in the target_pos array
    let solve_one = |j: usize| -> Result<(f64, f64), LocalKrigeError> {
        let target_point = target_pos.column(j).to_vec();

        // tree.within() returns a Vec of tuples (squared_distance, index) of all points within the radius
        let neighbors = tree
            .within(&target_point, radius_sq, &squared_euclidean)
            .expect("target point is finite and matches `dim`");
        if neighbors.is_empty() {
            return Err(LocalKrigeError::NoNeighbors { target_index: j });
        }

        let nbr_indices: Vec<usize> = neighbors.iter().map(|&(_, idx)| *idx).collect();
        let nbr_dists: Vec<f64> = neighbors
            .iter()
            .map(|&(sq_dist, _)| sq_dist.sqrt())
            .collect();
        let k = nbr_indices.len();

        // Select the neighbor data from the full conditioning arrays using the indices found by the k-d tree search
        let nbr_pos: Array2<f64> = cond_pos.select(Axis(1), &nbr_indices);
        let nbr_cond_err: Array1<f64> = cond_err.select(Axis(0), &nbr_indices);
        let nbr_drift: Array2<f64> = drift_cond.select(Axis(1), &nbr_indices);
        let nbr_cond_val: Array1<f64> = cond_val.select(Axis(0), &nbr_indices);

        let lhs: Array2<f64> = build_local_lhs(
            nbr_pos.view(),
            model,
            nbr_cond_err.view(),
            unbiased,
            nbr_drift.view(),
        );
        let rhs: Array1<f64> =
            build_local_rhs(&nbr_dists, model, exact, unbiased, drift_target.column(j));

        let weights: Array1<f64> = solve_dense(lhs.view(), rhs.view())
            .map_err(|_| LocalKrigeError::SingularSystem { target_index: j })?;

        let pad = drift_target.nrows() + unbiased as usize;
        let mut local_cond: Array1<f64> = Array1::<f64>::zeros(k + pad);
        for i in 0..k {
            local_cond[i] = nbr_cond_val[i];
        }

        let field_j: f64 = local_cond.dot(&weights);
        let error_j: f64 = rhs.dot(&weights);
        Ok((field_j, error_j))
    };

    // let (field, error): (Vec<f64>, Vec<f64>) = (0..pnt_cnt) // range over target points is the parameter for the closure solve_one
    //     .map(solve_one)
    //     .collect::<Result<Vec<_>, _>>()?
    //     .into_iter()
    //     .unzip();

    let results: Vec<(f64, f64)> = rayon::ThreadPoolBuilder::new()
    .num_threads(num_threads.unwrap_or(rayon::current_num_threads()))
    .build()
    .unwrap()
    .install(|| {
        (0..pnt_cnt)
            .into_par_iter()
            .map(solve_one)
            .collect::<Result<Vec<_>, _>>()
    })?;

    let (field, error): (Vec<f64>, Vec<f64>) = results.into_iter().unzip();

    Ok((Array1::from(field), Array1::from(error)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    fn gaussian(var: f64, len_scale: f64, nugget: f64) -> CovModel {
        CovModel::build(crate::covmodel_spec::CovModelSpec::Gaussian {
            var,
            len_scale,
            nugget,
        })
        .unwrap()
    }

    // --- solve_dense -----------------------------------------------------

    #[test]
    fn solve_dense_matches_hand_solved_system() {
        // [2 1; 1 3] x = [3; 5]  ->  x = [4/5, 7/5]
        let a = array![[2.0, 1.0], [1.0, 3.0]];
        let b = array![3.0, 5.0];
        let x = solve_dense(a.view(), b.view()).unwrap();
        assert!((x[0] - 0.8).abs() < 1e-10);
        assert!((x[1] - 1.4).abs() < 1e-10);
    }

    #[test]
    fn solve_dense_needs_pivoting() {
        // zero in the natural pivot position -> must swap rows to solve
        let a = array![[0.0, 1.0], [1.0, 1.0]];
        let b = array![2.0, 3.0];
        let x = solve_dense(a.view(), b.view()).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn solve_dense_lstsq_fallback_finds_min_norm_solution_for_consistent_singular_system() {
        // row2 = 2 * row1, and b is consistent with that (b2 = 2*b1): the
        // system x1 + 2*x2 = 1 has infinitely many exact solutions.  LU's
        // pivot search hits an exact zero (row2 becomes all-zero after
        // eliminating row1) and reports singular; the SVD fallback must
        // not error out here -- it picks the minimum-norm one among all
        // exact solutions, which (for a single constraint a.x = c) is
        // x = c * a / |a|^2 = (1, 2) / 5.
        let a = array![[1.0, 2.0], [2.0, 4.0]];
        let b = array![1.0, 2.0];

        let x = solve_dense(a.view(), b.view()).expect("lstsq fallback should not error out");

        assert!((x[0] - 0.2).abs() < 1e-8);
        assert!((x[1] - 0.4).abs() < 1e-8);
        // it is a genuine solution of the (consistent) original system:
        assert!((a.row(0).dot(&x) - b[0]).abs() < 1e-8);
        assert!((a.row(1).dot(&x) - b[1]).abs() < 1e-8);
    }

    #[test]
    fn solve_dense_lstsq_fallback_minimizes_residual_for_inconsistent_singular_system() {
        // Same singular `a`, but now b is *inconsistent* with row2 = 2 *
        // row1 (b2 != 2*b1): no exact solution exists. This is the case
        // NumPy's `lstsq` (and our SVD fallback) is actually for --
        // minimizing |a.x - b|^2. Since a = (1,2)(1,2)^T, a.x only depends
        // on s = x1 + 2*x2, and minimizing |(s,2s) - (1,3)|^2 over s gives
        // s = 1.4 (hand-derived below), with the minimum-norm x for that s
        // being s * (1, 2) / 5 = (0.28, 0.56).
        let a = array![[1.0, 2.0], [2.0, 4.0]];
        let b = array![1.0, 3.0];

        let x = solve_dense(a.view(), b.view()).expect("lstsq fallback should not error out");

        assert!((x[0] - 0.28).abs() < 1e-8);
        assert!((x[1] - 0.56).abs() < 1e-8);

        // sanity: this residual must be strictly smaller than for a
        // (non-least-squares) candidate exact solution of just row 1.
        let exact_row0_only = array![1.0, 0.0]; // satisfies row 0 exactly, ignores row 1
        let residual = |v: &Array1<f64>| {
            let r0 = a.row(0).dot(v) - b[0];
            let r1 = a.row(1).dot(v) - b[1];
            r0 * r0 + r1 * r1
        };
        assert!(residual(&x) < residual(&exact_row0_only));
    }

    // --- build_local_lhs / build_local_rhs --------------------------------

    #[test]
    fn lhs_matches_model_covariance_and_adds_cond_err_on_diagonal() {
        let model = gaussian(1.5, 4.0, 0.1);
        let nbr_pos = array![[0.0, 3.0], [0.0, 0.0]]; // 1D positions on the x-axis
        let cond_err = array![0.1, 0.2];
        let drift = Array2::<f64>::zeros((0, 2));

        let lhs = build_local_lhs(nbr_pos.view(), &model, cond_err.view(), false, drift.view());

        assert_eq!(lhs.dim(), (2, 2));
        assert!((lhs[[0, 0]] - (model.covariance(0.0) + 0.1)).abs() < 1e-12);
        assert!((lhs[[1, 1]] - (model.covariance(0.0) + 0.2)).abs() < 1e-12);
        assert!((lhs[[0, 1]] - model.covariance(3.0)).abs() < 1e-12);
        assert_eq!(lhs[[0, 1]], lhs[[1, 0]]); // symmetric
    }

    #[test]
    fn lhs_unbiased_adds_ones_row_and_column() {
        let model = gaussian(1.0, 2.0, 0.0);
        let nbr_pos = array![[0.0, 1.0]];
        let cond_err = array![0.0, 0.0];
        let drift = Array2::<f64>::zeros((0, 2));

        let lhs = build_local_lhs(nbr_pos.view(), &model, cond_err.view(), true, drift.view());

        assert_eq!(lhs.dim(), (3, 3));
        assert_eq!(lhs.row(2).to_vec(), vec![1.0, 1.0, 0.0]);
        assert_eq!(lhs.column(2).to_vec(), vec![1.0, 1.0, 0.0]);
    }

    #[test]
    fn rhs_uses_cov_nugget_only_in_exact_mode() {
        let model = gaussian(1.5, 4.0, 0.2);
        let drift = Array1::<f64>::zeros(0);

        let rhs_normal = build_local_rhs(&[0.0, 3.0], &model, false, false, drift.view());
        let rhs_exact = build_local_rhs(&[0.0, 3.0], &model, true, false, drift.view());

        // away from the origin both agree...
        assert!((rhs_normal[1] - rhs_exact[1]).abs() < 1e-12);
        // ...but at distance 0 `exact` must include the nugget (the sill).
        assert!((rhs_normal[0] - model.covariance(0.0)).abs() < 1e-12);
        assert!((rhs_exact[0] - model.sill()).abs() < 1e-12);
    }

    // --- calc_field_krige_local -------------------------------------------

    fn scattered_cond() -> (Array2<f64>, Array1<f64>) {
        let cond_pos = array![[0.0, 4.0, 8.0, 2.0, 6.0], [0.0, 3.0, 1.0, 7.0, 5.0]];
        let cond_val = array![0.3, -0.5, 1.2, 0.1, -0.8];
        (cond_pos, cond_val)
    }

    #[test]
    fn exact_mode_reproduces_conditioning_value_at_conditioning_point() {
        let model = gaussian(2.0, 3.0, 0.0);
        let (cond_pos, cond_val) = scattered_cond();
        let cond_no = cond_val.len();
        let cond_err = Array1::<f64>::zeros(cond_no);
        let drift_cond = Array2::<f64>::zeros((0, cond_no));

        // target == the first conditioning point exactly
        let target_pos = cond_pos.column(0).to_owned().insert_axis(Axis(1));
        let drift_target = Array2::<f64>::zeros((0, 1));

        let (field, _error) = calc_field_krige_local(
            cond_pos.view(),
            cond_val.view(),
            target_pos.view(),
            &model,
            cond_err.view(),
            drift_cond.view(),
            drift_target.view(),
            true,
            true,
            100.0,
            None,
        )
        .unwrap();

        assert!((field[0] - cond_val[0]).abs() < 1e-6);
    }

    #[test]
    fn no_neighbors_within_radius_is_an_error() {
        let model = gaussian(1.0, 2.0, 0.1);
        let (cond_pos, cond_val) = scattered_cond();
        let cond_no = cond_val.len();
        let cond_err = Array1::<f64>::from_elem(cond_no, 0.1);
        let drift_cond = Array2::<f64>::zeros((0, cond_no));

        let target_pos = array![[1000.0], [1000.0]]; // far from every cond point
        let drift_target = Array2::<f64>::zeros((0, 1));

        let result = calc_field_krige_local(
            cond_pos.view(),
            cond_val.view(),
            target_pos.view(),
            &model,
            cond_err.view(),
            drift_cond.view(),
            drift_target.view(),
            true,
            false,
            1.0,
            None,
        );

        assert_eq!(
            result,
            Err(LocalKrigeError::NoNeighbors { target_index: 0 })
        );
    }

    #[test]
    fn matches_manual_lhs_rhs_solve_when_radius_covers_everyone() {
        let model = gaussian(1.5, 3.0, 0.1);
        let (cond_pos, cond_val) = scattered_cond();
        let cond_no = cond_val.len();
        let cond_err = Array1::<f64>::from_elem(cond_no, 0.1);
        let drift_cond = Array2::<f64>::zeros((0, cond_no));

        let target_pos = array![[3.0], [3.0]];
        let drift_target = Array2::<f64>::zeros((0, 1));

        let (field, error) = calc_field_krige_local(
            cond_pos.view(),
            cond_val.view(),
            target_pos.view(),
            &model,
            cond_err.view(),
            drift_cond.view(),
            drift_target.view(),
            true,
            false,
            1e6,
            None,
        )
        .unwrap();

        // Manually assemble the very same (full-neighborhood) system with
        // build_local_lhs/build_local_rhs/solve_dense directly, bypassing
        // the k-d tree entirely, and check the two paths agree.
        let target = target_pos.column(0);
        let dists: Vec<f64> = (0..cond_no)
            .map(|i| euclidean_dist(cond_pos.column(i), target))
            .collect();
        let lhs = build_local_lhs(
            cond_pos.view(),
            &model,
            cond_err.view(),
            true,
            drift_cond.view(),
        );
        let rhs = build_local_rhs(&dists, &model, false, true, drift_target.column(0));
        let weights = solve_dense(lhs.view(), rhs.view()).unwrap();

        let mut local_cond = Array1::<f64>::zeros(cond_no + 1);
        for i in 0..cond_no {
            local_cond[i] = cond_val[i];
        }
        let expected_field = local_cond.dot(&weights);
        let expected_error = rhs.dot(&weights);

        assert!((field[0] - expected_field).abs() < 1e-8);
        assert!((error[0] - expected_error).abs() < 1e-8);
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
        let cond_val = array![0.1, 0.4, -0.2, 0.9];
        let cond_no = cond_val.len();
        let cond_err = Array1::<f64>::from_elem(cond_no, 0.1);
        let drift_cond = Array2::<f64>::zeros((0, cond_no));
        let target_pos = array![[1.0, 6.0]]; // shape (1, 2)
        let drift_target = Array2::<f64>::zeros((0, 2));

        let (field, _error) = calc_field_krige_local(
            cond_pos.view(),
            cond_val.view(),
            target_pos.view(),
            &model,
            cond_err.view(),
            drift_cond.view(),
            drift_target.view(),
            true,
            false,
            100.0,
            None,
        )
        .unwrap();

        assert_eq!(field.len(), 2);
    }

    #[test]
    fn works_with_3d_positions() {
        let model = gaussian(1.0, 3.0, 0.1);
        let cond_pos = array![
            [0.0, 2.0, 5.0, 8.0],
            [1.0, 0.0, 3.0, 2.0],
            [4.0, 1.0, 0.0, 5.0],
        ]; // shape (3, 4)
        let cond_val = array![0.1, 0.4, -0.2, 0.9];
        let cond_no = cond_val.len();
        let cond_err = Array1::<f64>::from_elem(cond_no, 0.1);
        let drift_cond = Array2::<f64>::zeros((0, cond_no));
        let target_pos = array![[1.0, 6.0], [0.5, 1.0], [2.0, 3.0]]; // shape (3, 2)
        let drift_target = Array2::<f64>::zeros((0, 2));

        let (field, _error) = calc_field_krige_local(
            cond_pos.view(),
            cond_val.view(),
            target_pos.view(),
            &model,
            cond_err.view(),
            drift_cond.view(),
            drift_target.view(),
            true,
            false,
            100.0,
            None,
        )
        .unwrap();

        assert_eq!(field.len(), 2);
    }
}
