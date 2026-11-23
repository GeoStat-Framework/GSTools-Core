//! Helpers shared by the kriging tests.
//!
//! [`build_lhs`] / [`build_rhs`] (with
//! [`solve_kriging`](crate::krige::solve::solve_kriging)) assemble one
//! local kriging system the straightforward way (layout: see
//! [`crate::krige::local`]). They are an independent reference for the
//! tests; the production code selects sub-matrices of a cached full
//! neighborhood instead.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::covmodel::CovModel;
use crate::covmodel_spec::CovModelSpec;
use crate::krige::local::euclidean_dist;

/// Small deterministic random number generator (no extra dependency).
pub(crate) struct Lcg(pub(crate) u64);

impl Lcg {
    /// Uniform in `[0, 1)`.
    pub(crate) fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `[lo, hi)`.
    pub(crate) fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
}

pub(crate) fn gaussian(var: f64, len_scale: f64, nugget: f64) -> CovModel {
    CovModel::build(CovModelSpec::Gaussian {
        var,
        len_scale,
        nugget,
    })
    .unwrap()
}

pub(crate) fn exponential(var: f64, len_scale: f64, nugget: f64) -> CovModel {
    CovModel::build(CovModelSpec::Exponential {
        var,
        len_scale,
        nugget,
    })
    .unwrap()
}

/// Assemble the local kriging matrix (LHS) for one target point's
/// neighborhood: pairwise covariances between the `k` neighbors, plus
/// measurement error on the diagonal, plus (optionally) the
/// unbiasedness row/column and drift rows/columns.
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
pub(crate) fn build_lhs(
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
/// entries matching [`build_lhs`]'s padding.
///
/// # Arguments
/// * `nbr_dists` - distance from each neighbor to the target point,
///   shape `(k,)`
/// * `model` - covariance model
/// * `exact` - use `model.cov_nugget` instead of `model.covariance`, so a
///   target point that coincides with a neighbor reproduces its value
///   exactly
/// * `unbiased` - must match [`build_lhs`]
/// * `drift_target` - drift terms evaluated at the target position,
///   shape `(p,)`, `p` must match [`build_lhs`]'s `drift_nbrs`
///
/// # Returns
/// the RHS vector, same length as [`build_lhs`]'s matrix dimension
pub(crate) fn build_rhs(
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn lhs_matches_model_covariance_and_adds_cond_err_on_diagonal() {
        let model = gaussian(1.5, 4.0, 0.1);
        let nbr_pos = array![[0.0, 3.0], [0.0, 0.0]]; // 1D positions on the x-axis
        let cond_err = array![0.1, 0.2];
        let drift = Array2::<f64>::zeros((0, 2));

        let lhs = build_lhs(nbr_pos.view(), &model, cond_err.view(), false, drift.view());

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

        let lhs = build_lhs(nbr_pos.view(), &model, cond_err.view(), true, drift.view());

        assert_eq!(lhs.dim(), (3, 3));
        assert_eq!(lhs.row(2).to_vec(), vec![1.0, 1.0, 0.0]);
        assert_eq!(lhs.column(2).to_vec(), vec![1.0, 1.0, 0.0]);
    }

    #[test]
    fn rhs_uses_cov_nugget_only_in_exact_mode() {
        let model = gaussian(1.5, 4.0, 0.2);
        let drift = Array1::<f64>::zeros(0);

        let rhs_normal = build_rhs(&[0.0, 3.0], &model, false, false, drift.view());
        let rhs_exact = build_rhs(&[0.0, 3.0], &model, true, false, drift.view());

        // away from the origin both agree...
        assert!((rhs_normal[1] - rhs_exact[1]).abs() < 1e-12);
        // ...but at distance 0 `exact` must include the nugget (the sill).
        assert!((rhs_normal[0] - model.covariance(0.0)).abs() < 1e-12);
        assert!((rhs_exact[0] - model.sill()).abs() < 1e-12);
    }
}
