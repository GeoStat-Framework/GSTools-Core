//! Fast solver for local kriging systems, and the rule for neighborhoods
//! that are too small for the drift.
//!
//! # Too few neighbors for the drift: the drift is dropped
//!
//! ```text
//! p = unbiased (0 or 1) + number of drift terms        (number of constraints)
//! n = neighbors with data
//!
//! n >  p   ->  solved as requested (universal / external drift kriging)
//! n <= p   ->  ALL drift terms are dropped: ordinary kriging
//!              (simple kriging if `unbiased` is false)
//! no drift ->  never changes anything
//! ```
//!
//! With `n <= p` the constraints alone fix the weights and the covariance
//! model has no influence. For example external drift kriging with two
//! neighbors (`p = 2`) solves `w1 + w2 = 1`, `w1 h1 + w2 h2 = h0`, so
//! `w1 = (h0 - h2) / (h1 - h2)`: a straight line through two points,
//! singular for `h1 = h2` and extrapolating (negative weights) whenever
//! `h0` lies outside `[h1, h2]`. With one neighbor there is no solution at
//! all. Ordinary kriging is well-defined for any `n >= 1`.
//!
//! The rule lives in [`drift_terms_used`] and is applied by the local
//! kriging functions
//! ([`calc_field_krige_local`](crate::local_krige::calc_field_krige_local),
//! [`calc_field_krige_local_multi`](crate::local_krige_multi::calc_field_krige_local_multi)).
//! The returned kriging variance is the one of the system actually solved.
//!
//! # Solver
//!
//! A kriging system with `k` neighbors and `p` constraints (unbiasedness
//! and drift terms) has the saddle-point form
//!
//! ```text
//! [ C   F ] [ w  ]   [ r ]      C: (k, k) covariance between neighbors (+ cond_err)
//! [ F^T 0 ] [ mu ] = [ f ]      F: (k, p) constraint columns (ones, drift values)
//! ```
//!
//! `C` is symmetric positive definite for a valid covariance model and
//! distinct neighbors, so instead of an LU of the whole `(k+p, k+p)`
//! matrix it is cheaper to factorize `C = L L^T` (Cholesky, about half the
//! flops of LU) and eliminate `w`:
//!
//! ```text
//! [Y | z] = C^-1 [F | r]
//! (F^T Y) mu = F^T z - f      <- small (p, p) Schur complement system
//! w = z - Y mu
//! ```
//!
//! The Schur complement `S = F^T C^-1 F` is symmetric positive definite as
//! well (if `F` has full column rank), so it is solved by Cholesky too.
//! Both factorizations use `faer` single-threaded (callers parallelize over
//! target points) and per-thread buffers, so a solve does not allocate.
//! If `C` or `S` is not positive definite, or `S` is nearly singular (see
//! [`SCHUR_EPS`]; e.g. two neighbors at the same position, or a drift with
//! the same value at all neighbors) [`solve_kriging`] falls back to
//! [`solve_dense`] (LU, then SVD least squares).

use std::cell::RefCell;

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor::{
    cholesky_in_place, cholesky_in_place_scratch, LltRegularization,
};
use faer::linalg::cholesky::llt::solve::{solve_in_place, solve_in_place_scratch};
use faer::{MatMut, MatRef, Par};
use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::local_krige::{solve_dense, SingularSystem};

// =========================================================================
//   RULE: TOO FEW NEIGHBORS FOR THE DRIFT  ->  ALL DRIFT TERMS ARE DROPPED
//
//     n_neighbors <= unbiased + drift_no   ->   0 drift terms
//                                                (ordinary / simple kriging)
//     otherwise                            ->   drift_no drift terms
//
//   See the module docs for why. Used by every local kriging function.
// =========================================================================

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

/// Relative pivot tolerance for the Schur complement `S`: its Cholesky
/// factorization is rejected if a pivot `L_ii^2` is not larger than
/// `SCHUR_EPS` times the largest diagonal entry of `S`. Cholesky alone only
/// fails for pivots `<= 0`; this also catches a nearly rank-deficient `F`
/// (e.g. drift values that are equal up to rounding), which would give huge
/// Lagrange multipliers.
const SCHUR_EPS: f64 = 1e-12;

/// Reusable buffers for [`solve_cholesky_schur`].
#[derive(Default)]
pub(crate) struct SolveWorkspace {
    /// `(k, k)` covariance block, column-major, factorized in place
    c: Vec<f64>,
    /// `(k, p + 1)` right-hand sides `[F | r]`, solved in place
    y: Vec<f64>,
    /// `(p, p)` Schur complement, column-major, factorized in place
    s: Vec<f64>,
    /// `(p,)` right-hand side of the Schur system, solved in place (`mu`)
    t: Vec<f64>,
    /// scratch memory for `faer`
    mem: Option<MemBuffer>,
    /// largest `(k, p)` the scratch memory was sized for
    mem_size: (usize, usize),
}

thread_local! {
    /// One workspace per thread (rayon threads live for the whole call).
    static WORKSPACE: RefCell<SolveWorkspace> = RefCell::new(SolveWorkspace::default());
}

/// Run `f` with this thread's workspace.
pub(crate) fn with_workspace<R>(f: impl FnOnce(&mut SolveWorkspace) -> R) -> R {
    WORKSPACE.with(|ws| f(&mut ws.borrow_mut()))
}

/// Solve the saddle-point system above via Cholesky + Schur complement.
///
/// # Arguments
/// * `k`, `p` - number of neighbors and constraints
/// * `cov(a, b)` - entry of `C` (only `a >= b` is read)
/// * `con(i, q)` - entry of `F`
/// * `rhs` - `[r; f]`, length `k + p`
/// * `x` - output `[w; mu]`, resized to `k + p`
///
/// # Returns
/// `false` if `C` or the Schur complement is not (numerically) positive
/// definite or the result is not finite; `x` is then unspecified and the
/// caller should fall back to [`solve_dense`].
pub(crate) fn solve_cholesky_schur(
    ws: &mut SolveWorkspace,
    k: usize,
    p: usize,
    cov: impl Fn(usize, usize) -> f64,
    con: impl Fn(usize, usize) -> f64,
    rhs: &[f64],
    x: &mut Vec<f64>,
) -> bool {
    debug_assert_eq!(rhs.len(), k + p);
    x.clear();
    x.resize(k + p, 0.0);
    if k == 0 {
        return false;
    }
    let nrhs = p + 1;

    // scratch memory, grown when needed
    // (sized for the (k, k) and the (p, p) factorization)
    if ws.mem.is_none() || k.max(p) > ws.mem_size.0 || p > ws.mem_size.1 {
        let size = (k.max(p).max(ws.mem_size.0), p.max(ws.mem_size.1));
        let req = cholesky_in_place_scratch::<f64>(size.0, Par::Seq, Default::default())
            .or(solve_in_place_scratch::<f64>(size.0, size.1 + 1, Par::Seq));
        ws.mem = Some(MemBuffer::new(req));
        ws.mem_size = size;
    }
    let stack = MemStack::new(ws.mem.as_mut().expect("allocated above"));

    // C (lower triangle) and [F | r], column-major
    ws.c.resize(k * k, 0.0);
    for b in 0..k {
        for a in b..k {
            ws.c[a + b * k] = cov(a, b);
        }
    }
    ws.y.resize(k * nrhs, 0.0);
    for q in 0..p {
        for i in 0..k {
            ws.y[i + q * k] = con(i, q);
        }
    }
    ws.y[p * k..(p + 1) * k].copy_from_slice(&rhs[..k]);

    // C = L L^T, then [Y | z] = C^-1 [F | r]
    let mut c = MatMut::from_column_major_slice_mut(ws.c.as_mut_slice(), k, k);
    if cholesky_in_place(
        c.as_mut(),
        LltRegularization::default(),
        Par::Seq,
        stack,
        Default::default(),
    )
    .is_err()
    {
        return false;
    }
    let l = MatRef::from_column_major_slice(ws.c.as_slice(), k, k);
    let ymat = MatMut::from_column_major_slice_mut(ws.y.as_mut_slice(), k, nrhs);
    solve_in_place(l, ymat, Par::Seq, stack);

    let y = &ws.y;
    let z = &y[p * k..(p + 1) * k];
    if p > 0 {
        // Schur complement S = F^T Y (p, p, lower triangle, column-major)
        // and t = F^T z - f
        ws.s.resize(p * p, 0.0);
        ws.t.resize(p, 0.0);
        let mut scale = 0.0f64; // largest diagonal entry of S
        for b in 0..p {
            for a in b..p {
                ws.s[a + b * p] = (0..k).map(|i| con(i, a) * y[i + b * k]).sum();
            }
            scale = scale.max(ws.s[b + b * p]);
            ws.t[b] = (0..k).map(|i| con(i, b) * z[i]).sum::<f64>() - rhs[k + b];
        }
        if scale <= 0.0 || !scale.is_finite() {
            return false;
        }
        // S = L_S L_S^T, rejected if not positive definite or nearly
        // singular (see SCHUR_EPS)
        let mut smat = MatMut::from_column_major_slice_mut(ws.s.as_mut_slice(), p, p);
        if cholesky_in_place(
            smat.as_mut(),
            LltRegularization::default(),
            Par::Seq,
            stack,
            Default::default(),
        )
        .is_err()
        {
            return false;
        }
        let ls = MatRef::from_column_major_slice(ws.s.as_slice(), p, p);
        if (0..p).any(|i| ls[(i, i)] * ls[(i, i)] <= SCHUR_EPS * scale) {
            return false;
        }
        // mu = S^-1 t
        let tmat = MatMut::from_column_major_slice_mut(ws.t.as_mut_slice(), p, 1);
        solve_in_place(ls, tmat, Par::Seq, stack);
        let mu = &ws.t;
        for i in 0..k {
            x[i] = z[i] - (0..p).map(|q| y[i + q * k] * mu[q]).sum::<f64>();
        }
        x[k..].copy_from_slice(mu);
    } else {
        x.copy_from_slice(z);
    }
    x.iter().all(|v| v.is_finite())
}

/// Solve a local kriging system in the layout of
/// [`build_local_lhs`](crate::local_krige::build_local_lhs) /
/// [`build_local_rhs`](crate::local_krige::build_local_rhs): the first `k`
/// rows/columns are the neighbors, the rest the constraints.
///
/// Uses [`solve_cholesky_schur`] and falls back to [`solve_dense`] (LU,
/// then SVD least squares) if that fails.
pub fn solve_kriging(
    lhs: ArrayView2<'_, f64>,
    rhs: ArrayView1<'_, f64>,
    k: usize,
) -> Result<Array1<f64>, SingularSystem> {
    let n = rhs.len();
    assert!(k <= n, "more neighbors than unknowns");
    let mut x = Vec::with_capacity(n);
    let rhs_vec = rhs.to_vec();
    let ok = with_workspace(|ws| {
        solve_cholesky_schur(
            ws,
            k,
            n - k,
            |a, b| lhs[[a, b]],
            |i, q| lhs[[i, k + q]],
            &rhs_vec,
            &mut x,
        )
    });
    if ok {
        Ok(Array1::from_vec(x))
    } else {
        solve_dense(lhs, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// exponential-covariance kriging system with `p` constraints
    /// (unbiasedness + `p - 1` drift terms)
    fn system(k: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut r = Lcg(seed);
        let pts: Vec<[f64; 2]> = (0..k).map(|_| [r.next() * 10.0, r.next() * 10.0]).collect();
        let n = k + p;
        let mut lhs = Array2::<f64>::zeros((n, n));
        let mut rhs = Array1::<f64>::zeros(n);
        for a in 0..k {
            for b in 0..k {
                let d = ((pts[a][0] - pts[b][0]).powi(2) + (pts[a][1] - pts[b][1]).powi(2)).sqrt();
                lhs[[a, b]] = 2.0 * (-d / 3.0).exp();
            }
            rhs[a] = 2.0 * (-(pts[a][0].powi(2) + pts[a][1].powi(2)).sqrt() / 3.0).exp();
            for q in 0..p {
                let v = if q == 0 { 1.0 } else { 100.0 * r.next() };
                lhs[[a, k + q]] = v;
                lhs[[k + q, a]] = v;
            }
        }
        for q in 0..p {
            rhs[k + q] = if q == 0 { 1.0 } else { 50.0 * r.next() };
        }
        (lhs, rhs)
    }

    fn max_rel_diff(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let scale = b.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        a.iter()
            .zip(b)
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
            / scale
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
    fn matches_lu_for_any_number_of_constraints() {
        for p in 0..4 {
            for (k, seed) in [(5, 1), (30, 2), (170, 3)] {
                let (lhs, rhs) = system(k, p, seed);
                let x = solve_kriging(lhs.view(), rhs.view(), k).unwrap();
                let reference = solve_dense(lhs.view(), rhs.view()).unwrap();
                assert!(max_rel_diff(&x, &reference) < 1e-10, "k={k} p={p}");
            }
        }
    }

    #[test]
    fn cholesky_path_is_taken_for_a_regular_system() {
        let (lhs, rhs) = system(20, 2, 7);
        let mut x = Vec::new();
        let ok = with_workspace(|ws| {
            solve_cholesky_schur(
                ws,
                20,
                2,
                |a, b| lhs[[a, b]],
                |i, q| lhs[[i, 20 + q]],
                rhs.as_slice().unwrap(),
                &mut x,
            )
        });
        assert!(ok);
    }

    #[test]
    fn workspace_is_reused_across_sizes() {
        // big, small, big again: buffers must be resized correctly
        for (k, p) in [(100, 2), (4, 1), (150, 3), (10, 0)] {
            let (lhs, rhs) = system(k, p, k as u64);
            let x = solve_kriging(lhs.view(), rhs.view(), k).unwrap();
            let reference = solve_dense(lhs.view(), rhs.view()).unwrap();
            assert!(max_rel_diff(&x, &reference) < 1e-10, "k={k} p={p}");
        }
    }

    #[test]
    fn duplicate_neighbors_fall_back_to_least_squares() {
        // two neighbors at the same position: C is singular
        let (mut lhs, mut rhs) = system(6, 1, 11);
        for c in 0..7 {
            lhs[[1, c]] = lhs[[0, c]];
        }
        for r in 0..7 {
            lhs[[r, 1]] = lhs[[r, 0]];
        }
        rhs[1] = rhs[0];
        let x = solve_kriging(lhs.view(), rhs.view(), 6).unwrap();
        let reference = solve_dense(lhs.view(), rhs.view()).unwrap();
        // the system is consistent: the solution must satisfy it
        let residual = (&lhs.dot(&x) - &rhs)
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(residual < 1e-8, "residual {residual}");
        // and the two duplicates share the weight like the fallback does
        assert!((x[0] + x[1] - reference[0] - reference[1]).abs() < 1e-8);
    }

    /// `solve_cholesky_schur` on a system in `build_local_lhs` layout
    fn cholesky_schur(lhs: &Array2<f64>, rhs: &Array1<f64>, k: usize) -> bool {
        let mut x = Vec::new();
        with_workspace(|ws| {
            solve_cholesky_schur(
                ws,
                k,
                lhs.nrows() - k,
                |a, b| lhs[[a, b]],
                |i, q| lhs[[i, k + q]],
                rhs.as_slice().unwrap(),
                &mut x,
            )
        })
    }

    #[test]
    fn rank_deficient_constraints_fall_back() {
        // drift column = 5 * unbiasedness column (+ tiny noise): F has
        // (nearly) rank 1, S is (nearly) singular. Cholesky itself fails for
        // 0 and 1e-13; for 1e-7 it succeeds with a tiny positive pivot that
        // only the SCHUR_EPS check rejects.
        for noise in [0.0, 1e-13, 1e-7] {
            let (mut lhs, rhs) = system(12, 2, 21);
            let mut r = Lcg(5);
            for a in 0..12 {
                let v = 5.0 + noise * r.next();
                lhs[[a, 13]] = v;
                lhs[[13, a]] = v;
            }
            assert!(!cholesky_schur(&lhs, &rhs, 12), "noise={noise}");
            // the fallback still gives a solution
            assert!(solve_kriging(lhs.view(), rhs.view(), 12).is_ok());
        }
    }

    #[test]
    fn indefinite_covariance_block_is_rejected() {
        let c = [[1.0, 2.0], [2.0, 1.0]];
        let mut x = Vec::new();
        let ok = with_workspace(|ws| {
            solve_cholesky_schur(ws, 2, 0, |a, b| c[a][b], |_, _| 0.0, &[1.0, 1.0], &mut x)
        });
        assert!(!ok);
    }
}
