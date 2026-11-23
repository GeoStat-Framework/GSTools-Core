//! Linear algebra for the local kriging systems. Knows nothing about
//! kriging itself (layout and drift rule: [`crate::krige::local`]).
//!
//! //! A kriging system with `k` neighbors and `p` constraints (unbiasedness
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
//! `SCHUR_EPS`; e.g. two neighbors at the same position, or a drift with
//! the same value at all neighbors) [`solve_saddle_point`] falls back to
//! [`solve_dense`] (LU, then SVD least squares).
//!
//! Entry points:
//! * [`solve_saddle_point`] - the system given by entry functions (used by
//!   local kriging)
//! * [`solve_kriging`] - the same for an assembled matrix; not used by the
//!   production code anymore, kept as a reference for the tests (with
//!   `build_lhs` / `build_rhs` in `krige/test_util.rs`)
//! * [`solve_dense`] - general dense solver, the fallback

use std::cell::RefCell;

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor::{
    cholesky_in_place, cholesky_in_place_scratch, LltRegularization,
};
use faer::linalg::cholesky::llt::solve::{solve_in_place, solve_in_place_scratch};
use faer::linalg::solvers::{PartialPivLu, Solve, Svd};
use faer::{Mat, MatMut, MatRef, Par};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Relative pivot tolerance for the Schur complement `S`: its Cholesky
/// factorization is rejected if a pivot `L_ii^2` is not larger than
/// `SCHUR_EPS` times the largest diagonal entry of `S`. Cholesky alone only
/// fails for pivots `<= 0`; this also catches a nearly rank-deficient `F`
/// (e.g. drift values that are equal up to rounding), which would give huge
/// Lagrange multipliers.
const SCHUR_EPS: f64 = 1e-12;

/// Reusable buffers for [`solve_cholesky_schur`].
#[derive(Default)]
struct SolveWorkspace {
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
fn with_workspace<R>(f: impl FnOnce(&mut SolveWorkspace) -> R) -> R {
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
/// definite or the result is not finite; `x` is then unspecified.
/// [`solve_saddle_point`] then falls back to [`solve_dense`].
fn solve_cholesky_schur(
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

/// Solve the saddle-point system (see module docs): Cholesky + Schur
/// complement, and if that fails [`solve_dense`] on the same entries.
///
/// # Arguments
/// * `k`, `p` - number of neighbors and constraints
/// * `cov(a, b)` - entry of `C`; must be symmetric and defined for all
///   `a, b < k` (the fast path only reads `a >= b`, the fallback all)
/// * `con(i, q)` - entry of `F`
/// * `rhs` - `[r; f]`, length `k + p`
/// * `x` - output `[w; mu]`, resized to `k + p`
pub fn solve_saddle_point(
    k: usize,
    p: usize,
    cov: impl Fn(usize, usize) -> f64,
    con: impl Fn(usize, usize) -> f64,
    rhs: &[f64],
    x: &mut Vec<f64>,
) -> Result<(), SingularSystem> {
    if with_workspace(|ws| solve_cholesky_schur(ws, k, p, &cov, &con, rhs, x)) {
        return Ok(());
    }
    // fallback: full (k + p, k + p) matrix, LU then SVD least squares
    let n = k + p;
    let mut lhs = Array2::<f64>::zeros((n, n));
    for a in 0..k {
        for b in 0..k {
            lhs[[a, b]] = cov(a, b);
        }
        for q in 0..p {
            lhs[[a, k + q]] = con(a, q);
            lhs[[k + q, a]] = con(a, q);
        }
    }
    let sol = solve_dense(lhs.view(), ArrayView1::from(rhs))?;
    x.clear();
    x.extend(sol.iter().copied());
    Ok(())
}

/// **Not used by the production code anymore**, kept as a reference for the
/// tests (together with `build_lhs` / `build_rhs` in `krige/test_util.rs`).
///
/// Solve an assembled local kriging system in the layout of
/// the local kriging system (see [`crate::krige::local`]): the first `k`
/// rows/columns are the neighbors, the rest the constraints. `lhs` must be
/// symmetric. See [`solve_saddle_point`].
pub fn solve_kriging(
    lhs: ArrayView2<'_, f64>,
    rhs: ArrayView1<'_, f64>,
    k: usize,
) -> Result<Array1<f64>, SingularSystem> {
    let n = rhs.len();
    assert!(k <= n, "more neighbors than unknowns");
    let mut x = Vec::with_capacity(n);
    solve_saddle_point(
        k,
        n - k,
        |a, b| lhs[[a, b]],
        |i, q| lhs[[i, k + q]],
        &rhs.to_vec(),
        &mut x,
    )?;
    Ok(Array1::from_vec(x))
}

/// Tolerance below which an SVD singular value is treated as zero in the
/// least-squares fallback -- analogous to NumPy `lstsq`'s default `rcond`
/// cutoff.
const LSTSQ_EPS: f64 = 1e-12;

/// Solve the dense linear system `a @ x = b`. Pure function (takes views,
/// returns a fresh `Array1`), so it is independently testable with plain
/// literal matrices, unrelated to kriging.
///
/// Kriging systems are normally solved by the faster
/// [`solve_saddle_point`] (Cholesky + Schur complement); this general
/// solver is its fallback when that fails (e.g. two neighbors at the same
/// position).
///
/// 1. `faer`'s partial-pivoted `LU`. Accepted unless a pivot is *exactly*
///    zero or the result is not finite (no tolerance, so a numerically
///    near-singular system can still pass).
/// 2. Otherwise a least-squares solve via the SVD pseudo-inverse
///    (`x = V S^+ U^T b`, singular values not above `LSTSQ_EPS` (1e-12) are
///    treated as zero), as NumPy's `lstsq` in the Python prototype. It
///    finds *a* solution for consistent rank-deficient systems and the
///    residual-minimizing one for inconsistent ones.
///
/// `Err(SingularSystem)` only if the SVD itself fails (does not converge),
/// in practice essentially never.
pub fn solve_dense(
    a: ArrayView2<'_, f64>,
    b: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, SingularSystem> {
    let n = b.len();
    assert_eq!(a.nrows(), n, "matrix/vector size mismatch");
    assert_eq!(a.ncols(), n, "matrix must be square");

    let a_f = Mat::<f64>::from_fn(n, n, |r, c| a[[r, c]]);

    // 1. LU with partial pivoting
    let lu = PartialPivLu::new(a_f.as_ref());
    let u = lu.U();
    if (0..n).all(|i| u[(i, i)] != 0.0) {
        let mut x = Mat::<f64>::from_fn(n, 1, |r, _| b[r]);
        lu.solve_in_place(x.as_mut());
        if x.col(0).iter().all(|v| v.is_finite()) {
            return Ok(Array1::from_iter(x.col(0).iter().copied()));
        }
    }

    // 2. least squares via the SVD pseudo-inverse
    let svd = Svd::new(a_f.as_ref()).map_err(|_| SingularSystem)?;
    let (u, sv, v) = (svd.U(), svd.S(), svd.V());
    // c = S^+ U^T b
    let c: Vec<f64> = (0..n)
        .map(|i| {
            let sigma = sv[i];
            if sigma > LSTSQ_EPS {
                (0..n).map(|r| u[(r, i)] * b[r]).sum::<f64>() / sigma
            } else {
                0.0
            }
        })
        .collect();
    // x = V c
    Ok(Array1::from_iter(
        (0..n).map(|r| (0..n).map(|i| v[(r, i)] * c[i]).sum::<f64>()),
    ))
}

/// A linear system could not be solved, not even by the least-squares
/// fallback of [`solve_dense`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SingularSystem;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::krige::test_util::Lcg;
    use ndarray::array;

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

    /// `solve_cholesky_schur` on a system in `build_lhs` layout
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
}
