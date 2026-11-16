//! Fast solver for local kriging systems.
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
//! The factorization uses `faer` single-threaded (callers parallelize over
//! target points) and per-thread buffers, so a solve does not allocate.
//! If `C` or the Schur complement is not positive definite (e.g. two
//! neighbors at the same position) [`solve_kriging`] falls back to
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

/// Relative pivot tolerance for the small Schur complement system.
const SCHUR_EPS: f64 = 1e-12;

/// Reusable buffers for [`solve_cholesky_schur`].
#[derive(Default)]
pub(crate) struct SolveWorkspace {
    /// `(k, k)` covariance block, column-major, factorized in place
    c: Vec<f64>,
    /// `(k, p + 1)` right-hand sides `[F | r]`, solved in place
    y: Vec<f64>,
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
    if ws.mem.is_none() || k > ws.mem_size.0 || p > ws.mem_size.1 {
        let size = (k.max(ws.mem_size.0), p.max(ws.mem_size.1));
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
        // Schur complement S = F^T Y (p, p) and t = F^T z - f
        let mut s = vec![0.0; p * p]; // row-major, p is tiny
        let mut t = vec![0.0; p];
        for a in 0..p {
            for b in 0..p {
                s[a * p + b] = (0..k).map(|i| con(i, a) * y[i + b * k]).sum();
            }
            t[a] = (0..k).map(|i| con(i, a) * z[i]).sum::<f64>() - rhs[k + a];
        }
        let Some(mu) = solve_small(&mut s, &mut t, p) else {
            return false;
        };
        for i in 0..k {
            x[i] = z[i] - (0..p).map(|q| y[i + q * k] * mu[q]).sum::<f64>();
        }
        x[k..].copy_from_slice(&mu);
    } else {
        x.copy_from_slice(z);
    }
    x.iter().all(|v| v.is_finite())
}

/// Gaussian elimination with partial pivoting for the tiny `(p, p)` Schur
/// system (row-major `s`). `None` if a pivot is (relatively) zero.
fn solve_small(s: &mut [f64], t: &mut [f64], p: usize) -> Option<Vec<f64>> {
    let scale = s.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if scale == 0.0 || !scale.is_finite() {
        return None;
    }
    for col in 0..p {
        let piv = (col..p)
            .max_by(|&a, &b| s[a * p + col].abs().total_cmp(&s[b * p + col].abs()))
            .expect("non-empty range");
        if s[piv * p + col].abs() <= SCHUR_EPS * scale {
            return None;
        }
        if piv != col {
            for c in 0..p {
                s.swap(piv * p + c, col * p + c);
            }
            t.swap(piv, col);
        }
        for row in col + 1..p {
            let f = s[row * p + col] / s[col * p + col];
            for c in col..p {
                s[row * p + c] -= f * s[col * p + c];
            }
            t[row] -= f * t[col];
        }
    }
    let mut mu = vec![0.0; p];
    for row in (0..p).rev() {
        let acc: f64 = (row + 1..p).map(|c| s[row * p + c] * mu[c]).sum();
        mu[row] = (t[row] - acc) / s[row * p + row];
    }
    Some(mu)
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
