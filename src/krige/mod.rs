//! Provides different kriging functionality

use ndarray::{Array1, ArrayView1, ArrayView2, Zip};
use rayon::prelude::*;

pub mod methods;

/// Calculate the interpolated field and also return the variance.
///
/// # Arguments
///
/// * `krige_mat` - the kriging matrix
///   <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = (no. of kriging conditions `cond.dim()`, returned field `field.dim().0`)
/// * `krige_vecs` - the right hand side of the kriging system
///   <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = (`krig_mat.dim().0`, returned field `field.dim().0`)
/// * `cond` - the kriging conditions
///   <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = `krig_mat.dim().0`
/// * `num_threads` - the number of parallel threads used, if None, use rayon's default
///
/// # Returns
///
/// * `field` - the kriging field
///   <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = `krig_mat.dim().1`
/// * `error` - the error variance
///   <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = `krig_mat.dim().1`
pub fn calculator_field_krige_and_variance(
    krig_mat: ArrayView2<'_, f64>,
    krig_vecs: ArrayView2<'_, f64>,
    cond: ArrayView1<'_, f64>,
    num_threads: Option<usize>,
) -> (Array1<f64>, Array1<f64>) {
    assert_eq!(krig_mat.dim().0, krig_mat.dim().1);
    assert_eq!(krig_mat.dim().0, krig_vecs.dim().0);
    assert_eq!(krig_mat.dim().0, cond.dim());

    let mut field = Array1::<f64>::zeros(krig_vecs.shape()[1]);
    let mut error = Array1::<f64>::zeros(krig_vecs.shape()[1]);

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads.unwrap_or(rayon::current_num_threads()))
        .build()
        .unwrap()
        .install(|| {
            Zip::from(field.view_mut())
                .and(error.view_mut())
                .and(krig_vecs.columns())
                .par_for_each(|f, e, v_col| {
                    let acc = Zip::from(cond)
                        .and(v_col)
                        .and(krig_mat.columns())
                        .into_par_iter()
                        .fold(
                            || (0.0, 0.0),
                            |mut acc, (c, v, m_row)| {
                                let krig_fac = m_row.dot(&v_col);
                                acc.0 += c * krig_fac;
                                acc.1 += v * krig_fac;
                                acc
                            },
                        )
                        .reduce(
                            || (0.0, 0.0),
                            |mut lhs, rhs| {
                                lhs.0 += rhs.0;
                                lhs.1 += rhs.1;
                                lhs
                            },
                        );

                    *f = acc.0;
                    *e = acc.1;
                })
        });

    (field, error)
}

/// Calculate the interpolated field.
///
/// # Arguments
///
/// * `krige_mat` - the kriging matrix
///   <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = (no. of kriging conditions `cond.dim()`, returned field `field.dim().0`)
/// * `krige_vecs` - the right hand side of the kriging system
///   <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = (`krig_mat.dim().0`, returned field `field.dim().0`)
/// * `cond` - the kriging conditions
///   <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = `krig_mat.dim().0`
/// * `num_threads` - the number of parallel threads used, if None, use rayon's default
///
/// # Returns
///
/// * `field` - the kriging field
///   <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = `krig_mat.dim().1`
pub fn calculator_field_krige(
    krig_mat: ArrayView2<'_, f64>,
    krig_vecs: ArrayView2<'_, f64>,
    cond: ArrayView1<'_, f64>,
    num_threads: Option<usize>,
) -> Array1<f64> {
    assert_eq!(krig_mat.dim().0, krig_mat.dim().1);
    assert_eq!(krig_mat.dim().0, krig_vecs.dim().0);
    assert_eq!(krig_mat.dim().0, cond.dim());

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads.unwrap_or(rayon::current_num_threads()))
        .build()
        .unwrap()
        .install(|| {
            Zip::from(krig_vecs.columns()).par_map_collect(|v_col| {
                Zip::from(cond)
                    .and(krig_mat.columns())
                    .into_par_iter()
                    .map(|(c, m_row)| {
                        let krig_fac = m_row.dot(&v_col);
                        c * krig_fac
                    })
                    .sum()
            })
        })
}
