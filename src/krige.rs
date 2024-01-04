//! Perform the kriging matrix operations.

use ndarray::{Array1, ArrayView1, ArrayView2, Zip};
use rayon::prelude::*;

/// Calculate the interpolated field and also return the variance.
///
/// # Arguments
///
/// * `krige_mat` - the kriging matrix
/// <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = (no. of kriging conditions `cond.dim()`, returned field `field.dim().0`)
/// * `krige_vecs` - the right hand side of the kriging system
/// <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = (`krig_mat.dim().0`, returned field `field.dim().0`)
/// * `cond` - the kriging conditions
/// <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = `krig_mat.dim().0`
/// * `num_threads` - the number of parallel threads used, if None, use rayon's default
///
/// # Returns
///
/// * `field` - the kriging field
/// <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = `krig_mat.dim().1`
/// * `error` - the error variance
/// <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = `krig_mat.dim().1`
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
/// <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = (no. of kriging conditions `cond.dim()`, returned field `field.dim().0`)
/// * `krige_vecs` - the right hand side of the kriging system
/// <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = (`krig_mat.dim().0`, returned field `field.dim().0`)
/// * `cond` - the kriging conditions
/// <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = `krig_mat.dim().0`
/// * `num_threads` - the number of parallel threads used, if None, use rayon's default
///
/// # Returns
///
/// * `field` - the kriging field
/// <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = `krig_mat.dim().1`
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

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_ulps_eq;
    use ndarray::{arr1, arr2, Array2};

    struct Setup {
        krig_mat: Array2<f64>,
        krig_vecs: Array2<f64>,
        cond: Array1<f64>,
    }

    impl Setup {
        fn new() -> Self {
            Self {
                krig_mat: arr2(&[
                    [
                        5.00000000068981e-01,
                        -5.87287095364834e-06,
                        7.82325812566282e-12,
                    ],
                    [
                        -5.87287095378827e-06,
                        5.00000000070158e-01,
                        -7.67370103394336e-07,
                    ],
                    [
                        7.82331319334681e-12,
                        -7.67370103410243e-07,
                        5.00000000001178e-01,
                    ],
                ]),
                krig_vecs: arr2(&[
                    [
                        3.00650970845165e-01,
                        7.92958674144233e-11,
                        7.34102993092809e-02,
                        1.10371060304999e-08,
                        2.00114256042442e-01,
                        7.23018134159345e-03,
                    ],
                    [
                        5.51416575736629e-09,
                        4.79656668238205e-09,
                        3.91247964853073e-03,
                        3.59846942149471e-11,
                        2.10720573114332e-10,
                        4.83625846265317e-04,
                    ],
                    [
                        7.08796598544206e-13,
                        1.09700007286403e-01,
                        2.46322359027701e-05,
                        1.75889992745405e-07,
                        3.05671083940413e-17,
                        2.38513785599550e-11,
                    ],
                ]),
                cond: arr1(&[
                    -1.27755407195723e+00,
                    1.15554040655641e+00,
                    8.47374235895458e-01,
                ]),
            }
        }
    }

    #[test]
    fn test_calculator_field_krige_and_variance() {
        let setup = Setup::new();

        let (kr_field, kr_error) = calculator_field_krige_and_variance(
            setup.krig_mat.view(),
            setup.krig_vecs.view(),
            setup.cond.view(),
            None,
        );
        assert_ulps_eq!(
            kr_field,
            arr1(&[
                -0.19205097317842723,
                0.04647838537175125,
                -0.04462233428403452,
                0.0000000674926344864219,
                -0.12782974926973434,
                -0.0043390949462510245
            ]),
            max_ulps = 6,
        );
        assert_ulps_eq!(
            kr_error,
            arr1(&[
                0.04519550314128594,
                0.006017045799331816,
                0.0027021867008690937,
                0.000000000000015529554261898964,
                0.020022857738471924,
                0.00002625466702800745
            ]),
            max_ulps = 6,
        );
    }

    #[test]
    fn test_calculator_field_krige() {
        let setup = Setup::new();

        assert_ulps_eq!(
            calculator_field_krige(
                setup.krig_mat.view(),
                setup.krig_vecs.view(),
                setup.cond.view(),
                None
            ),
            arr1(&[
                -0.19205097317842723,
                0.04647838537175125,
                -0.04462233428403452,
                0.0000000674926344864219,
                -0.12782974926973434,
                -0.0043390949462510245
            ]),
            max_ulps = 6,
        );
    }
}
