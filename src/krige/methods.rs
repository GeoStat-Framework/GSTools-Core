//! Different kriging algorithms implemented here. So far, only
//! simple kriging is implemented.

use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Zip};
use ndarray_linalg::{norm::Norm, solve::Inverse};

use crate::calculator_field_krige_and_variance;
use crate::cov_model::{CovModel, Covariance};

/// Compute the L2 distance between each pair of the two input arrays.
///
///  # Arguments
///
///  * `a` - first input array
///  * `b` - second input array
///
///  # Returns
///
///  * `res` - the pairwise distance matrix
pub fn cdist(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Array2<f64> {
    let mut res = Array::<f64, _>::zeros((a.dim().1, b.dim().1));
    // TODO benchmark if parallelization helps here
    Zip::from(res.rows_mut())
        .and(a.columns())
        .for_each(|rr, a| {
            Zip::from(rr).and(b.columns()).for_each(|r, b| {
                *r = (&b - &a).norm_l2();
            })
        });
    res
}

/// Generate the kriging field with simple kriging.
pub fn simple<T>(
    pos: ArrayView2<'_, f64>,
    cov_model: &T,
    cond_pos: ArrayView2<'_, f64>,
    cond_val: ArrayView1<'_, f64>,
    mean: f64,
) -> (Array1<f64>, Array1<f64>)
where
    T: CovModel + Covariance,
{
    // TODO calculate pseudo inverse, instead of inverse
    let krig_mat = match cdist(cond_pos, cond_pos)
        .mapv(|d| cov_model.covariance(d))
        .inv()
    {
        Ok(inv) => inv,
        Err(error) => {
            panic!("{error:?}\nKrige matrix does not have an inverse. Pseudo inverse needs to be implemented.")
        }
    };

    // calculate the RHS of kriging system
    let krig_vecs = cdist(cond_pos, pos).mapv(|d| cov_model.covariance(d));
    let krige_cond = cond_val.mapv(|v| v - mean);
    let (field, error) = calculator_field_krige_and_variance(
        krig_mat.view(),
        krig_vecs.view(),
        krige_cond.view(),
        None,
    );
    let field = field.mapv(|f| f + mean);
    let error = error.mapv(|e| (cov_model.var() - e).max(0.0));
    (field, error)
}

#[cfg(test)]
mod tests {
    use crate::{
        cov_model::{BaseCovModel, Gaussian},
        krige::calculator_field_krige,
    };

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

    #[test]
    fn test_linalg_1d() {
        let a = arr2(&[[0.3, 1.9, 1.1, 3.3, 4.7]]);
        assert_ulps_eq!(
            cdist(a.view(), a.view()),
            arr2(&[
                [0.0, 1.6, 0.8, 3.0, 4.4],
                [1.6, 0.0, 0.8, 1.4, 2.8],
                [0.8, 0.8, 0.0, 2.2, 3.6],
                [3., 1.4, 2.2, 0.0, 1.4],
                [4.4, 2.8, 3.6, 1.4, 0.0],
            ]),
            max_ulps = 6,
        );
        let b = Array::linspace(0.0, 15.0, 4);
        let b = b.broadcast((1, b.len())).unwrap();
        assert_ulps_eq!(
            cdist(a.view(), b.view()),
            arr2(&[
                [0.3, 4.7, 9.7, 14.7],
                [1.9, 3.1, 8.1, 13.1],
                [1.1, 3.9, 8.9, 13.9],
                [3.3, 1.7, 6.7, 11.7],
                [4.7, 0.3, 5.3, 10.3],
            ]),
            max_ulps = 6,
        );
    }
    #[test]
    fn test_linalg_2d() {
        let a = arr2(&[[0.3, 1.9, 1.1, 3.3, 4.7], [1.2, 0.6, 3.2, 4.4, 3.8]]);
        assert_ulps_eq!(
            cdist(a.view(), a.view()),
            arr2(&[
                [
                    0.0,
                    1.708800749063506,
                    2.154065922853802,
                    4.386342439892262,
                    5.110772935672255
                ],
                [
                    1.708800749063506,
                    0.0,
                    2.7202941017470885,
                    4.049691346263318,
                    4.252058325093859
                ],
                [
                    2.154065922853802,
                    2.7202941017470885,
                    0.0,
                    2.5059928172283334,
                    3.6496575181789317
                ],
                [
                    4.386342439892262,
                    4.049691346263318,
                    2.5059928172283334,
                    0.0,
                    1.5231546211727822
                ],
                [
                    5.110772935672255,
                    4.252058325093859,
                    3.6496575181789317,
                    1.5231546211727822,
                    0.0
                ]
            ]),
            max_ulps = 6,
        );
        let b = arr2(&[[0., 0., 5., 5.], [-5., 0., 5., -5.]]);
        assert_ulps_eq!(
            cdist(a.view(), b.view()),
            arr2(&[
                [
                    6.2072538211353985,
                    1.2369316876852983,
                    6.04400529450463,
                    7.780102827083971
                ],
                [
                    5.913543776789007,
                    1.9924858845171274,
                    5.3823786563191565,
                    6.400781202322104
                ],
                [
                    8.27345151674922,
                    3.3837848631377265,
                    4.295346318982906,
                    9.080198235721507
                ],
                [
                    9.962429422585638,
                    5.5,
                    1.8027756377319946,
                    9.552486587271401
                ],
                [
                    9.976472322419385,
                    6.04400529450463,
                    1.2369316876852983,
                    8.805112151472008
                ]
            ]),
            max_ulps = 6,
        );
    }
    #[test]
    fn test_krige_1d() {
        let pos = Array::linspace(0.0, 15.0, 4);
        let pos = pos.broadcast((1, pos.len())).unwrap();

        let model = Gaussian {
            base: BaseCovModel {
                dim: 1,
                var: 0.5,
                len_scale: 2.0,
            },
            ..Default::default()
        };
        let cond_pos = arr2(&[[0.3, 1.9, 1.1, 3.3, 4.7]]);
        let cond_val = arr1(&[0.47, 0.56, 0.74, 1.47, 1.74]);
        let (field, error) = simple(pos.view(), &model, cond_pos.view(), cond_val.view(), 1.0);
        assert_ulps_eq!(
            field,
            arr1(&[
                0.18792146005726795,
                1.485600187143261,
                0.9892592457925125,
                0.9999999972832743
            ]),
            max_ulps = 6,
        );
        assert_ulps_eq!(
            error,
            arr1(&[
                0.0007866097958594831,
                0.0036412749665615807,
                0.499975563693493,
                0.5
            ]),
            max_ulps = 6,
        );
    }
    #[test]
    fn test_krige_2d() {
        let model = Gaussian {
            base: BaseCovModel {
                dim: 2,
                var: 0.8,
                len_scale: 3.0,
            },
            ..Default::default()
        };
        // TODO maybe use meshgrid?
        // https://github.com/rust-ndarray/ndarray/issues/1355
        let pos = arr2(&[
            [0., 0., 0., 5., 5., 5., 10., 10., 10., 15., 15., 15.],
            [-5., 0., 5., -5., 0., 5., -5., 0., 5., -5., 0., 5.],
        ]);
        let cond_pos = arr2(&[[0.3, 1.9, 1.1, 3.3, 4.7], [1.2, 0.6, 3.2, 4.4, 3.8]]);
        let cond_val = arr1(&[0.47, 0.56, 0.74, 1.47, 1.74]);
        let (field, error) = simple(pos.view(), &model, cond_pos.view(), cond_val.view(), 0.0);

        assert_ulps_eq!(
            field,
            arr1(&[
                0.009557358693436949,
                0.3277615223971525,
                0.3145520862286027,
                0.0021019598010470313,
                0.4966425910922256,
                1.5162561066738178,
                0.00015835319007678262,
                0.04079589391922381,
                0.12775437308455703,
                1.8320948846857e-7,
                4.501362280436092e-5,
                0.00014020995740099762
            ]),
            max_ulps = 6,
        );
        assert_ulps_eq!(
            error,
            arr1(&[
                0.797396133570957,
                0.10565469106661851,
                0.36630191151733443,
                0.7987743258653218,
                0.5324374361056077,
                0.1548205035388157,
                0.7999998997165715,
                0.7987431018679746,
                0.7901204798243383,
                0.7999999999999668,
                0.7999999980743688,
                0.7999999824714752
            ]),
            max_ulps = 6,
        );
    }
}
