use ndarray::{Array1, ArrayView1, ArrayView2, Zip};

pub fn calculator_field_krige_and_variance(
    krig_mat: ArrayView2<'_, f64>,
    krig_vecs: ArrayView2<'_, f64>,
    cond: ArrayView1<'_, f64>,
) -> (Array1<f64>, Array1<f64>) {
    let mat_i = krig_mat.shape()[0];
    let res_i = krig_vecs.shape()[1];

    let mut field = Array1::<f64>::zeros(res_i);
    let mut error = Array1::<f64>::zeros(res_i);

    //TODO make parallel
    (0..res_i).into_iter().for_each(|k| {
        (0..mat_i).into_iter().for_each(|i| {
            let mut krig_fac = 0.0;
            Zip::from(krig_mat.rows())
                .and(krig_vecs.rows())
                .for_each(|mat_row, vec_row| {
                    krig_fac += mat_row[i] * vec_row[k];
                });
            error[k] += krig_vecs[[i, k]] * krig_fac;
            field[k] += cond[i] * krig_fac;
        });
    });

    (field, error)
}

pub fn calculator_field_krige(
    krig_mat: ArrayView2<'_, f64>,
    krig_vecs: ArrayView2<'_, f64>,
    cond: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let mat_i = krig_mat.shape()[0];
    let res_i = krig_vecs.shape()[1];

    let mut field = Array1::<f64>::zeros(res_i);

    //TODO make parallel
    (0..res_i).into_iter().for_each(|k| {
        (0..mat_i).into_iter().for_each(|i| {
            let mut krig_fac = 0.0;
            Zip::from(krig_mat.rows())
                .and(krig_vecs.rows())
                .for_each(|mat_row, vec_row| {
                    krig_fac += mat_row[i] * vec_row[k];
                });
            field[k] += cond[i] * krig_fac;
        });
    });

    field
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
                        5.000000000689811541e-01,
                        -5.872870953648344543e-06,
                        7.823258125662829818e-12,
                    ],
                    [
                        -5.872870953788274386e-06,
                        5.000000000701587677e-01,
                        -7.673701033943365310e-07,
                    ],
                    [
                        7.823313193346819870e-12,
                        -7.673701034102435980e-07,
                        5.000000000011780577e-01,
                    ],
                ]),
                krig_vecs: arr2(&[
                    [
                        3.006509708451656770e-01,
                        7.929586741442330193e-11,
                        7.341029930928096026e-02,
                        1.103710603049992320e-08,
                        2.001142560424424288e-01,
                        7.230181341593456233e-03,
                    ],
                    [
                        5.514165757366291183e-09,
                        4.796566682382054559e-09,
                        3.912479648530730811e-03,
                        3.598469421494711835e-11,
                        2.107205731143327863e-10,
                        4.836258462653173222e-04,
                    ],
                    [
                        7.087965985442062730e-13,
                        1.097000072864030285e-01,
                        2.463223590277016835e-05,
                        1.758899927454055565e-07,
                        3.056710839404138744e-17,
                        2.385137855995505371e-11,
                    ],
                ]),
                cond: arr1(&[
                    -1.277554071957236692e+00,
                    1.155540406556412325e+00,
                    8.473742358954585718e-01,
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
        );
        assert_ulps_eq!(
            kr_field,
            arr1(&[
                -0.1920509731784287,
                0.0464783853717513,
                -0.044622334284034795,
                0.00000006749263448642215,
                -0.12782974926973528,
                -0.004339094946251052
            ]),
            max_ulps = 6,
        );
        assert_ulps_eq!(
            kr_error,
            arr1(&[
                0.04519550314128615,
                0.00601704579933182,
                0.002702186700869098,
                0.000000000000015529554261899065,
                0.020022857738472007,
                0.0000262546670280075
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
                setup.cond.view()
            ),
            arr1(&[
                -0.1920509731784287,
                0.0464783853717513,
                -0.044622334284034795,
                0.00000006749263448642215,
                -0.12782974926973528,
                -0.004339094946251052
            ]),
            max_ulps = 6,
        );
    }
}
