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
    use ndarray::{arr1, arr2, stack, Array2, Axis};

    struct Setup {
        pos: Array2<f64>,
        cond_pos: Array2<f64>,
        cond_val: Array1<f64>,
    }

    impl Setup {
        fn new() -> Self {
            Self {
                pos: stack![
                    Axis(0),
                    Array1::linspace(0., 10., 100000),
                    Array1::linspace(-5., 5., 100000),
                    Array1::linspace(-6., 8., 100000)
                ],
                cond_pos: arr2(&[
                    [0.3, 1.9, 1.1, 3.3, 4.7],
                    [1.2, 0.6, 3.2, 4.4, 3.8],
                    [0.5, 1., 1.5, 2., 2.5],
                ]),
                cond_val: arr1(&[4.7, 3.8, 2.5, 1.74]),
            }
        }
    }

    #[test]
    fn test_calculator_field_krige_and_variance() {
        let setup = Setup::new();

        let (kr_field, kr_error) = calculator_field_krige_and_variance(
            setup.pos.view(),
            setup.cond_pos.view(),
            setup.cond_val.view(),
        );
        assert_ulps_eq!(
            kr_field,
            arr1(&[
                -67.99735997359974,
                -118.99687996879967,
                -197.19255992559926,
                -329.789919899199,
                -373.98943989439897
            ]),
            max_ulps = 6,
        );
        assert_ulps_eq!(
            kr_error,
            arr1(&[
                -17.999515995159953,
                -31.498985989859897,
                -144.9960319603196,
                -329.79117991179913,
                -373.98943989439897
            ]),
            max_ulps = 6,
        );
    }

    #[test]
    fn test_calculator_field_krige() {
        let setup = Setup::new();

        assert_ulps_eq!(
            calculator_field_krige(
                setup.pos.view(),
                setup.cond_pos.view(),
                setup.cond_val.view()
            ),
            arr1(&[
                -67.99735997359974,
                -118.99687996879967,
                -197.19255992559926,
                -329.789919899199,
                -373.98943989439897
            ]),
            max_ulps = 6,
        );
    }
}
