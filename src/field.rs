use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};

use rayon::prelude::*;

pub fn summator(
    cov_samples: ArrayView2<'_, f64>,
    z1: ArrayView1<'_, f64>,
    z2: ArrayView1<'_, f64>,
    pos: ArrayView2<'_, f64>,
) -> Array1<f64> {
    assert!(cov_samples.shape()[0] == pos.shape()[0]);
    assert!(cov_samples.shape()[1] == z1.shape()[0]);
    assert!(z1.shape()[0] == z2.shape()[0]);

    let mut summed_modes = Array1::<f64>::zeros(pos.shape()[1]);

    Zip::from(&mut summed_modes)
        .and(pos.columns())
        .par_for_each(|sum, pos| {
            Zip::from(cov_samples.columns())
                .and(z1)
                .and(z2)
                .for_each(|sample, &z1, &z2| {
                    let mut phase = 0.0;
                    Zip::from(sample).and(pos).for_each(|&s, &p| {
                        phase += s * p;
                    });
                    *sum += z1 * phase.cos() + z2 * phase.sin();
                })
        });
    summed_modes
}

pub fn summator_incompr(
    cov_samples: ArrayView2<'_, f64>,
    z1: ArrayView1<'_, f64>,
    z2: ArrayView1<'_, f64>,
    pos: ArrayView2<'_, f64>,
) -> Array2<f64> {
    assert!(cov_samples.shape()[0] == pos.shape()[0]);
    assert!(cov_samples.shape()[1] == z1.shape()[0]);
    assert!(z1.shape()[0] == z2.shape()[0]);

    let dim = pos.shape()[0];

    let mut summed_modes = Array2::<f64>::zeros(pos.dim());

    // unit vector in x dir.
    let mut e1 = Array1::<f64>::zeros(dim);
    e1[0] = 1.0;

    Zip::from(pos.columns())
        .and(summed_modes.columns_mut())
        .par_for_each(|pos, mut summed_modes| {
            let sum = Zip::from(cov_samples.columns())
                .and(z1)
                .and(z2)
                .into_par_iter()
                .fold(
                    || Array1::<f64>::zeros(dim),
                    |mut sum, (cov_samples, z1, z2)| {
                        let k_2 = cov_samples.dot(&cov_samples);
                        let phase = cov_samples.dot(&pos);

                        Zip::from(&mut sum).and(&e1).and(cov_samples).par_for_each(
                            |sum, e1, cs| {
                                let proj = *e1 - cs * cov_samples[0] / k_2;
                                *sum += proj * (z1 * phase.cos() + z2 * phase.sin());
                            },
                        );

                        sum
                    },
                )
                .reduce(
                    || Array1::<f64>::zeros(dim),
                    |mut lhs, rhs| {
                        lhs += &rhs;
                        lhs
                    },
                );

            summed_modes.assign(&sum);
        });

    summed_modes
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    struct Setup {
        cov_samples: Array2<f64>,
        z_1: Array1<f64>,
        z_2: Array1<f64>,
        pos: Array2<f64>,
    }

    impl Setup {
        fn new() -> Self {
            Self {
                cov_samples: arr2(&[
                    [
                        -2.15, 1.04, 0.69, -1.09, -1.54, -2.32, -1.81, -2.78, 1.57, -3.44,
                    ],
                    [
                        0.19, -1.24, -2.10, -2.86, -0.63, -0.51, -1.68, -0.07, 0.29, -0.007,
                    ],
                    [
                        0.98, -2.83, -0.10, 3.23, 0.51, 0.13, -1.03, 1.53, -0.51, 2.82,
                    ],
                ]),
                z_1: arr1(&[
                    -1.93, 0.46, 0.66, 0.02, -0.10, 1.29, 0.93, -1.14, 1.81, 1.47,
                ]),
                z_2: arr1(&[
                    -0.26, 0.98, -1.30, 0.66, 0.57, -0.25, -0.31, -0.29, 0.69, 1.14,
                ]),
                pos: arr2(&[
                    [0.00, 1.43, 2.86, 4.29, 5.71, 7.14, 9.57, 10.00],
                    [-5.00, -3.57, -2.14, -0.71, 0.71, 2.14, 3.57, 5.00],
                    [-6.00, -4.00, -2.00, 0.00, 2.00, 4.00, 6.00, 8.00],
                ]),
            }
        }
    }

    #[test]
    fn test_summate_3d() {
        let setup = Setup::new();

        assert_eq!(
            summator(
                setup.cov_samples.view(),
                setup.z_1.view(),
                setup.z_2.view(),
                setup.pos.view()
            ),
            arr1(&[
                0.3773130601113641,
                -4.298994445846448,
                0.9285578931297425,
                0.893013192171638,
                -1.4956409956178418,
                -1.488542499264307,
                0.19211668257573278,
                2.3427520079106143
            ])
        );
    }

    #[test]
    fn test_summate_incompr_3d() {
        let setup = Setup::new();

        assert_eq!(
            summator_incompr(
                setup.cov_samples.view(),
                setup.z_1.view(),
                setup.z_2.view(),
                setup.pos.view()
            ),
            arr2(&[
                [
                    0.7026540940472319,
                    -1.9323916721330978,
                    -0.4166102970790725,
                    0.27803989953742114,
                    -2.0809691290114567,
                    0.20148641078244162,
                    0.7758364517737109,
                    0.12811415623445488
                ],
                [
                    0.3498241912898348,
                    -0.07775049450238455,
                    -0.5970579726508763,
                    0.03011066817308309,
                    -0.6406632397415202,
                    0.4669548537557405,
                    0.908893008714896,
                    -0.5120295866263118
                ],
                [
                    0.2838955719581232,
                    -0.9042103150526011,
                    -0.6494289973178196,
                    -0.5654019280252776,
                    -0.8386683161758316,
                    -0.4648269322196026,
                    -0.0656185245433833,
                    1.6593799470196355
                ]
            ])
        );
    }
}
