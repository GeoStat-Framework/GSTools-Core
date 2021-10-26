use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip};
use rayon::prelude::*;

use crate::short_vec::ShortVec;

pub fn summator(
    cov_samples: ArrayView2<'_, f64>,
    z1: ArrayView1<'_, f64>,
    z2: ArrayView1<'_, f64>,
    pos: ArrayView2<'_, f64>,
) -> Array1<f64> {
    assert_eq!(cov_samples.dim().0, pos.dim().0);
    assert_eq!(cov_samples.dim().1, z1.dim());
    assert_eq!(cov_samples.dim().1, z2.dim());

    let mut summed_modes = Array1::<f64>::zeros(pos.dim().1);

    Zip::from(&mut summed_modes)
        .and(pos.columns())
        .par_for_each(|sum, pos| {
            Zip::from(cov_samples.columns())
                .and(z1)
                .and(z2)
                .for_each(|sample, &z1, &z2| {
                    let phase = sample.dot(&pos);

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
    assert_eq!(cov_samples.dim().0, pos.dim().0);
    assert_eq!(cov_samples.dim().1, z1.dim());
    assert_eq!(cov_samples.dim().1, z2.dim());

    fn inner<const N: usize>(
        cov_samples: ArrayView2<'_, f64>,
        z1: ArrayView1<'_, f64>,
        z2: ArrayView1<'_, f64>,
        pos: ArrayView2<'_, f64>,
    ) -> Array2<f64> {
        let cov_samples = cov_samples
            .axis_iter(Axis(1))
            .map(ShortVec::<N>::from_array)
            .collect::<Vec<_>>();

        let pos = pos
            .axis_iter(Axis(1))
            .map(ShortVec::<N>::from_array)
            .collect::<Vec<_>>();

        let summed_modes = cov_samples
            .par_iter()
            .zip(z1.axis_iter(Axis(0)))
            .zip(z2.axis_iter(Axis(0)))
            .with_min_len(100)
            .fold(
                || vec![ShortVec::<N>::zero(); pos.len()],
                |mut summed_modes, ((cov_samples, z1), z2)| {
                    let k_2 = cov_samples[0] / cov_samples.dot(cov_samples);
                    let z1 = z1.into_scalar();
                    let z2 = z2.into_scalar();

                    pos.par_iter()
                        .zip(&mut summed_modes)
                        .for_each(|(pos, sum)| {
                            let phase = cov_samples.dot(pos);
                            let z12 = z1 * phase.cos() + z2 * phase.sin();

                            sum[0] += (1.0 - cov_samples[0] * k_2) * z12;

                            (1..N).for_each(|idx| {
                                sum[idx] -= cov_samples[idx] * k_2 * z12;
                            });
                        });

                    summed_modes
                },
            )
            .reduce_with(|mut lhs, rhs| {
                lhs.iter_mut().zip(&rhs).for_each(|(lhs, rhs)| lhs.add(rhs));

                lhs
            })
            .unwrap();

        Array2::<f64>::from_shape_vec(
            (pos.len(), N),
            summed_modes
                .into_iter()
                .flat_map(ShortVec::<N>::into_iter)
                .collect(),
        )
        .unwrap()
        .reversed_axes()
    }

    match pos.dim().0 {
        2 => inner::<2>(cov_samples, z1, z2, pos),
        3 => inner::<3>(cov_samples, z1, z2, pos),
        _ => panic!("Only two- and three-dimensional problems are supported."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_ulps_eq;
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

        assert_ulps_eq!(
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
            ]),
            max_ulps = 6
        );
    }
}
