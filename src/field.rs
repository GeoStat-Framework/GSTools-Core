//! Compute the randomization methods for random field generations.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip};
use rayon::prelude::*;

use crate::short_vec::ShortVec;

/// The randomization method for scalar fields.
///
/// Computes the isotropic spatial random field $u(x)$ by the randomization method according to
/// $$
/// u(x) = \sum_{i=1}^N (z_{1,i} \cdot \cos(\langle k_i, x \rangle) +
///     z_{2,i} \cdot \sin(\langle k_i, x \rangle))
/// $$
/// with
/// * $N$ being the number of Fourier modes
/// * $z_1, z_2$ being independent samples from a standard normal distribution
/// * $k$ being the samples from the spectral density distribution of the covariance model
///   and are given by the argument `cov_samples`.
///
/// # Arguments
///
/// * `cov_samples` - the samples from the spectral density distribution of the covariance model
///   <br>&nbsp;&nbsp;&nbsp;&nbsp; dim = (spatial dim. of field $d$, Fourier modes $N$)
/// * `z1` - independent samples from a standard normal distribution
///   <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = Fourier modes $N$
/// * `z2` - independent samples from a standard normal distribution
///   <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = Fourier modes $N$
/// * `pos` - the position $x$ where the spatial random field is calculated
///   <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = (spatial dim. of field $d$, no. of spatial points where field is calculated $j$)
/// * `num_threads` - the number of parallel threads used, if None, use rayon's default
///
/// # Returns
///
/// * `summed_modes` - the isotropic spatial field
///   <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = no. of spatial points where field is calculated $j$
pub fn summator(
    cov_samples: ArrayView2<'_, f64>,
    z1: ArrayView1<'_, f64>,
    z2: ArrayView1<'_, f64>,
    pos: ArrayView2<'_, f64>,
    num_threads: Option<usize>,
) -> Array1<f64> {
    assert_eq!(cov_samples.dim().0, pos.dim().0);
    assert_eq!(cov_samples.dim().1, z1.dim());
    assert_eq!(cov_samples.dim().1, z2.dim());

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads.unwrap_or(rayon::current_num_threads()))
        .build()
        .unwrap()
        .install(|| {
            Zip::from(pos.columns()).par_map_collect(|pos| {
                Zip::from(cov_samples.columns()).and(z1).and(z2).fold(
                    0.0,
                    |sum, sample, &z1, &z2| {
                        let phase = sample.dot(&pos);
                        let z12 = z1 * phase.cos() + z2 * phase.sin();

                        sum + z12
                    },
                )
            })
        })
}

/// The randomization method for vector fields.
///
/// Computes the isotropic incompressible spatial random field $u(x)$ by the randomization method according to
/// $$
/// u(x)\_i = \sum_{j=1}^N p_i(k_j) (z_{1,j} \cdot \cos(\langle k_j, x \rangle) +
///     z_{2,j} \cdot \sin(\langle k_j, x \rangle))
/// $$
/// with
/// * $N$ being the number of Fourier modes
/// * $z_1, z_2$ being independent samples from a standard normal distribution
/// * $k$ being the samples from the spectral density distribution of the covariance model
///   and are given by the argument `cov_samples`.
/// * $p_i(k_j) = e_1 - \frac{k_ik_1}{|k|^2}$ being the projector ensuring the incompressibility
///
/// # Arguments
///
/// * `cov_samples` - the samples from the spectral density distribution of the covariance model
///   <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = (spatial dim. of field $d$, Fourier modes $N$)
/// * `z1` - independent samples from a standard normal distribution
///   <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = Fourier modes $N$
/// * `z2` - independent samples from a standard normal distribution
///   <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = Fourier modes $N$
/// * `pos` - the position $x$ where the spatial random field is calculated
///   <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = (spatial dim. of field $d$, no. of spatial points where field is calculated $j$)
/// * `num_threads` - the number of parallel threads used, if None, use rayon's default
///
/// # Returns
///
/// * `summed_modes` - the isotropic incompressible spatial field
///   <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = (spatial dim. of field $d$, no. of spatial points where field is calculated $j$)
pub fn summator_incompr(
    cov_samples: ArrayView2<'_, f64>,
    z1: ArrayView1<'_, f64>,
    z2: ArrayView1<'_, f64>,
    pos: ArrayView2<'_, f64>,
    num_threads: Option<usize>,
) -> Array2<f64> {
    assert_eq!(cov_samples.dim().0, pos.dim().0);
    assert_eq!(cov_samples.dim().1, z1.dim());
    assert_eq!(cov_samples.dim().1, z2.dim());

    fn inner<const N: usize>(
        cov_samples: ArrayView2<'_, f64>,
        z1: ArrayView1<'_, f64>,
        z2: ArrayView1<'_, f64>,
        pos: ArrayView2<'_, f64>,
        num_threads: Option<usize>,
    ) -> Array2<f64> {
        let cov_samples = cov_samples
            .axis_iter(Axis(1))
            .map(ShortVec::<N>::from_array)
            .collect::<Vec<_>>();

        let pos = pos
            .axis_iter(Axis(1))
            .map(ShortVec::<N>::from_array)
            .collect::<Vec<_>>();

        let summed_modes = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads.unwrap_or(rayon::current_num_threads()))
            .build()
            .unwrap()
            .install(|| {
                cov_samples
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
                    .unwrap()
            });

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
        2 => inner::<2>(cov_samples, z1, z2, pos, num_threads),
        3 => inner::<3>(cov_samples, z1, z2, pos, num_threads),
        _ => panic!("Only two- and three-dimensional problems are supported."),
    }
}

/// The Fourier method for scalar fields.
///
/// Computes the periodic, isotropic spatial random field $u(x)$ by the Fourier
/// method according to
/// $$
/// u(x) = \sum_{i=1}^N \sqrt{2S(k_i\Delta k}\left(
/// z_{1,i} \cdot \cos(\langle k_i, x \rangle) +
/// z_{2,i} \cdot \sin(\langle k_i, x \rangle)\right)
/// $$
/// with
/// * $S$ being the spectrum of the covariance model
/// * $N$ being the number of Fourier modes
/// * $z_1, z_2$ being independent samples from a standard normal distribution
/// * $k$ being the equidistant Fourier grid
///   and are given by the argument `modes`.
/// * $\Delta k$ being the cell size of the Fourier grid
///
/// # Arguments
///
/// * `spectrum_factor` - the pre-calculated factor $\sqrt{2S(k_i\Delta k)}
///   <br>&nbsp;&nbsp;&nbsp;&nbsp; dim = Fourier modes $N$
/// * `modes` - equidistant Fourier grid, $k$ in Eq. above
///   <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = (spatial dim. of field $d$, Fourier modes $N$)
/// * `z1` - independent samples from a standard normal distribution
///   <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = Fourier modes $N$
/// * `z2` - independent samples from a standard normal distribution
///   <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = Fourier modes $N$
/// * `pos` - the position $x$ where the spatial random field is calculated
///   <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = (spatial dim. of field $d$, no. of spatial points where field is calculated $j$)
/// * `num_threads` - the number of parallel threads used, if None, use rayon's default
///
/// # Returns
///
/// * `summed_modes` - the isotropic spatial field
///   <br>&nbsp;&nbsp;&nbsp;&nbsp;dim = no. of spatial points where field is calculated $j$
pub fn summator_fourier(
    spectrum_factor: ArrayView1<'_, f64>,
    modes: ArrayView2<'_, f64>,
    z1: ArrayView1<'_, f64>,
    z2: ArrayView1<'_, f64>,
    pos: ArrayView2<'_, f64>,
    num_threads: Option<usize>,
) -> Array1<f64> {
    assert_eq!(modes.dim().0, pos.dim().0);
    assert_eq!(modes.dim().1, z1.dim());
    assert_eq!(modes.dim().1, z2.dim());

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads.unwrap_or(rayon::current_num_threads()))
        .build()
        .unwrap()
        .install(|| {
            Zip::from(pos.columns()).par_map_collect(|pos| {
                Zip::from(modes.columns())
                    .and(spectrum_factor)
                    .and(z1)
                    .and(z2)
                    .fold(0.0, |sum, k, &spectrum_factor, &z1, &z2| {
                        let phase = k.dot(&pos);
                        let z12 = spectrum_factor * (z1 * phase.cos() + z2 * phase.sin());

                        sum + z12
                    })
            })
        })
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

    struct SetupFourier {
        spectrum_factor: Array1<f64>,
        modes: Array2<f64>,
        z_1: Array1<f64>,
        z_2: Array1<f64>,
        pos: Array2<f64>,
    }

    impl SetupFourier {
        fn new() -> Self {
            Self {
                spectrum_factor: arr1(&[
                    -2.15, 1.04, 0.69, -1.09, -1.54, -2.32, -1.81, -2.78, 1.57, -3.44,
                ]),
                modes: arr2(&[
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
    fn test_summate_fourier_2d() {
        let setup = SetupFourier::new();
        assert_eq!(
            summator_fourier(
                setup.spectrum_factor.view(),
                setup.modes.view(),
                setup.z_1.view(),
                setup.z_2.view(),
                setup.pos.view(),
                None,
            ),
            arr1(&[
                1.0666558330143816,
                -3.5855143411414883,
                -2.70208228699285,
                9.808554698975039,
                0.01634921830347258,
                -2.2356422006860663,
                14.730786907708966,
                -2.851408419726332,
            ])
        );
    }

    #[test]
    fn test_summate_2d() {
        let setup = Setup::new();

        assert_eq!(
            summator(
                setup.cov_samples.view(),
                setup.z_1.view(),
                setup.z_2.view(),
                setup.pos.view(),
                None,
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
    fn test_summate_incompr_2d() {
        let setup = Setup::new();

        assert_ulps_eq!(
            summator_incompr(
                setup.cov_samples.view(),
                setup.z_1.view(),
                setup.z_2.view(),
                setup.pos.view(),
                None,
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
