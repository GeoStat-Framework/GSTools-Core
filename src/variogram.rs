//! Estimate empirical variograms.
//!
//! Calculate the empirical variogram according to
//! $$
//! \gamma(r_k) = \frac{1}{2N(r_k)} \sum_{i=1}^{N(r_k)}(f(x_i) - f(x_i^\prime))^2
//! $$
//! with
//! * $r_k \leq \lVert x_i - x_i^\prime \rVert < r_{k+1}$ being the bins
//! * $N(r_k)$ being the number of points in bin $r_k$
//!
//! If the estimator type 'c' for Cressie was chosen, the variogram is calculated by
//! $$
//! \gamma(r_k) = \frac{\frac{1}{2} \left( \frac{1}{N(r_k)} \sum_{i=1}^{N(r_k)}|f(x_i) - f(x_i^\prime)|^{0.5}\right)^4}{0.457 + 0.494 / N(r_k) + 0.045 / N^2(r_k)}
//! $$

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, FoldWhile, Zip};
use rayon::iter::{IntoParallelIterator, ParallelExtend, ParallelIterator};

trait Estimator {
    fn estimate(f_diff: f64) -> f64;
    fn normalize(v: &mut f64, c: u64);
}

macro_rules! choose_estimator {
    ( $estimator_type:expr => $estimator:ident: $code:block ) => {
        match $estimator_type {
            'c' => {
                type $estimator = Cressie;

                $code
            }
            _ => {
                type $estimator = Matheron;

                $code
            }
        }
    };
}

struct Matheron;

impl Estimator for Matheron {
    fn estimate(f_diff: f64) -> f64 {
        f_diff.powi(2)
    }

    fn normalize(v: &mut f64, c: u64) {
        let cf = if c == 0 { 1.0 } else { c as f64 };
        *v /= 2.0 * cf;
    }
}

struct Cressie;

impl Estimator for Cressie {
    fn estimate(f_diff: f64) -> f64 {
        f_diff.abs().sqrt()
    }

    fn normalize(v: &mut f64, c: u64) {
        let cf = if c == 0 { 1.0 } else { c as f64 };
        *v = 0.5 * (1. / cf * *v).powi(4) / (0.457 + 0.494 / cf + 0.045 / (cf * cf))
    }
}

trait Distance {
    fn dist(lhs: ArrayView1<f64>, rhs: ArrayView1<f64>) -> f64;

    fn check_dim(_dim: usize) {}
}

macro_rules! choose_distance {
    ( $distance_type:expr => $distance:ident: $code:block ) => {
        match $distance_type {
            'e' => {
                type $distance = Euclid;

                $code
            }
            _ => {
                type $distance = Haversine;

                $code
            }
        }
    };
}

struct Euclid;

impl Distance for Euclid {
    fn dist(lhs: ArrayView1<f64>, rhs: ArrayView1<f64>) -> f64 {
        Zip::from(lhs)
            .and(rhs)
            .fold(0.0, |mut acc, lhs, rhs| {
                acc += (lhs - rhs).powi(2);

                acc
            })
            .sqrt()
    }
}

struct Haversine;

impl Distance for Haversine {
    fn dist(lhs: ArrayView1<f64>, rhs: ArrayView1<f64>) -> f64 {
        let diff_lat = (lhs[0] - rhs[0]).to_radians();
        let diff_lon = (lhs[1] - rhs[1]).to_radians();

        let arg = (diff_lat / 2.0).sin().powi(2)
            + lhs[0].to_radians().cos()
                * rhs[0].to_radians().cos()
                * (diff_lon / 2.0).sin().powi(2);

        2.0 * arg.sqrt().atan2((1.0 - arg).sqrt())
    }

    fn check_dim(dim: usize) {
        assert_eq!(dim, 2, "Haversine: dim = {} != 2", dim);
    }
}

/// Variogram estimation on a structured grid.
///
/// Calculates the empirical variogram according to the equations shown in the [module documentation](crate::variogram).
///
/// # Arguments
///
/// * `f` - the spatially distributed data
/// * `estimator_type` - the estimator function, can be
///     * 'm' - Matheron, the standard method of moments by Matheron
///     * 'c' - Cressie, an estimator more robust to outliers
/// * `num_threads` - the number of parallel threads used, if None, use rayon's default
pub fn variogram_structured(
    f: ArrayView2<'_, f64>,
    estimator_type: char,
    num_threads: Option<usize>,
) -> Array1<f64> {
    fn inner<E: Estimator>(f: ArrayView2<'_, f64>, num_threads: Option<usize>) -> Array1<f64> {
        let size = f.dim().0;

        let mut variogram = Vec::with_capacity(size);

        variogram.push(0.0);

        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads.unwrap_or(rayon::current_num_threads()))
            .build()
            .unwrap()
            .install(|| {
                variogram.par_extend((1..size).into_par_iter().map(|k| {
                    let mut value = 0.0;
                    let mut count = 0;

                    Zip::from(f.slice(s![..size - k, ..]))
                        .and(f.slice(s![k.., ..]))
                        .for_each(|f_i, f_j| {
                            value += E::estimate(f_i - f_j);
                            count += 1;
                        });

                    E::normalize(&mut value, count);

                    value
                }))
            });

        Array1::from_vec(variogram)
    }

    choose_estimator!(estimator_type => E: {
        inner::<E>(f, num_threads)
    })
}

/// Variogram estimation of a masked field on a structured grid.
///
/// Calculates the empirical variogram according to the equations shown in the [module documentation](crate::variogram).
///
/// # Arguments
///
/// * `f` - the spatially distributed data
/// * `mask` - the mask for the data `f`
/// * `estimator_type` - the estimator function, can be
///     * 'm' - Matheron, the standard method of moments by Matheron
///     * 'c' - Cressie, an estimator more robust to outliers
/// * `num_threads` - the number of parallel threads used, if None, use rayon's default
pub fn variogram_ma_structured(
    f: ArrayView2<'_, f64>,
    mask: ArrayView2<'_, bool>,
    estimator_type: char,
    num_threads: Option<usize>,
) -> Array1<f64> {
    fn inner<E: Estimator>(
        f: ArrayView2<'_, f64>,
        mask: ArrayView2<'_, bool>,
        num_threads: Option<usize>,
    ) -> Array1<f64> {
        let size = f.dim().0;

        let mut variogram = Vec::with_capacity(size);

        variogram.push(0.0);

        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads.unwrap_or(rayon::current_num_threads()))
            .build()
            .unwrap()
            .install(|| {
                variogram.par_extend((1..size).into_par_iter().map(|k| {
                    let mut value = 0.0;
                    let mut count = 0;

                    Zip::from(f.slice(s![..size - k, ..]))
                        .and(f.slice(s![k.., ..]))
                        .and(mask.slice(s![..size - k, ..]))
                        .and(mask.slice(s![k.., ..]))
                        .for_each(|f_i, f_j, m_i, m_j| {
                            if *m_i || *m_j {
                                return;
                            }

                            value += E::estimate(f_i - f_j);
                            count += 1;
                        });

                    E::normalize(&mut value, count);

                    value
                }))
            });

        Array1::from_vec(variogram)
    }

    choose_estimator!(estimator_type => E: {
        inner::<E>(f, mask, num_threads)
    })
}

fn dir_test(
    dir: ArrayView1<'_, f64>,
    pos_i: ArrayView1<'_, f64>,
    pos_j: ArrayView1<'_, f64>,
    dist: f64,
    angles_tol: f64,
    bandwidth: f64,
) -> bool {
    //scalar-product calculation for bandwidth projection and angle calculation
    let s_prod = Zip::from(dir)
        .and(pos_i)
        .and(pos_j)
        .fold(0.0, |mut acc, dir, pos_i, pos_j| {
            acc += (pos_i - pos_j) * dir;

            acc
        });

    //calculate band-distance by projection of point-pair-vec to direction line
    if bandwidth > 0.0 {
        let b_dist = Zip::from(dir)
            .and(pos_i)
            .and(pos_j)
            .fold(0.0, |mut acc, dir, pos_i, pos_j| {
                acc += ((pos_i - pos_j) - s_prod * dir).powi(2);

                acc
            })
            .sqrt();

        if b_dist >= bandwidth {
            return false;
        }
    }

    //allow repeating points (dist = 0)
    if dist > 0.0 {
        //use smallest angle by taking absolute value for arccos angle formula
        let angle = s_prod.abs() / dist;
        if angle < 1.0 {
            //else same direction (prevent numerical errors)
            if angle.acos() >= angles_tol {
                return false;
            }
        }
    }

    true
}

/// Directional variogram estimation on an unstructured grid.
///
/// Calculates the empirical variogram according to the equations shown in the [module documentation](crate::variogram).
///
/// # Arguments
///
/// * `f` - the spatially distributed data
/// <br>&nbsp;&nbsp;&nbsp;&nbsp; dim = (no. of data fields, no. of spatial data points per field $i$)
/// * `bin_edges` - the bins of the variogram
/// <br>&nbsp;&nbsp;&nbsp;&nbsp; dim = number of bins j
/// * `pos` - the positions of the data `f`
/// <br>&nbsp;&nbsp;&nbsp;&nbsp; dim = (spatial dim. $d$, no. of spatial data points $i$)
/// * `direction` - directions in which the variogram will be estimated
/// <br>&nbsp;&nbsp;&nbsp;&nbsp; dim = (no. of directions, spatial dim. $d$)
/// * `angles_tol` - the tolerance of the angles
/// * `bandwidth` - bandwidth to cut off the angular tolerance
/// * `separate_dirs` - do the direction bands overlap?
/// * `estimator_type` - the estimator function, can be
///     * 'm' - Matheron, the standard method of moments by Matheron
///     * 'c' - Cressie, an estimator more robust to outliers
/// * `num_threads` - the number of parallel threads used, if None, use rayon's default
#[allow(clippy::too_many_arguments)]
pub fn variogram_directional(
    f: ArrayView2<'_, f64>,
    bin_edges: ArrayView1<'_, f64>,
    pos: ArrayView2<'_, f64>,
    direction: ArrayView2<'_, f64>, // should be normed
    angles_tol: f64,
    bandwidth: f64,
    separate_dirs: bool,
    estimator_type: char,
    num_threads: Option<usize>,
) -> (Array2<f64>, Array2<u64>) {
    assert_eq!(
        pos.dim().0,
        direction.dim().1,
        "dim(pos) = {} != dim(direction) = {}",
        pos.dim().0,
        direction.dim().1,
    );
    assert_eq!(
        pos.dim().1,
        f.dim().1,
        "len(pos) = {} != len(f) = {}",
        pos.dim().1,
        f.dim().1,
    );
    assert!(
        bin_edges.dim() > 1,
        "len(bin_edges) = {} < 2 too small",
        bin_edges.dim()
    );
    assert!(
        angles_tol > 0.0,
        "tolerance for angle search masks must be > 0",
    );

    fn inner<E: Estimator>(
        f: ArrayView2<'_, f64>,
        bin_edges: ArrayView1<'_, f64>,
        pos: ArrayView2<'_, f64>,
        direction: ArrayView2<'_, f64>,
        angles_tol: f64,
        bandwidth: f64,
        separate_dirs: bool,
        num_threads: Option<usize>,
    ) -> (Array2<f64>, Array2<u64>) {
        let out_size = (direction.dim().0, bin_edges.dim() - 1);
        let in_size = pos.dim().1 - 1;

        let mut variogram = Array2::<f64>::zeros(out_size);
        let mut counts = Array2::<u64>::zeros(out_size);

        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads.unwrap_or(rayon::current_num_threads()))
            .build()
            .unwrap()
            .install(|| {
                Zip::from(bin_edges.slice(s![..out_size.1]))
                    .and(bin_edges.slice(s![1..]))
                    .and(variogram.columns_mut())
                    .and(counts.columns_mut())
                    .par_for_each(
                        |lower_bin_edge, upper_bin_edge, mut variogram, mut counts| {
                            Zip::indexed(f.slice(s![.., ..in_size]).columns())
                                .and(pos.slice(s![.., ..in_size]).columns())
                                .for_each(|i, f_i, pos_i| {
                                    Zip::from(f.slice(s![.., i + 1..]).columns())
                                        .and(pos.slice(s![.., i + 1..]).columns())
                                        .for_each(|f_j, pos_j| {
                                            let dist = Euclid::dist(pos_i, pos_j);
                                            if dist < *lower_bin_edge || dist >= *upper_bin_edge {
                                                return; //skip if not in current bin
                                            }

                                            Zip::from(direction.rows())
                                                .and(&mut variogram)
                                                .and(&mut counts)
                                                .fold_while((), |(), dir, variogram, counts| {
                                                    if !dir_test(
                                                        dir, pos_i, pos_j, dist, angles_tol,
                                                        bandwidth,
                                                    ) {
                                                        return FoldWhile::Continue(());
                                                    }

                                                    Zip::from(f_i).and(f_j).for_each(|f_i, f_j| {
                                                        let f_ij = f_i - f_j;
                                                        if f_ij.is_nan() {
                                                            return; // skip no data values
                                                        }

                                                        *counts += 1;
                                                        *variogram += E::estimate(f_ij);
                                                    });

                                                    //once we found a fitting direction
                                                    //break the search if directions are separated
                                                    if separate_dirs {
                                                        return FoldWhile::Done(());
                                                    }

                                                    FoldWhile::Continue(())
                                                });
                                        });
                                });

                            Zip::from(variogram)
                                .and(counts)
                                .for_each(|variogram, counts| {
                                    E::normalize(variogram, *counts);
                                });
                        },
                    )
            });

        (variogram, counts)
    }

    choose_estimator!(estimator_type => E: {
        inner::<E>(
            f,
            bin_edges,
            pos,
            direction,
            angles_tol,
            bandwidth,
            separate_dirs,
            num_threads,
        )
    })
}

/// Variogram estimation on an unstructured grid.
///
/// Calculates the empirical variogram according to the equations shown in the [module documentation](crate::variogram).
///
/// # Arguments
///
/// * `f` - the spatially distributed data
/// <br> dim = (no. of data fields, no. of spatial data points per field $i$)
/// * `bin_edges` - the bins of the variogram
/// <br> dim = number of bins j
/// * `pos` - the positions of the data `f`
/// <br> dim = (spatial dim. $d$, no. of spatial data points $i$)
/// * `estimator_type` - the estimator function, can be
///     * 'm' - Matheron, the standard method of moments by Matheron
///     * 'c' - Cressie, an estimator more robust to outliers
/// * `distance_type` - the distance function, can be
///     * 'e' - Euclidean, the Euclidean distance
///     * 'h' - Haversine, the great-circle distance
/// * `num_threads` - the number of parallel threads used, if None, use rayon's default
pub fn variogram_unstructured(
    f: ArrayView2<'_, f64>,
    bin_edges: ArrayView1<'_, f64>,
    pos: ArrayView2<'_, f64>,
    estimator_type: char,
    distance_type: char,
    num_threads: Option<usize>,
) -> (Array1<f64>, Array1<u64>) {
    assert_eq!(
        pos.dim().1,
        f.dim().1,
        "len(pos) = {} != len(f) = {}",
        pos.dim().1,
        f.dim().1,
    );
    assert!(
        bin_edges.dim() > 1,
        "len(bin_edges) = {} < 2 too small",
        bin_edges.dim()
    );

    fn inner<E: Estimator, D: Distance>(
        f: ArrayView2<'_, f64>,
        bin_edges: ArrayView1<'_, f64>,
        pos: ArrayView2<'_, f64>,
        num_threads: Option<usize>,
    ) -> (Array1<f64>, Array1<u64>) {
        D::check_dim(pos.dim().0);

        let out_size = bin_edges.dim() - 1;
        let in_size = pos.dim().1 - 1;

        let mut variogram = Array1::<f64>::zeros(out_size);
        let mut counts = Array1::<u64>::zeros(out_size);

        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads.unwrap_or(rayon::current_num_threads()))
            .build()
            .unwrap()
            .install(|| {
                Zip::from(bin_edges.slice(s![..out_size]))
                    .and(bin_edges.slice(s![1..]))
                    .and(&mut variogram)
                    .and(&mut counts)
                    .par_for_each(|lower_bin_edge, upper_bin_edge, variogram, counts| {
                        Zip::indexed(f.slice(s![.., ..in_size]).columns())
                            .and(pos.slice(s![.., ..in_size]).columns())
                            .for_each(|i, f_i, pos_i| {
                                Zip::from(f.slice(s![.., i + 1..]).columns())
                                    .and(pos.slice(s![.., i + 1..]).columns())
                                    .for_each(|f_j, pos_j| {
                                        let dist = D::dist(pos_i, pos_j);
                                        if dist < *lower_bin_edge || dist >= *upper_bin_edge {
                                            return; //skip if not in current bin
                                        }

                                        Zip::from(f_i).and(f_j).for_each(|f_i, f_j| {
                                            let f_ij = f_i - f_j;
                                            if f_ij.is_nan() {
                                                return; // skip no data values
                                            }

                                            *counts += 1;
                                            *variogram += E::estimate(f_ij);
                                        });
                                    });
                            });

                        E::normalize(variogram, *counts);
                    })
            });

        (variogram, counts)
    }

    choose_estimator!(estimator_type => E: {
        choose_distance!(distance_type => D: {
            inner::<E, D>(f, bin_edges, pos, num_threads)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_ulps_eq;
    use ndarray::{arr1, arr2, stack, Axis};

    struct SetupStruct {
        field: Array2<f64>,
    }

    impl SetupStruct {
        fn new() -> Self {
            Self {
                field: arr2(&[
                    [41.2],
                    [40.2],
                    [39.7],
                    [39.2],
                    [40.1],
                    [38.3],
                    [39.1],
                    [40.0],
                    [41.1],
                    [40.3],
                ]),
            }
        }
    }

    #[test]
    fn test_variogram_struct() {
        let setup = SetupStruct::new();

        assert_ulps_eq!(
            variogram_structured(setup.field.view(), 'm', None),
            arr1(&[
                0.0,
                0.49166666666666814,
                0.7625000000000011,
                1.090714285714288,
                0.9016666666666685,
                1.3360000000000025,
                0.9524999999999989,
                0.4349999999999996,
                0.004999999999999788,
                0.40500000000000513
            ]),
            max_ulps = 6
        );
    }

    #[test]
    fn test_variogram_ma_struct() {
        let setup = SetupStruct::new();
        let mask1 = arr2(&[
            [false],
            [false],
            [false],
            [false],
            [false],
            [false],
            [false],
            [false],
            [false],
            [false],
        ]);
        let mask2 = arr2(&[
            [true],
            [false],
            [false],
            [false],
            [false],
            [false],
            [false],
            [false],
            [false],
            [false],
        ]);

        assert_ulps_eq!(
            variogram_ma_structured(setup.field.view(), mask1.view(), 'm', None),
            arr1(&[
                0.0,
                0.49166666666666814,
                0.7625000000000011,
                1.090714285714288,
                0.9016666666666685,
                1.3360000000000025,
                0.9524999999999989,
                0.4349999999999996,
                0.004999999999999788,
                0.40500000000000513
            ]),
            max_ulps = 6
        );
        assert_ulps_eq!(
            variogram_ma_structured(setup.field.view(), mask2.view(), 'm', None),
            arr1(&[
                0.0,
                0.4906250000000017,
                0.710714285714287,
                0.9391666666666693,
                0.9610000000000019,
                0.6187499999999992,
                0.5349999999999975,
                0.29249999999999765,
                0.004999999999999432,
                0.0
            ]),
            max_ulps = 6
        );
    }

    struct SetupUnstruct {
        pos: Array2<f64>,
        field: Array2<f64>,
        bin_edges: Array1<f64>,
    }

    impl SetupUnstruct {
        fn new() -> Self {
            Self {
                pos: stack![
                    Axis(0),
                    Array1::range(0., 10., 1.),
                    Array1::range(0., 10., 1.)
                ],
                field: arr2(&[[
                    -1.2427955,
                    -0.59811704,
                    -0.57745039,
                    0.01531904,
                    -0.26474262,
                    -0.53626347,
                    -0.85106795,
                    -1.96939178,
                    -1.83650493,
                    -1.23548617,
                ]]),
                bin_edges: Array1::linspace(0., 5., 4),
            }
        }
    }

    #[test]
    fn test_variogram_unstruct() {
        let setup = SetupUnstruct::new();
        let (gamma, cnts) = variogram_unstructured(
            setup.field.view(),
            setup.bin_edges.view(),
            setup.pos.view(),
            'm',
            'e',
            None,
        );
        assert_ulps_eq!(
            gamma,
            arr1(&[0.14712242466045536, 0.320522186616688, 0.5136105328106929]),
            max_ulps = 6,
        );
        assert_eq!(cnts, arr1(&[9, 8, 7]),);
    }

    #[test]
    fn test_variogram_unstruct_multi_field() {
        let setup = SetupUnstruct::new();
        let field2 = arr2(&[[
            1.2427955,
            1.59811704,
            1.57745039,
            -1.01531904,
            1.26474262,
            1.53626347,
            1.85106795,
            0.96939178,
            0.83650493,
            0.23548617,
        ]]);
        let field_multi = arr2(&[
            [
                -1.2427955,
                -0.59811704,
                -0.57745039,
                0.01531904,
                -0.26474262,
                -0.53626347,
                -0.85106795,
                -1.96939178,
                -1.83650493,
                -1.23548617,
            ],
            [
                1.2427955,
                1.59811704,
                1.57745039,
                -1.01531904,
                1.26474262,
                1.53626347,
                1.85106795,
                0.96939178,
                0.83650493,
                0.23548617,
            ],
        ]);
        let (gamma, _) = variogram_unstructured(
            setup.field.view(),
            setup.bin_edges.view(),
            setup.pos.view(),
            'm',
            'e',
            None,
        );
        let (gamma2, _) = variogram_unstructured(
            field2.view(),
            setup.bin_edges.view(),
            setup.pos.view(),
            'm',
            'e',
            None,
        );
        let (gamma_multi, _) = variogram_unstructured(
            field_multi.view(),
            setup.bin_edges.view(),
            setup.pos.view(),
            'm',
            'e',
            None,
        );
        let gamma_single = 0.5 * (&gamma + &gamma2);
        assert_ulps_eq!(gamma_multi, gamma_single, max_ulps = 6,);

        let direction = arr2(&[[0., std::f64::consts::PI], [0., 0.]]);
        let (gamma, _) = variogram_directional(
            setup.field.view(),
            setup.bin_edges.view(),
            setup.pos.view(),
            direction.view(),
            std::f64::consts::PI / 8.,
            -1.0,
            false,
            'm',
            None,
        );
        let (gamma2, _) = variogram_directional(
            field2.view(),
            setup.bin_edges.view(),
            setup.pos.view(),
            direction.view(),
            std::f64::consts::PI / 8.,
            -1.0,
            false,
            'm',
            None,
        );
        let (gamma_multi, _) = variogram_directional(
            field_multi.view(),
            setup.bin_edges.view(),
            setup.pos.view(),
            direction.view(),
            std::f64::consts::PI / 8.,
            -1.0,
            false,
            'm',
            None,
        );

        let gamma_single = 0.5 * (&gamma + &gamma2);
        assert_ulps_eq!(gamma_multi, gamma_single, max_ulps = 6,);
    }

    #[test]
    fn test_variogram_directional() {
        let setup = SetupUnstruct::new();
        let direction = arr2(&[[0., std::f64::consts::PI], [0., 0.]]);
        let (gamma, cnts) = variogram_directional(
            setup.field.view(),
            setup.bin_edges.view(),
            setup.pos.view(),
            direction.view(),
            std::f64::consts::PI / 8.,
            -1.0,
            false,
            'm',
            None,
        );
        assert_ulps_eq!(
            gamma,
            arr2(&[
                [0.14712242466045536, 0.320522186616688, 0.5136105328106929],
                [0., 0., 0.]
            ]),
            max_ulps = 6,
        );
        assert_eq!(cnts, arr2(&[[9, 8, 7], [0, 0, 0]]),);
    }
}
