use ndarray::{
    s, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, FoldWhile, Zip,
};

trait Estimator {
    fn estimate(f_diff: f64) -> f64;
    fn normalize(variogram: ArrayViewMut1<f64>, counts: ArrayView1<u64>);

    fn normalize_vec(mut variogram: ArrayViewMut2<f64>, counts: ArrayView2<u64>) {
        Zip::from(variogram.rows_mut())
            .and(counts.rows())
            .par_for_each(|variogram, counts| {
                Self::normalize(variogram, counts);
            });
    }
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

    fn normalize(variogram: ArrayViewMut1<f64>, counts: ArrayView1<u64>) {
        Zip::from(variogram).and(counts).for_each(|v, c| {
            let cf = if *c == 0 { 1.0 } else { *c as f64 };
            *v /= 2.0 * cf;
        });
    }
}

struct Cressie;

impl Estimator for Cressie {
    fn estimate(f_diff: f64) -> f64 {
        f_diff.abs().sqrt()
    }

    fn normalize(variogram: ArrayViewMut1<f64>, counts: ArrayView1<u64>) {
        Zip::from(variogram).and(counts).for_each(|v, c| {
            let cf = if *c == 0 { 1.0 } else { *c as f64 };
            *v = 0.5 * (1. / cf * *v).powi(4) / (0.457 + 0.494 / cf + 0.045 / (cf * cf))
        });
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

pub fn variogram_structured(f: ArrayView2<'_, f64>, estimator_type: char) -> Array1<f64> {
    fn inner<E: Estimator>(f: ArrayView2<'_, f64>) -> Array1<f64> {
        let size = f.dim().0;

        let mut variogram = Array1::<f64>::zeros(size);
        let mut counts = Array1::<u64>::zeros(size);

        Zip::indexed(variogram.slice_mut(s![1..]))
            .and(counts.slice_mut(s![1..]))
            .par_for_each(|k, variogram, counts| {
                Zip::from(f.slice(s![..size - k - 1, ..]))
                    .and(f.slice(s![k + 1.., ..]))
                    .for_each(|f_i, f_j| {
                        *counts += 1;
                        *variogram += E::estimate(f_i - f_j);
                    });
            });

        E::normalize(variogram.view_mut(), counts.view());

        variogram
    }

    choose_estimator!(estimator_type => E: {
        inner::<E>(f)
    })
}

pub fn variogram_ma_structured(
    f: ArrayView2<'_, f64>,
    mask: ArrayView2<'_, bool>,
    estimator_type: char,
) -> Array1<f64> {
    fn inner<E: Estimator>(f: ArrayView2<'_, f64>, mask: ArrayView2<'_, bool>) -> Array1<f64> {
        let size = f.dim().0;

        let mut variogram = Array1::<f64>::zeros(size);
        let mut counts = Array1::<u64>::zeros(size);

        Zip::indexed(variogram.slice_mut(s![1..]))
            .and(counts.slice_mut(s![1..]))
            .par_for_each(|k, variogram, counts| {
                Zip::from(f.slice(s![..size - k - 1, ..]))
                    .and(f.slice(s![k + 1.., ..]))
                    .and(mask.slice(s![..size - k - 1, ..]))
                    .and(mask.slice(s![k + 1.., ..]))
                    .for_each(|f_i, f_j, m_i, m_j| {
                        if *m_i || *m_j {
                            return;
                        }

                        *counts += 1;
                        *variogram += E::estimate(f_i - f_j);
                    });
            });

        E::normalize(variogram.view_mut(), counts.view());

        variogram
    }

    choose_estimator!(estimator_type => E: {
        inner::<E>(f, mask)
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
    ) -> (Array2<f64>, Array2<u64>) {
        let out_size = (direction.dim().0, bin_edges.dim() - 1);
        let in_size = pos.dim().1 - 1;

        let mut variogram = Array2::<f64>::zeros(out_size);
        let mut counts = Array2::<u64>::zeros(out_size);

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
                                                dir, pos_i, pos_j, dist, angles_tol, bandwidth,
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
                },
            );

        E::normalize_vec(variogram.view_mut(), counts.view());

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
            separate_dirs
        )
    })
}

pub fn variogram_unstructured(
    f: ArrayView2<'_, f64>,
    bin_edges: ArrayView1<'_, f64>,
    pos: ArrayView2<'_, f64>,
    estimator_type: char,
    distance_type: char,
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
    ) -> (Array1<f64>, Array1<u64>) {
        D::check_dim(pos.dim().0);

        let out_size = bin_edges.dim() - 1;
        let in_size = pos.dim().1 - 1;

        let mut variogram = Array1::<f64>::zeros(out_size);
        let mut counts = Array1::<u64>::zeros(out_size);

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
            });

        E::normalize(variogram.view_mut(), counts.view());

        (variogram, counts)
    }

    choose_estimator!(estimator_type => E: {
        choose_distance!(distance_type => D: {
            inner::<E, D>(f, bin_edges, pos)
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
            variogram_structured(setup.field.view(), 'm'),
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
            variogram_ma_structured(setup.field.view(), mask1.view(), 'm'),
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
            variogram_ma_structured(setup.field.view(), mask2.view(), 'm'),
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
        );
        let (gamma2, _) = variogram_unstructured(
            field2.view(),
            setup.bin_edges.view(),
            setup.pos.view(),
            'm',
            'e',
        );
        let (gamma_multi, _) = variogram_unstructured(
            field_multi.view(),
            setup.bin_edges.view(),
            setup.pos.view(),
            'm',
            'e',
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
