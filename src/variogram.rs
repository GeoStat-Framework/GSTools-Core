use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Zip};

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
    fn dist(dim: usize, pos: ArrayView2<f64>, i: usize, j: usize) -> f64;
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
    fn dist(dim: usize, pos: ArrayView2<f64>, i: usize, j: usize) -> f64 {
        (0..dim)
            .map(|d| (pos[[d, i]] - pos[[d, j]]).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

struct Haversine;

impl Distance for Haversine {
    fn dist(dim: usize, pos: ArrayView2<f64>, i: usize, j: usize) -> f64 {
        assert!(dim == 2, "Haversine: dim = {} != 2", dim);

        let diff_lat = (pos[[0, j]] - pos[[0, i]]).to_radians();
        let diff_lon = (pos[[1, j]] - pos[[1, i]]).to_radians();

        let arg = (diff_lat / 2.0).sin().powi(2)
            + (pos[[0, i]].to_radians()).cos()
                * (pos[[0, j]].to_radians()).cos()
                * (diff_lon / 2.0).sin().powi(2);

        2.0 * arg.sqrt().atan2((1.0 - arg).sqrt())
    }
}

pub fn variogram_structured(f: ArrayView2<'_, f64>, estimator_type: char) -> Array1<f64> {
    fn inner<E: Estimator>(f: ArrayView2<'_, f64>) -> Array1<f64> {
        let i_max = f.shape()[0] - 1;
        let j_max = f.shape()[1];
        let k_max = i_max + 1;

        let mut variogram = Array1::<f64>::zeros(k_max);
        let mut counts = Array1::<u64>::zeros(k_max);

        (0..i_max).into_iter().for_each(|i| {
            (0..j_max).into_iter().for_each(|j| {
                (1..k_max - i).into_iter().for_each(|k| {
                    counts[k] += 1;
                    variogram[k] += E::estimate(f[[i, j]] - f[[i + k, j]]);
                });
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
        let i_max = f.shape()[0] - 1;
        let j_max = f.shape()[1];
        let k_max = i_max + 1;

        let mut variogram = Array1::<f64>::zeros(k_max);
        let mut counts = Array1::<u64>::zeros(k_max);

        (0..i_max).into_iter().for_each(|i| {
            (0..j_max).into_iter().for_each(|j| {
                (1..k_max - i).into_iter().for_each(|k| {
                    if !mask[[i, j]] && !mask[[i + k, j]] {
                        counts[k] += 1;
                        variogram[k] += E::estimate(f[[i, j]] - f[[i + k, j]]);
                    }
                });
            });
        });

        E::normalize(variogram.view_mut(), counts.view());

        variogram
    }

    choose_estimator!(estimator_type => E: {
        inner::<E>(f, mask)
    })
}

#[allow(clippy::too_many_arguments)]
fn dir_test(
    dim: usize,
    pos: ArrayView2<'_, f64>,
    dist: f64,
    direction: ArrayView2<'_, f64>,
    angles_tol: f64,
    bandwidth: f64,
    i: usize,
    j: usize,
    d: usize,
) -> bool {
    let mut s_prod = 0.0; //scalar product
    let mut b_dist = 0.0; //band-distance
    let mut in_band = true;
    let mut in_angle = true;

    //scalar-product calculation for bandwidth projection and angle calculation
    for k in 0..dim {
        s_prod += (pos[[k, i]] - pos[[k, j]]) * direction[[d, k]];
    }

    //calculate band-distance by projection of point-pair-vec to direction line
    if bandwidth > 0.0 {
        for k in 0..dim {
            b_dist += ((pos[[k, i]] - pos[[k, j]]) - s_prod * direction[[d, k]]).powi(2);
        }
        in_band = b_dist.sqrt() < bandwidth;
    }

    //allow repeating points (dist = 0)
    if dist > 0.0 {
        //use smallest angle by taking absolute value for arccos angle formula
        let angle = s_prod.abs() / dist;
        if angle < 1.0 {
            //else same direction (prevent numerical errors)
            in_angle = angle.acos() < angles_tol;
        }
    }

    in_band && in_angle
}

#[allow(clippy::too_many_arguments)]
pub fn variogram_directional(
    dim: usize,
    f: ArrayView2<'_, f64>,
    bin_edges: ArrayView1<'_, f64>,
    pos: ArrayView2<'_, f64>,
    direction: ArrayView2<'_, f64>, // should be normed
    angles_tol: f64,
    bandwidth: f64,
    separate_dirs: bool,
    estimator_type: char,
) -> (Array2<f64>, Array2<u64>) {
    assert!(
        pos.shape()[1] == f.shape()[1],
        "len(pos) = {} != len(f) = {}",
        pos.shape()[1],
        f.shape()[1],
    );
    assert!(
        bin_edges.shape()[0] > 1,
        "len(bin_edges) = {} < 2 too small",
        bin_edges.shape()[0]
    );
    assert!(
        angles_tol > 0.0,
        "tolerance for angle search masks must be > 0",
    );

    fn inner<E: Estimator>(
        dim: usize,
        f: ArrayView2<'_, f64>,
        bin_edges: ArrayView1<'_, f64>,
        pos: ArrayView2<'_, f64>,
        direction: ArrayView2<'_, f64>,
        angles_tol: f64,
        bandwidth: f64,
        separate_dirs: bool,
    ) -> (Array2<f64>, Array2<u64>) {
        let d_max = direction.shape()[0];
        let i_max = bin_edges.shape()[0] - 1;
        let j_max = pos.shape()[1] - 1;
        let k_max = pos.shape()[1];
        let f_max = f.shape()[0];

        let mut variogram = Array2::<f64>::zeros((d_max, i_max));
        let mut counts = Array2::<u64>::zeros((d_max, i_max));

        for i in 0..i_max {
            for j in 0..j_max {
                for k in j + 1..k_max {
                    let dist = Euclid::dist(dim, pos, j, k);
                    if dist < bin_edges[i] || dist >= bin_edges[i + 1] {
                        continue; //skip if not in current bin
                    }
                    for d in 0..d_max {
                        if !dir_test(dim, pos, dist, direction, angles_tol, bandwidth, k, j, d) {
                            continue;
                        }
                        for m in 0..f_max {
                            if !(f[[m, k]].is_nan() || f[[m, j]].is_nan()) {
                                counts[[d, i]] += 1;
                                variogram[[d, i]] += E::estimate(f[[m, k]] - f[[m, j]]);
                            }
                        }
                        //once we found a fitting direction
                        //break the search if directions are separated
                        if separate_dirs {
                            break;
                        }
                    }
                }
            }
        }

        E::normalize_vec(variogram.view_mut(), counts.view());

        (variogram, counts)
    }

    choose_estimator!(estimator_type => E: {
        inner::<E>(
            dim,
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
    dim: usize,
    f: ArrayView2<'_, f64>,
    bin_edges: ArrayView1<'_, f64>,
    pos: ArrayView2<'_, f64>,
    estimator_type: char,
    distance_type: char,
) -> (Array1<f64>, Array1<u64>) {
    assert!(
        pos.shape()[1] == f.shape()[1],
        "len(pos) = {} != len(f) = {}",
        pos.shape()[1],
        f.shape()[1],
    );
    assert!(
        bin_edges.shape()[0] > 1,
        "len(bin_edges) = {} < 2 too small",
        bin_edges.shape()[0]
    );

    fn inner<E: Estimator, D: Distance>(
        dim: usize,
        f: ArrayView2<'_, f64>,
        bin_edges: ArrayView1<'_, f64>,
        pos: ArrayView2<'_, f64>,
    ) -> (Array1<f64>, Array1<u64>) {
        let i_max = bin_edges.shape()[0] - 1;
        let j_max = pos.shape()[1] - 1;
        let k_max = pos.shape()[1];
        let f_max = f.shape()[0];

        let mut variogram = Array1::<f64>::zeros(i_max);
        let mut counts = Array1::<u64>::zeros(i_max);

        for i in 0..i_max {
            for j in 0..j_max {
                for k in j + 1..k_max {
                    let dist = D::dist(dim, pos, j, k);
                    if dist < bin_edges[i] || dist >= bin_edges[i + 1] {
                        continue; //skip if not in current bin
                    }
                    for m in 0..f_max {
                        // skip no data values
                        if !(f[[m, k]].is_nan() || f[[m, j]].is_nan()) {
                            counts[i] += 1;
                            variogram[i] += E::estimate(f[[m, k]] - f[[m, j]]);
                        }
                    }
                }
            }
        }

        E::normalize(variogram.view_mut(), counts.view());

        (variogram, counts)
    }

    choose_estimator!(estimator_type => E: {
        choose_distance!(distance_type => D: {
            inner::<E, D>(dim, f, bin_edges, pos)
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
            2,
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
            2,
            setup.field.view(),
            setup.bin_edges.view(),
            setup.pos.view(),
            'm',
            'e',
        );
        let (gamma2, _) = variogram_unstructured(
            2,
            field2.view(),
            setup.bin_edges.view(),
            setup.pos.view(),
            'm',
            'e',
        );
        let (gamma_multi, _) = variogram_unstructured(
            2,
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
            2,
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
            2,
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
            2,
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
            2,
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
