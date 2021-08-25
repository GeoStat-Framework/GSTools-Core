use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};

fn choose_estimator_func(est_type: char) -> impl Fn(f64) -> f64 {
    let estimator_matheron = |f_diff: f64| f_diff.powi(2);
    let estimator_cressie = |f_diff: f64| f_diff.abs().sqrt();

    match est_type {
        'm' => estimator_matheron,
        'c' => estimator_cressie,
        _ => estimator_matheron,
    }
}

fn normalization_matheron(variogram: &mut Array1<f64>, counts: &Array1<u64>) {
    Zip::from(variogram).and(counts).par_for_each(|v, c| {
        let cf = if *c == 0 { 1.0 } else { *c as f64 };
        *v /= 2.0 * cf;
    });
}

fn normalization_cressie(variogram: &mut Array1<f64>, counts: &Array1<u64>) {
    Zip::from(variogram).and(counts).par_for_each(|v, c| {
        let cf = if *c == 0 { 1.0 } else { *c as f64 };
        *v = 0.5 * (1. / cf * *v).powi(4) / (0.457 + 0.494 / cf + 0.045 / (cf * cf))
    });
}

fn choose_normalization_func(est_type: char) -> impl Fn(&mut Array1<f64>, &Array1<u64>) {
    match est_type {
        'm' => normalization_matheron,
        'c' => normalization_cressie,
        _ => normalization_matheron,
    }
}

fn normalization_matheron_vec(variogram: &mut Array2<f64>, counts: &Array2<u64>) {
    for d in 0..variogram.shape()[0] {
        //TODO get this to work
        //normalization_matheron(&mut variogram.row_mut(d), &counts.row_mut(d));

        for i in 0..variogram.shape()[1] {
            let cf = if counts[[d, i]] == 0 {
                1.0
            } else {
                counts[[d, i]] as f64
            };
            variogram[[d, i]] /= 2.0 * cf;
        }
    }
}

fn normalization_cressie_vec(variogram: &mut Array2<f64>, counts: &Array2<u64>) {
    for d in 0..variogram.shape()[0] {
        //TODO get this to work
        //normalization_cressie(&mut variogram.row_mut(d), &counts.row_mut(d));

        for i in 0..variogram.shape()[1] {
            let cf = if counts[[d, i]] == 0 {
                1.0
            } else {
                counts[[d, i]] as f64
            };
            variogram[[d, i]] = 0.5 * (1. / cf * variogram[[d, i]]).powi(4)
                / (0.457 + 0.494 / cf + 0.045 / (cf * cf))
        }
    }
}

fn choose_normalization_vec_func(est_type: char) -> impl Fn(&mut Array2<f64>, &Array2<u64>) {
    match est_type {
        'm' => normalization_matheron_vec,
        'c' => normalization_cressie_vec,
        _ => normalization_matheron_vec,
    }
}

pub fn variogram_structured(f: ArrayView2<'_, f64>, estimator_type: char) -> Array1<f64> {
    let estimator_func = choose_estimator_func(estimator_type);
    let normalization_func = choose_normalization_func(estimator_type);

    let i_max = f.shape()[0] - 1;
    let j_max = f.shape()[1];
    let k_max = i_max + 1;

    let mut variogram = Array1::<f64>::zeros(k_max);
    let mut counts = Array1::<u64>::zeros(k_max);

    (0..i_max).into_iter().for_each(|i| {
        (0..j_max).into_iter().for_each(|j| {
            (1..k_max - i).into_iter().for_each(|k| {
                counts[k] += 1;
                variogram[k] += estimator_func(f[[i, j]] - f[[i + k, j]]);
            });
        });
    });

    normalization_func(&mut variogram, &counts);

    variogram
}

pub fn variogram_ma_structured(
    f: ArrayView2<'_, f64>,
    mask: ArrayView2<'_, bool>,
    estimator_type: char,
) -> Array1<f64> {
    let estimator_func = choose_estimator_func(estimator_type);
    let normalization_func = choose_normalization_func(estimator_type);

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
                    variogram[k] += estimator_func(f[[i, j]] - f[[i + k, j]]);
                }
            });
        });
    });

    normalization_func(&mut variogram, &counts);

    variogram
}

fn dist_euclid(dim: usize, pos: ArrayView2<f64>, i: usize, j: usize) -> f64 {
    let mut dist_squared = 0.0;
    (0..dim).into_iter().for_each(|d| {
        dist_squared += (pos[[d, i]] - pos[[d, j]]).powi(2);
    });

    dist_squared.sqrt()
}

fn dist_haversine(_dim: usize, pos: ArrayView2<f64>, i: usize, j: usize) -> f64 {
    let deg_2_rad = std::f64::consts::PI / 180.0;
    let diff_lat = (pos[[0, j]] - pos[[0, i]]) * deg_2_rad;
    let diff_lon = (pos[[1, j]] - pos[[1, i]]) * deg_2_rad;
    let arg = (diff_lat / 2.0).sin().powi(2)
        + (pos[[0, i]] * deg_2_rad).cos()
            * (pos[[0, j]] * deg_2_rad).cos()
            * (diff_lon / 2.0).sin().powi(2);

    2.0 * arg.sqrt().atan2((1.0 - arg).sqrt())
}

fn choose_distance_func(dist_type: char) -> impl Fn(usize, ArrayView2<f64>, usize, usize) -> f64 {
    match dist_type {
        'e' => dist_euclid,
        _ => dist_haversine,
    }
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

    let estimator_func = choose_estimator_func(estimator_type);
    let normalization_func = choose_normalization_vec_func(estimator_type);

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
                let dist = dist_euclid(dim, pos, j, k);
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
                            variogram[[d, i]] += estimator_func(f[[m, k]] - f[[m, j]]);
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

    normalization_func(&mut variogram, &counts);

    (variogram, counts)
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

    let estimator_func = choose_estimator_func(estimator_type);
    let normalization_func = choose_normalization_func(estimator_type);
    let distance_func = choose_distance_func(distance_type);
    if distance_type == 'h' {
        assert!(dim == 2, "Haversine: dim = {} != 2", dim);
    }

    let i_max = bin_edges.shape()[0] - 1;
    let j_max = pos.shape()[1] - 1;
    let k_max = pos.shape()[1];
    let f_max = f.shape()[0];

    let mut variogram = Array1::<f64>::zeros(i_max);
    let mut counts = Array1::<u64>::zeros(i_max);

    for i in 0..i_max {
        for j in 0..j_max {
            for k in j + 1..k_max {
                let dist = distance_func(dim, pos, j, k);
                if dist < bin_edges[i] || dist >= bin_edges[i + 1] {
                    continue; //skip if not in current bin
                }
                for m in 0..f_max {
                    // skip no data values
                    if !(f[[m, k]].is_nan() || f[[m, j]].is_nan()) {
                        counts[i] += 1;
                        variogram[i] += estimator_func(f[[m, k]] - f[[m, j]]);
                    }
                }
            }
        }
    }

    normalization_func(&mut variogram, &counts);

    (variogram, counts)
}
