use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Zip};

//#[macro_use]
//use this for into_par_iter
//use rayon::prelude::*;

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
        .and(pos.gencolumns())
        .par_apply(|sum, pos| {
            Zip::from(cov_samples.gencolumns())
                .and(z1)
                .and(z2)
                .apply(|sample, &z1, &z2| {
                    let mut phase = 0.0;
                    Zip::from(sample).and(pos).apply(|&s, &p| {
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

    let mut summed_modes = Array2::<f64>::zeros(pos.raw_dim());

    // unit vector in x dir.
    let mut e1 = Array1::<f64>::zeros(dim);
    e1[0] = 1.0;
    let e1 = e1;

    let mut proj = Array1::<f64>::default(dim);

    (0..pos.shape()[1]).into_iter().for_each(|i| {
        (0..cov_samples.shape()[1]).into_iter().for_each(|j| {
            let k_2 = cov_samples
                .slice(s![.., j])
                .dot(&cov_samples.slice(s![.., j]));
            let phase: f64 = cov_samples
                .slice(s![.., j])
                .iter()
                .zip(pos.slice(s![.., i]))
                .map(|(s, p)| s * p)
                .sum();
            (0..dim).into_iter().for_each(|d| {
                proj[d] = e1[d] - cov_samples[[d, j]] * cov_samples[[0, j]] / k_2;
            });
            (0..dim).into_iter().for_each(|d| {
                summed_modes[[d, i]] += proj[d] * (z1[j] * phase.cos() + z2[j] * phase.sin());
            });
        });
    });
    summed_modes
}
