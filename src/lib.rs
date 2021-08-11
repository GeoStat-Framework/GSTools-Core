use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip, s};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
//#[macro_use]
//use this for into_par_iter
//use rayon::prelude::*;


#[pymodule]
#[allow(non_snake_case)]
fn gstools_core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    fn summator(
        cov_samples: ArrayView2<'_, f64>,
        z1: ArrayView1<'_, f64>,
        z2: ArrayView1<'_, f64>,
        pos: ArrayView2<'_, f64>
    ) -> Array1<f64> {
        assert!(cov_samples.shape()[0] == pos.shape()[0]);
        assert!(cov_samples.shape()[1] == z1.shape()[0]);
        assert!(z1.shape() == z2.shape());

        let mut summed_modes = Array1::<f64>::zeros(pos.dim().1);

        Zip::from(&mut summed_modes)
            .and(pos.gencolumns())
            .par_apply(|sum, pos| {
                Zip::from(cov_samples.gencolumns())
                    .and(z1)
                    .and(z2)
                    .apply(|sample, &z1, &z2| {
                        let mut phase = 0.0;
                        Zip::from(sample)
                            .and(pos)
                            .apply(|&s, &p| {
                                phase += s * p;
                        });
                    *sum += z1 * phase.cos() + z2 * phase.sin();
                })
        });
        summed_modes
    }

    //fn abs_square(vec: ArrayView1<'_, f64>) -> f64 {
        //// TODO test if parallel version really is faster
        //vec.into_par_iter()
            //.map(|&v| v * v)
            //.sum()
    //}

    fn summator_incompr(
        cov_samples: ArrayView2<'_, f64>,
        z1: ArrayView1<'_, f64>,
        z2: ArrayView1<'_, f64>,
        pos: ArrayView2<'_, f64>
    ) -> Array2<f64> {
        assert!(cov_samples.shape()[0] == pos.shape()[0]);
        assert!(cov_samples.shape()[1] == z1.shape()[0]);
        assert!(z1.shape() == z2.shape());

        let dim = pos.dim().0;

        let mut summed_modes = Array2::<f64>::zeros(pos.raw_dim());

        // unit vector in x dir.
        let mut e1 = Array1::<f64>::zeros(dim);
        e1[0] = 1.0;
        let e1 = e1;

        let mut proj = Array1::<f64>::default(dim);

        let no_pos = pos.shape()[1];
        let N = cov_samples.shape()[1];

        (0..no_pos).into_iter().for_each(|i| {
            (0..N).into_iter().for_each(|j| {
                let k_2 = cov_samples.slice(s![.., j]).dot(&cov_samples.slice(s![.., j]));
                let phase: f64 = cov_samples.slice(s![.., j]).iter()
                    .zip(pos.slice(s![.., i]))
                    .map(|(s, p)| s * p)
                    .sum();
                (0..dim).into_iter().for_each(|d| {
                    proj[d] = e1[d] - cov_samples[[d, j]] * cov_samples[[0, j]] / k_2;
                });
                (0..dim).into_iter().for_each(|d| {
                    summed_modes[[d, i]] +=
                        proj[d] * (z1[j] * phase.cos() + z2[j] * phase.sin());
                });
            });
        });
        summed_modes
    }

    fn calculator_field_krige_and_variance(
        krig_mat: ArrayView2<'_, f64>,
        krig_vecs: ArrayView2<'_, f64>,
        cond: ArrayView1<'_, f64>
    ) -> (Array1<f64>, Array1<f64>) {

        let mat_i = krig_mat.shape()[0];
        let res_i = krig_vecs.shape()[1];

        let mut field = Array1::<f64>::zeros(res_i);
        let mut error = Array1::<f64>::zeros(res_i);

        //TODO make parallel
        (0..res_i).into_iter().for_each(|k| {
            (0..mat_i).into_iter().for_each(|i| {
                let mut krig_fac = 0.0;
                (0..mat_i).into_iter().for_each(|j| {
                    krig_fac += krig_mat[[i, j]] * krig_vecs[[j, k]];
                });
                error[k] += krig_vecs[[i, k]] * krig_fac;
                field[k] += cond[i] * krig_fac;
            });
        });

        (field, error)
    }

    fn calculator_field_krige(
        krig_mat: ArrayView2<'_, f64>,
        krig_vecs: ArrayView2<'_, f64>,
        cond: ArrayView1<'_, f64>
    ) -> Array1<f64> {

        let mat_i = krig_mat.shape()[0];
        let res_i = krig_vecs.shape()[1];

        let mut field = Array1::<f64>::zeros(res_i);

        //TODO make parallel
        (0..res_i).into_iter().for_each(|k| {
            (0..mat_i).into_iter().for_each(|i| {
                let mut krig_fac = 0.0;
                (0..mat_i).into_iter().for_each(|j| {
                    krig_fac += krig_mat[[i, j]] * krig_vecs[[j, k]];
                });
                field[k] += cond[i] * krig_fac;
            });
        });

        field
    }


    fn choose_estimator_func(est_type: char) -> impl Fn(f64) -> f64 {
        let estimator_matheron = |f_diff: f64| f_diff.powi(2);
        let estimator_cressie = |f_diff: f64| f_diff.abs().sqrt();

        let estimator_func = match est_type {
            'm' =>  estimator_matheron,
            'c' =>  estimator_cressie,
            _ =>  estimator_matheron,
        };

        estimator_func
    }

    fn choose_normalization_func(est_type: char) -> impl Fn(&mut Array1<f64>, &Array1<u64>) {

        fn normalization_matheron(variogram: &mut Array1<f64>, counts: &Array1<u64>) {
            Zip::from(variogram).and(counts).par_apply(|v, c| {
                let cf = if *c == 0 {
                    1.0
                } else {
                    *c as f64
                };
                *v /= 2.0 * cf;
            });
        }

        fn normalization_cressie(variogram: &mut Array1<f64>, counts: &Array1<u64>) {
            Zip::from(variogram).and(counts).par_apply(|v, c| {
                let cf = if *c == 0 {
                    1.0
                } else {
                    *c as f64
                };
                *v = 0.5 * (1./cf * *v).powi(4) / (0.457 + 0.494 / cf + 0.045 / (cf*cf))
            });
        }

        let normalization_func = match est_type {
            'm' =>  normalization_matheron,
            'c' =>  normalization_cressie,
            _ =>  normalization_matheron,
        };

        normalization_func
    }


    fn variogram_structured(
        f: ArrayView2<'_, f64>,
        estimator_type: char
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
                (1..k_max-i).into_iter().for_each(|k| {
                    counts[k] += 1;
                    variogram[k] += estimator_func(f[[i, j]] - f[[i+k, j]]);
                });
            });
        });

        normalization_func(&mut variogram, &counts);
        //(0..k_max).into_iter().for_each(|i| {
            //variogram[i] /= 2.0 * counts[i].max(1) as f64;
        //});

        variogram
    }

    fn variogram_ma_structured(
        f: ArrayView2<'_, f64>,
        mask: ArrayView2<'_, bool>,
        estimator_type: char
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
                (1..k_max-i).into_iter().for_each(|k| {
                    if !mask[[i, j]] && !mask[[i+k, j]] {
                        counts[k] += 1;
                        variogram[k] += estimator_func(f[[i, j]] - f[[i+k, j]]);
                    }
                });
            });
        });

        normalization_func(&mut variogram, &counts);

        variogram
    }


    #[pyfn(m, "summate")]
    fn summate_py<'py>(
        py: Python<'py>,
        cov_samples: PyReadonlyArray2<f64>,
        z1: PyReadonlyArray1<f64>,
        z2: PyReadonlyArray1<f64>,
        pos: PyReadonlyArray2<f64>,
    ) -> &'py PyArray1<f64> {
        let cov_samples = cov_samples.as_array();
        let z1 = z1.as_array();
        let z2 = z2.as_array();
        let pos = pos.as_array();
        summator(cov_samples, z1, z2, pos).into_pyarray(py)
    }

    #[pyfn(m, "summate_incompr")]
    fn summate_incompr_py<'py>(
        py: Python<'py>,
        cov_samples: PyReadonlyArray2<f64>,
        z1: PyReadonlyArray1<f64>,
        z2: PyReadonlyArray1<f64>,
        pos: PyReadonlyArray2<f64>,
    ) -> &'py PyArray2<f64> {
        let cov_samples = cov_samples.as_array();
        let z1 = z1.as_array();
        let z2 = z2.as_array();
        let pos = pos.as_array();
        summator_incompr(cov_samples, z1, z2, pos).into_pyarray(py)
    }

    #[pyfn(m, "calc_field_krige_and_variance")]
    fn calc_field_krige_and_variance_py<'py>(
        py: Python<'py>,
        krige_mat: PyReadonlyArray2<f64>,
        krig_vecs: PyReadonlyArray2<f64>,
        cond: PyReadonlyArray1<f64>,
    ) -> (&'py PyArray1<f64>, &'py PyArray1<f64>) {
        let krige_mat = krige_mat.as_array();
        let krig_vecs = krig_vecs.as_array();
        let cond = cond.as_array();
        let (field, error) = calculator_field_krige_and_variance(
            krige_mat,
            krig_vecs, cond
        );
        let field = field.into_pyarray(py);
        let error = error.into_pyarray(py);
        (field, error)
    }

    #[pyfn(m, "calc_field_krige")]
    fn calc_field_krige_py<'py>(
        py: Python<'py>,
        krige_mat: PyReadonlyArray2<f64>,
        krig_vecs: PyReadonlyArray2<f64>,
        cond: PyReadonlyArray1<f64>,
    ) -> &'py PyArray1<f64> {
        let krige_mat = krige_mat.as_array();
        let krig_vecs = krig_vecs.as_array();
        let cond = cond.as_array();
        calculator_field_krige(
            krige_mat,
            krig_vecs, cond
        ).into_pyarray(py)
    }

    #[pyfn(m, "variogram_structured")]
    fn variogram_structured_py<'py>(
        py: Python<'py>,
        f: PyReadonlyArray2<f64>,
        estimator_type: char
    ) -> &'py PyArray1<f64> {
        let f = f.as_array();
        variogram_structured(f, estimator_type).into_pyarray(py)
    }

    #[pyfn(m, "variogram_ma_structured")]
    fn variogram_ma_structured_py<'py>(
        py: Python<'py>,
        f: PyReadonlyArray2<f64>,
        mask: PyReadonlyArray2<bool>,
        estimator_type: char
    ) -> &'py PyArray1<f64> {
        let f = f.as_array();
        let mask = mask.as_array();
        variogram_ma_structured(f, mask, estimator_type).into_pyarray(py)
    }

    Ok(())
}
