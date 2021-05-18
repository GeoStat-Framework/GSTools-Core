use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip, s};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};


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

    Ok(())
}
