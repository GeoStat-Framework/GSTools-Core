//! GSTools-Core
//!
//! `gstools_core` is a Rust implementation of the core algorithms of [GSTools].
//! At the moment, it is a drop in replacement for the Cython code included in GSTools.
//!
//! This crate includes
//! - [randomization methods](field) for the random field generation
//! - the [matrix operations](krige) of the kriging methods
//! - the [variogram estimation](variogram)
//!
//! [GSTools]: https://github.com/GeoStat-Framework/GSTools

#![warn(missing_docs)]

use pyo3::prelude::pymodule;

pub mod covmodel;
pub mod local_krige;
pub mod covmodel_spec;
pub mod field;
pub mod krige;
mod short_vec;
pub mod variogram;

#[pymodule]
mod gstools_core {
    use crate::covmodel::CovModel;
    use crate::local_krige::calc_field_krige_local;
    use crate::field::{summator, summator_fourier, summator_incompr};
    use crate::krige::{calculator_field_krige, calculator_field_krige_and_variance};
    use crate::variogram::{
        variogram_directional, variogram_ma_structured, variogram_structured,
        variogram_unstructured,
    };
    use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        Ok(())
    }

    #[pyfunction(name = "summate")]
    fn summate_py<'py>(
        py: Python<'py>,
        cov_samples: PyReadonlyArray2<f64>,
        z1: PyReadonlyArray1<f64>,
        z2: PyReadonlyArray1<f64>,
        pos: PyReadonlyArray2<f64>,
        num_threads: Option<usize>,
    ) -> Bound<'py, PyArray1<f64>> {
        let cov_samples = cov_samples.as_array();
        let z1 = z1.as_array();
        let z2 = z2.as_array();
        let pos = pos.as_array();
        summator(cov_samples, z1, z2, pos, num_threads).into_pyarray(py)
    }

    #[pyfunction(name = "summate_incompr")]
    fn summate_incompr_py<'py>(
        py: Python<'py>,
        cov_samples: PyReadonlyArray2<f64>,
        z1: PyReadonlyArray1<f64>,
        z2: PyReadonlyArray1<f64>,
        pos: PyReadonlyArray2<f64>,
        num_threads: Option<usize>,
    ) -> Bound<'py, PyArray2<f64>> {
        let cov_samples = cov_samples.as_array();
        let z1 = z1.as_array();
        let z2 = z2.as_array();
        let pos = pos.as_array();
        summator_incompr(cov_samples, z1, z2, pos, num_threads).into_pyarray(py)
    }

    #[pyfunction(name = "summate_fourier")]
    fn summate_fourier_py<'py>(
        py: Python<'py>,
        spectrum_factor: PyReadonlyArray1<f64>,
        modes: PyReadonlyArray2<f64>,
        z1: PyReadonlyArray1<f64>,
        z2: PyReadonlyArray1<f64>,
        pos: PyReadonlyArray2<f64>,
        num_threads: Option<usize>,
    ) -> Bound<'py, PyArray1<f64>> {
        let spectrum_factor = spectrum_factor.as_array();
        let modes = modes.as_array();
        let z1 = z1.as_array();
        let z2 = z2.as_array();
        let pos = pos.as_array();
        summator_fourier(spectrum_factor, modes, z1, z2, pos, num_threads).into_pyarray(py)
    }

    #[pyfunction(name = "calc_field_krige_and_variance")]
    fn calc_field_krige_and_variance_py<'py>(
        py: Python<'py>,
        krige_mat: PyReadonlyArray2<f64>,
        krig_vecs: PyReadonlyArray2<f64>,
        cond: PyReadonlyArray1<f64>,
        num_threads: Option<usize>,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        let krige_mat = krige_mat.as_array();
        let krig_vecs = krig_vecs.as_array();
        let cond = cond.as_array();
        let (field, error) =
            calculator_field_krige_and_variance(krige_mat, krig_vecs, cond, num_threads);
        let field = field.into_pyarray(py);
        let error = error.into_pyarray(py);
        (field, error)
    }

    #[pyfunction(name = "calc_field_krige")]
    fn calc_field_krige_py<'py>(
        py: Python<'py>,
        krige_mat: PyReadonlyArray2<f64>,
        krig_vecs: PyReadonlyArray2<f64>,
        cond: PyReadonlyArray1<f64>,
        num_threads: Option<usize>,
    ) -> Bound<'py, PyArray1<f64>> {
        let krige_mat = krige_mat.as_array();
        let krig_vecs = krig_vecs.as_array();
        let cond = cond.as_array();
        calculator_field_krige(krige_mat, krig_vecs, cond, num_threads).into_pyarray(py)
    }

    #[pyfunction(name = "calc_field_krige_local")]
    #[allow(clippy::too_many_arguments)]
    fn calc_field_krige_local_py<'py>(
        py: Python<'py>,
        cond_pos: PyReadonlyArray2<f64>,
        cond_val: PyReadonlyArray1<f64>,
        target_pos: PyReadonlyArray2<f64>,
        cov_model_json: &str,
        cond_err: PyReadonlyArray1<f64>,
        drift_cond: PyReadonlyArray2<f64>,
        drift_target: PyReadonlyArray2<f64>,
        unbiased: bool,
        exact: bool,
        local_radius: f64,
        num_threads: Option<usize>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let model = CovModel::from_json(cov_model_json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let cond_pos = cond_pos.as_array();
        let cond_val = cond_val.as_array();
        let target_pos = target_pos.as_array();
        let cond_err = cond_err.as_array();
        let drift_cond = drift_cond.as_array();
        let drift_target = drift_target.as_array();
        let (field, error) = calc_field_krige_local(
            cond_pos,
            cond_val,
            target_pos,
            &model,
            cond_err,
            drift_cond,
            drift_target,
            unbiased,
            exact,
            local_radius,
            num_threads,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((field.into_pyarray(py), error.into_pyarray(py)))
    }

    #[pyfunction(name = "variogram_structured")]
    fn variogram_structured_py<'py>(
        py: Python<'py>,
        f: PyReadonlyArray2<f64>,
        estimator_type: Option<char>,
        num_threads: Option<usize>,
    ) -> Bound<'py, PyArray1<f64>> {
        let f = f.as_array();
        let estimator_type = estimator_type.unwrap_or('m');
        variogram_structured(f, estimator_type, num_threads).into_pyarray(py)
    }

    #[pyfunction(name = "variogram_ma_structured")]
    fn variogram_ma_structured_py<'py>(
        py: Python<'py>,
        f: PyReadonlyArray2<f64>,
        mask: PyReadonlyArray2<bool>,
        estimator_type: Option<char>,
        num_threads: Option<usize>,
    ) -> Bound<'py, PyArray1<f64>> {
        let f = f.as_array();
        let mask = mask.as_array();
        let estimator_type = estimator_type.unwrap_or('m');
        variogram_ma_structured(f, mask, estimator_type, num_threads).into_pyarray(py)
    }

    #[pyfunction(name = "variogram_directional")]
    #[allow(clippy::too_many_arguments)]
    fn variogram_directional_py<'py>(
        py: Python<'py>,
        f: PyReadonlyArray2<f64>,
        bin_edges: PyReadonlyArray1<f64>,
        pos: PyReadonlyArray2<f64>,
        direction: PyReadonlyArray2<f64>, //should be normed
        angles_tol: Option<f64>,
        bandwidth: Option<f64>,
        separate_dirs: Option<bool>,
        estimator_type: Option<char>,
        num_threads: Option<usize>,
    ) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<u64>>) {
        let f = f.as_array();
        let bin_edges = bin_edges.as_array();
        let pos = pos.as_array();
        let direction = direction.as_array();
        let angles_tol = angles_tol.unwrap_or(std::f64::consts::PI / 8.0);
        let bandwidth = bandwidth.unwrap_or(-1.0);
        let separate_dirs = separate_dirs.unwrap_or(false);
        let estimator_type = estimator_type.unwrap_or('m');
        let (variogram, counts) = variogram_directional(
            f,
            bin_edges,
            pos,
            direction,
            angles_tol,
            bandwidth,
            separate_dirs,
            estimator_type,
            num_threads,
        );
        let variogram = variogram.into_pyarray(py);
        let counts = counts.into_pyarray(py);

        (variogram, counts)
    }

    #[pyfunction(name = "variogram_unstructured")]
    fn variogram_unstructured_py<'py>(
        py: Python<'py>,
        f: PyReadonlyArray2<f64>,
        bin_edges: PyReadonlyArray1<f64>,
        pos: PyReadonlyArray2<f64>,
        estimator_type: Option<char>,
        distance_type: Option<char>,
        num_threads: Option<usize>,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<u64>>) {
        let f = f.as_array();
        let bin_edges = bin_edges.as_array();
        let pos = pos.as_array();
        let estimator_type = estimator_type.unwrap_or('m');
        let distance_type = distance_type.unwrap_or('e');
        let (variogram, counts) = variogram_unstructured(
            f,
            bin_edges,
            pos,
            estimator_type,
            distance_type,
            num_threads,
        );
        let variogram = variogram.into_pyarray(py);
        let counts = counts.into_pyarray(py);

        (variogram, counts)
    }
}
