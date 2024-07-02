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

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

use field::{summator, summator_fourier, summator_incompr};
use krige::{calculator_field_krige, calculator_field_krige_and_variance};
use variogram::{
    variogram_directional, variogram_ma_structured, variogram_structured, variogram_unstructured,
};

pub mod field;
pub mod krige;
mod short_vec;
pub mod variogram;

#[pymodule]
fn gstools_core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    #[pyfn(m)]
    #[pyo3(name = "summate")]
    fn summate_py<'py>(
        py: Python<'py>,
        cov_samples: PyReadonlyArray2<f64>,
        z1: PyReadonlyArray1<f64>,
        z2: PyReadonlyArray1<f64>,
        pos: PyReadonlyArray2<f64>,
        num_threads: Option<usize>,
    ) -> &'py PyArray1<f64> {
        let cov_samples = cov_samples.as_array();
        let z1 = z1.as_array();
        let z2 = z2.as_array();
        let pos = pos.as_array();
        summator(cov_samples, z1, z2, pos, num_threads).into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "summate_incompr")]
    fn summate_incompr_py<'py>(
        py: Python<'py>,
        cov_samples: PyReadonlyArray2<f64>,
        z1: PyReadonlyArray1<f64>,
        z2: PyReadonlyArray1<f64>,
        pos: PyReadonlyArray2<f64>,
        num_threads: Option<usize>,
    ) -> &'py PyArray2<f64> {
        let cov_samples = cov_samples.as_array();
        let z1 = z1.as_array();
        let z2 = z2.as_array();
        let pos = pos.as_array();
        summator_incompr(cov_samples, z1, z2, pos, num_threads).into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "summate_fourier")]
    fn summate_fourier_py<'py>(
        py: Python<'py>,
        spectrum_factor: PyReadonlyArray1<f64>,
        modes: PyReadonlyArray2<f64>,
        z1: PyReadonlyArray1<f64>,
        z2: PyReadonlyArray1<f64>,
        pos: PyReadonlyArray2<f64>,
        num_threads: Option<usize>,
    ) -> &'py PyArray1<f64> {
        let spectrum_factor = spectrum_factor.as_array();
        let modes = modes.as_array();
        let z1 = z1.as_array();
        let z2 = z2.as_array();
        let pos = pos.as_array();
        summator_fourier(spectrum_factor, modes, z1, z2, pos, num_threads).into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "calc_field_krige_and_variance")]
    fn calc_field_krige_and_variance_py<'py>(
        py: Python<'py>,
        krige_mat: PyReadonlyArray2<f64>,
        krig_vecs: PyReadonlyArray2<f64>,
        cond: PyReadonlyArray1<f64>,
        num_threads: Option<usize>,
    ) -> (&'py PyArray1<f64>, &'py PyArray1<f64>) {
        let krige_mat = krige_mat.as_array();
        let krig_vecs = krig_vecs.as_array();
        let cond = cond.as_array();
        let (field, error) =
            calculator_field_krige_and_variance(krige_mat, krig_vecs, cond, num_threads);
        let field = field.into_pyarray(py);
        let error = error.into_pyarray(py);
        (field, error)
    }

    #[pyfn(m)]
    #[pyo3(name = "calc_field_krige")]
    fn calc_field_krige_py<'py>(
        py: Python<'py>,
        krige_mat: PyReadonlyArray2<f64>,
        krig_vecs: PyReadonlyArray2<f64>,
        cond: PyReadonlyArray1<f64>,
        num_threads: Option<usize>,
    ) -> &'py PyArray1<f64> {
        let krige_mat = krige_mat.as_array();
        let krig_vecs = krig_vecs.as_array();
        let cond = cond.as_array();
        calculator_field_krige(krige_mat, krig_vecs, cond, num_threads).into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "variogram_structured")]
    fn variogram_structured_py<'py>(
        py: Python<'py>,
        f: PyReadonlyArray2<f64>,
        estimator_type: Option<char>,
        num_threads: Option<usize>,
    ) -> &'py PyArray1<f64> {
        let f = f.as_array();
        let estimator_type = estimator_type.unwrap_or('m');
        variogram_structured(f, estimator_type, num_threads).into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "variogram_ma_structured")]
    fn variogram_ma_structured_py<'py>(
        py: Python<'py>,
        f: PyReadonlyArray2<f64>,
        mask: PyReadonlyArray2<bool>,
        estimator_type: Option<char>,
        num_threads: Option<usize>,
    ) -> &'py PyArray1<f64> {
        let f = f.as_array();
        let mask = mask.as_array();
        let estimator_type = estimator_type.unwrap_or('m');
        variogram_ma_structured(f, mask, estimator_type, num_threads).into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "variogram_directional")]
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
    ) -> (&'py PyArray2<f64>, &'py PyArray2<u64>) {
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

    #[pyfn(m)]
    #[pyo3(name = "variogram_unstructured")]
    fn variogram_unstructured_py<'py>(
        py: Python<'py>,
        f: PyReadonlyArray2<f64>,
        bin_edges: PyReadonlyArray1<f64>,
        pos: PyReadonlyArray2<f64>,
        estimator_type: Option<char>,
        distance_type: Option<char>,
        num_threads: Option<usize>,
    ) -> (&'py PyArray1<f64>, &'py PyArray1<u64>) {
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

    Ok(())
}
