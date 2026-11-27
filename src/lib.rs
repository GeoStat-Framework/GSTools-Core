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

pub mod field;
pub mod krige;
pub mod mps;
pub mod mps_engine;
mod short_vec;
pub mod variogram;

#[pymodule]
mod gstools_core {
    use crate::field::{summator, summator_fourier, summator_incompr};
    use crate::krige::{calculator_field_krige, calculator_field_krige_and_variance};
    use crate::mps::{
        dist_block_categorical, dist_block_categorical_masked, dist_block_categorical_rayon,
        dist_block_l1, dist_block_l1_masked, dist_block_l2, dist_block_l2_masked, dist_block_lp,
        dist_block_lp_masked, dist_block_variation, dist_block_variation_masked, scan_node,
        scan_node_categorical, validate_block_inputs, validate_scan_geometry, validate_scan_inputs,
    };
    use crate::mps_engine::{cleanup_pass_engine, simulate_engine};
    use crate::variogram::{
        variogram_directional, variogram_ma_structured, variogram_structured,
        variogram_unstructured,
    };
    use numpy::{
        IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    };
    use pyo3::prelude::*;
    use pyo3::{exceptions::PyValueError, PyResult};

    type MpsSimulationPyOutput<'py> =
        (Bound<'py, PyArray2<f64>>, usize, usize, usize, usize, usize);

    type MpsCleanupPyOutput<'py> = (Bound<'py, PyArray2<f64>>, usize, usize, usize);

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

    #[pyfunction(name = "mps_dist_block_cat")]
    #[allow(clippy::too_many_arguments)]
    fn mps_dist_block_cat_py<'py>(
        py: Python<'py>,
        de_sim: PyReadonlyArray1<f64>,
        ti_flat: PyReadonlyArray1<f64>,
        base_flat: PyReadonlyArray1<i64>,
        lag_flat: PyReadonlyArray1<i64>,
        node_weights: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        validate_block_inputs(
            de_sim.as_array(),
            ti_flat.as_array(),
            base_flat.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
            false,
            None,
        )
        .map_err(PyValueError::new_err)?;
        Ok(dist_block_categorical(
            de_sim.as_array(),
            ti_flat.as_array(),
            base_flat.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
        )
        .into_pyarray(py))
    }

    #[pyfunction(name = "mps_dist_block_cat_rayon")]
    #[allow(clippy::too_many_arguments)]
    fn mps_dist_block_cat_rayon_py<'py>(
        py: Python<'py>,
        de_sim: PyReadonlyArray1<f64>,
        ti_flat: PyReadonlyArray1<f64>,
        base_flat: PyReadonlyArray1<i64>,
        lag_flat: PyReadonlyArray1<i64>,
        node_weights: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        validate_block_inputs(
            de_sim.as_array(),
            ti_flat.as_array(),
            base_flat.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
            false,
            None,
        )
        .map_err(PyValueError::new_err)?;
        Ok(dist_block_categorical_rayon(
            de_sim.as_array(),
            ti_flat.as_array(),
            base_flat.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
        )
        .into_pyarray(py))
    }

    #[pyfunction(name = "mps_rayon_num_threads")]
    fn mps_rayon_num_threads_py() -> usize {
        rayon::current_num_threads()
    }

    /// MPS native ABI / capability revision.
    ///
    /// Bumped whenever the MPS entry-point signatures (`mps_simulate`,
    /// `mps_scan_node`, ...) change in a way the Python layer must detect. The
    /// crate `__version__` alone is insufficient: it can stay the same across
    /// an ABI change, so the Python `gstools.mps._core` capability gate keys
    /// off this integer instead.
    ///
    /// Revision history:
    /// * 1 — initial multivariate engine (scalar `distance_power`/`cond_weight`).
    /// * 2 — `mps_simulate` takes per-variable `distance_power`/`cond_weight`
    ///   arrays (length == number of TI variables) instead of scalars.
    /// * 3 — `mps_simulate`/`mps_scan_node` take per-variable categorical
    ///   penalty matrices (`penalty_flat`/`penalty_offsets`/`penalty_ncat`) and
    ///   honour `metric_kind == 3` (categorical penalty distance).
    /// * 4 — `mps_simulate` takes a `random_scan` flag (and `mps_scan_node` a
    ///   `random_scan`/`scan_seed` pair) selecting random vs sequential TI scan.
    /// * 5 — `mps_simulate` takes zone training images
    ///   (`zone_ti`/`zone_offsets`/`zone_shapes`/`zone_d_max`) and a
    ///   `node_domain` selector, scanning a per-node domain's TI.
    /// * 6 — adds `mps_cleanup_pass`, one serial Gauss–Seidel post-processing
    ///   (cleanup) pass that reopens and re-simulates an already-complete field.
    /// * 7 — `mps_cleanup_pass` takes `num_threads` and runs the pass over a
    ///   position-oriented dependency DAG (deterministic serial == parallel);
    ///   it now returns the effective thread count too.
    /// * 8 — explicit active simulation masks for main and cleanup engines.
    #[pyfunction(name = "mps_api_version")]
    fn mps_api_version_py() -> u32 {
        8
    }

    #[pyfunction(name = "mps_dist_block_cat_masked")]
    #[allow(clippy::too_many_arguments)]
    fn mps_dist_block_cat_masked_py<'py>(
        py: Python<'py>,
        de_sim: PyReadonlyArray1<f64>,
        ti_flat: PyReadonlyArray1<f64>,
        base_flat: PyReadonlyArray1<i64>,
        lag_flat: PyReadonlyArray1<i64>,
        node_weights: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        validate_block_inputs(
            de_sim.as_array(),
            ti_flat.as_array(),
            base_flat.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
            true,
            None,
        )
        .map_err(PyValueError::new_err)?;
        Ok(dist_block_categorical_masked(
            de_sim.as_array(),
            ti_flat.as_array(),
            base_flat.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
        )
        .into_pyarray(py))
    }

    #[pyfunction(name = "mps_dist_block_l1")]
    #[allow(clippy::too_many_arguments)]
    fn mps_dist_block_l1_py<'py>(
        py: Python<'py>,
        de_sim: PyReadonlyArray1<f64>,
        ti_flat: PyReadonlyArray1<f64>,
        base_flat: PyReadonlyArray1<i64>,
        lag_flat: PyReadonlyArray1<i64>,
        node_weights: PyReadonlyArray1<f64>,
        d_max: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        validate_block_inputs(
            de_sim.as_array(),
            ti_flat.as_array(),
            base_flat.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
            false,
            Some((d_max, 1.0)),
        )
        .map_err(PyValueError::new_err)?;
        Ok(dist_block_l1(
            de_sim.as_array(),
            ti_flat.as_array(),
            base_flat.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
            d_max,
        )
        .into_pyarray(py))
    }

    #[pyfunction(name = "mps_dist_block_l1_masked")]
    #[allow(clippy::too_many_arguments)]
    fn mps_dist_block_l1_masked_py<'py>(
        py: Python<'py>,
        de_sim: PyReadonlyArray1<f64>,
        ti_flat: PyReadonlyArray1<f64>,
        base_flat: PyReadonlyArray1<i64>,
        lag_flat: PyReadonlyArray1<i64>,
        node_weights: PyReadonlyArray1<f64>,
        d_max: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        validate_block_inputs(
            de_sim.as_array(),
            ti_flat.as_array(),
            base_flat.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
            true,
            Some((d_max, 1.0)),
        )
        .map_err(PyValueError::new_err)?;
        Ok(dist_block_l1_masked(
            de_sim.as_array(),
            ti_flat.as_array(),
            base_flat.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
            d_max,
        )
        .into_pyarray(py))
    }

    #[pyfunction(name = "mps_dist_block_l2")]
    #[allow(clippy::too_many_arguments)]
    fn mps_dist_block_l2_py<'py>(
        py: Python<'py>,
        de_sim: PyReadonlyArray1<f64>,
        ti_flat: PyReadonlyArray1<f64>,
        base_flat: PyReadonlyArray1<i64>,
        lag_flat: PyReadonlyArray1<i64>,
        node_weights: PyReadonlyArray1<f64>,
        d_max: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        validate_block_inputs(
            de_sim.as_array(),
            ti_flat.as_array(),
            base_flat.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
            false,
            Some((d_max, 1.0)),
        )
        .map_err(PyValueError::new_err)?;
        Ok(dist_block_l2(
            de_sim.as_array(),
            ti_flat.as_array(),
            base_flat.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
            d_max,
        )
        .into_pyarray(py))
    }

    #[pyfunction(name = "mps_dist_block_l2_masked")]
    #[allow(clippy::too_many_arguments)]
    fn mps_dist_block_l2_masked_py<'py>(
        py: Python<'py>,
        de_sim: PyReadonlyArray1<f64>,
        ti_flat: PyReadonlyArray1<f64>,
        base_flat: PyReadonlyArray1<i64>,
        lag_flat: PyReadonlyArray1<i64>,
        node_weights: PyReadonlyArray1<f64>,
        d_max: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        validate_block_inputs(
            de_sim.as_array(),
            ti_flat.as_array(),
            base_flat.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
            true,
            Some((d_max, 1.0)),
        )
        .map_err(PyValueError::new_err)?;
        Ok(dist_block_l2_masked(
            de_sim.as_array(),
            ti_flat.as_array(),
            base_flat.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
            d_max,
        )
        .into_pyarray(py))
    }

    #[pyfunction(name = "mps_dist_block_lp")]
    #[allow(clippy::too_many_arguments)]
    fn mps_dist_block_lp_py<'py>(
        py: Python<'py>,
        de_sim: PyReadonlyArray1<f64>,
        ti_flat: PyReadonlyArray1<f64>,
        base_flat: PyReadonlyArray1<i64>,
        lag_flat: PyReadonlyArray1<i64>,
        node_weights: PyReadonlyArray1<f64>,
        d_max: f64,
        p: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        validate_block_inputs(
            de_sim.as_array(),
            ti_flat.as_array(),
            base_flat.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
            false,
            Some((d_max, p)),
        )
        .map_err(PyValueError::new_err)?;
        Ok(dist_block_lp(
            de_sim.as_array(),
            ti_flat.as_array(),
            base_flat.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
            d_max,
            p,
        )
        .into_pyarray(py))
    }

    #[pyfunction(name = "mps_dist_block_lp_masked")]
    #[allow(clippy::too_many_arguments)]
    fn mps_dist_block_lp_masked_py<'py>(
        py: Python<'py>,
        de_sim: PyReadonlyArray1<f64>,
        ti_flat: PyReadonlyArray1<f64>,
        base_flat: PyReadonlyArray1<i64>,
        lag_flat: PyReadonlyArray1<i64>,
        node_weights: PyReadonlyArray1<f64>,
        d_max: f64,
        p: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        validate_block_inputs(
            de_sim.as_array(),
            ti_flat.as_array(),
            base_flat.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
            true,
            Some((d_max, p)),
        )
        .map_err(PyValueError::new_err)?;
        Ok(dist_block_lp_masked(
            de_sim.as_array(),
            ti_flat.as_array(),
            base_flat.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
            d_max,
            p,
        )
        .into_pyarray(py))
    }

    #[pyfunction(name = "mps_dist_block_variation")]
    #[allow(clippy::too_many_arguments)]
    fn mps_dist_block_variation_py<'py>(
        py: Python<'py>,
        de_sim: PyReadonlyArray1<f64>,
        ti_flat: PyReadonlyArray1<f64>,
        base_flat: PyReadonlyArray1<i64>,
        lag_flat: PyReadonlyArray1<i64>,
        node_weights: PyReadonlyArray1<f64>,
        d_max: f64,
        p: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        validate_block_inputs(
            de_sim.as_array(),
            ti_flat.as_array(),
            base_flat.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
            false,
            Some((d_max, p)),
        )
        .map_err(PyValueError::new_err)?;
        Ok(dist_block_variation(
            de_sim.as_array(),
            ti_flat.as_array(),
            base_flat.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
            d_max,
            p,
        )
        .into_pyarray(py))
    }

    #[pyfunction(name = "mps_dist_block_variation_masked")]
    #[allow(clippy::too_many_arguments)]
    fn mps_dist_block_variation_masked_py<'py>(
        py: Python<'py>,
        de_sim: PyReadonlyArray1<f64>,
        ti_flat: PyReadonlyArray1<f64>,
        base_flat: PyReadonlyArray1<i64>,
        lag_flat: PyReadonlyArray1<i64>,
        node_weights: PyReadonlyArray1<f64>,
        d_max: f64,
        p: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        validate_block_inputs(
            de_sim.as_array(),
            ti_flat.as_array(),
            base_flat.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
            true,
            Some((d_max, p)),
        )
        .map_err(PyValueError::new_err)?;
        Ok(dist_block_variation_masked(
            de_sim.as_array(),
            ti_flat.as_array(),
            base_flat.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
            d_max,
            p,
        )
        .into_pyarray(py))
    }

    #[pyfunction(name = "mps_scan_node_cat")]
    #[allow(clippy::too_many_arguments)]
    fn mps_scan_node_cat_py<'py>(
        py: Python<'py>,
        lo: PyReadonlyArray1<i64>,
        win_shape: PyReadonlyArray1<i64>,
        start: usize,
        max_scan: usize,
        threshold: f64,
        de_sim: PyReadonlyArray1<f64>,
        ti_flat: PyReadonlyArray1<f64>,
        ti_strides: PyReadonlyArray1<i64>,
        lag_flat: PyReadonlyArray1<i64>,
        node_weights: PyReadonlyArray1<f64>,
    ) -> PyResult<Option<Bound<'py, PyArray1<i64>>>> {
        validate_scan_geometry(
            lo.as_array(),
            win_shape.as_array(),
            ti_strides.as_array(),
            ti_flat.as_array().len(),
            lag_flat.as_array(),
            threshold,
        )
        .map_err(PyValueError::new_err)?;
        validate_block_inputs(
            de_sim.as_array(),
            ti_flat.as_array(),
            ndarray::ArrayView1::from(&[]),
            lag_flat.as_array(),
            node_weights.as_array(),
            false,
            None,
        )
        .map_err(PyValueError::new_err)?;
        Ok(scan_node_categorical(
            lo.as_array(),
            win_shape.as_array(),
            start,
            max_scan,
            threshold,
            de_sim.as_array(),
            ti_flat.as_array(),
            ti_strides.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
        )
        .map(|arr| arr.into_pyarray(py)))
    }

    #[pyfunction(name = "mps_scan_node")]
    #[allow(clippy::too_many_arguments)]
    fn mps_scan_node_py<'py>(
        py: Python<'py>,
        lo: PyReadonlyArray1<i64>,
        win_shape: PyReadonlyArray1<i64>,
        start: usize,
        max_scan: usize,
        threshold: f64,
        ti_flat: PyReadonlyArray2<f64>,
        ti_strides: PyReadonlyArray1<i64>,
        active_ti_rows: PyReadonlyArray1<i64>,
        event_offsets: PyReadonlyArray1<i64>,
        de_sim: PyReadonlyArray1<f64>,
        lag_flat: PyReadonlyArray1<i64>,
        node_weights: PyReadonlyArray1<f64>,
        variable_weights: PyReadonlyArray1<f64>,
        metric_kinds: PyReadonlyArray1<i64>,
        has_nan: PyReadonlyArray1<u8>,
        d_max: PyReadonlyArray1<f64>,
        p_norm: PyReadonlyArray1<f64>,
        random_scan: bool,
        scan_seed: u64,
        penalty_flat: PyReadonlyArray1<f64>,
        penalty_offsets: PyReadonlyArray1<i64>,
        penalty_ncat: PyReadonlyArray1<i64>,
        target_ti_rows: PyReadonlyArray1<i64>,
        check_centers: bool,
    ) -> PyResult<Option<Bound<'py, PyArray1<i64>>>> {
        validate_scan_inputs(
            lo.as_array(),
            win_shape.as_array(),
            threshold,
            ti_flat.as_array(),
            ti_strides.as_array(),
            active_ti_rows.as_array(),
            event_offsets.as_array(),
            de_sim.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
            variable_weights.as_array(),
            metric_kinds.as_array(),
            has_nan.as_array(),
            d_max.as_array(),
            p_norm.as_array(),
            penalty_flat.as_array(),
            penalty_offsets.as_array(),
            penalty_ncat.as_array(),
            target_ti_rows.as_array(),
        )
        .map_err(PyValueError::new_err)?;
        Ok(scan_node(
            lo.as_array(),
            win_shape.as_array(),
            start,
            max_scan,
            threshold,
            ti_flat.as_array(),
            ti_strides.as_array(),
            active_ti_rows.as_array(),
            event_offsets.as_array(),
            de_sim.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
            variable_weights.as_array(),
            metric_kinds.as_array(),
            has_nan.as_array(),
            d_max.as_array(),
            p_norm.as_array(),
            random_scan,
            scan_seed,
            penalty_flat.as_array(),
            penalty_offsets.as_array(),
            penalty_ncat.as_array(),
            target_ti_rows.as_array(),
            check_centers,
        )
        .map(|arr| arr.into_pyarray(py)))
    }

    #[pyfunction(name = "mps_simulate")]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (ti, ti_shape, initial_fields, conditioned, sim_shape, path, lag_matrices, u_start, u_fallback, offsets, n_neighbors, max_radius, variable_weights, metric_kinds, has_nan, d_max, p_norm, penalty_flat, penalty_offsets, penalty_ncat, threshold, scan_fraction, random_scan, distance_power, cond_weight, partial_boundary, num_threads, zone_ti, zone_offsets, zone_shapes, zone_d_max, node_domain, active_mask=None))]
    fn mps_simulate_py<'py>(
        py: Python<'py>,
        ti: PyReadonlyArray2<f64>,
        ti_shape: PyReadonlyArray1<i64>,
        initial_fields: PyReadonlyArray2<f64>,
        conditioned: PyReadonlyArray2<u8>,
        sim_shape: PyReadonlyArray1<i64>,
        path: PyReadonlyArray2<i64>,
        lag_matrices: PyReadonlyArray3<f64>,
        u_start: PyReadonlyArray1<f64>,
        u_fallback: PyReadonlyArray2<f64>,
        offsets: PyReadonlyArray2<i64>,
        n_neighbors: PyReadonlyArray1<i64>,
        max_radius: PyReadonlyArray1<f64>,
        variable_weights: PyReadonlyArray1<f64>,
        metric_kinds: PyReadonlyArray1<i64>,
        has_nan: PyReadonlyArray1<u8>,
        d_max: PyReadonlyArray1<f64>,
        p_norm: PyReadonlyArray1<f64>,
        penalty_flat: PyReadonlyArray1<f64>,
        penalty_offsets: PyReadonlyArray1<i64>,
        penalty_ncat: PyReadonlyArray1<i64>,
        threshold: f64,
        scan_fraction: f64,
        random_scan: bool,
        distance_power: PyReadonlyArray1<f64>,
        cond_weight: PyReadonlyArray1<f64>,
        partial_boundary: bool,
        num_threads: usize,
        zone_ti: PyReadonlyArray2<f64>,
        zone_offsets: PyReadonlyArray1<i64>,
        zone_shapes: PyReadonlyArray2<i64>,
        zone_d_max: PyReadonlyArray2<f64>,
        node_domain: PyReadonlyArray1<i64>,
        active_mask: Option<PyReadonlyArray1<u8>>,
    ) -> PyResult<MpsSimulationPyOutput<'py>> {
        // Own the arrays before detaching so Python code may run concurrently
        // without being able to mutate buffers borrowed by the Rust engine.
        // This deliberately copies the full TI and all run inputs once per
        // simulation. It is the GIL-safety boundary, not an allocation-free
        // path; release memory claims must include these Rust-owned copies.
        let ti = ti.as_array().to_owned();
        let ti_shape = ti_shape.as_array().to_owned();
        let initial_fields = initial_fields.as_array().to_owned();
        let conditioned = conditioned.as_array().to_owned();
        let sim_shape = sim_shape.as_array().to_owned();
        let path = path.as_array().to_owned();
        let lag_matrices = lag_matrices.as_array().to_owned();
        let u_start = u_start.as_array().to_owned();
        let u_fallback = u_fallback.as_array().to_owned();
        let offsets = offsets.as_array().to_owned();
        let n_neighbors = n_neighbors.as_array().to_owned();
        let max_radius = max_radius.as_array().to_owned();
        let variable_weights = variable_weights.as_array().to_owned();
        let metric_kinds = metric_kinds.as_array().to_owned();
        let has_nan = has_nan.as_array().to_owned();
        let d_max = d_max.as_array().to_owned();
        let p_norm = p_norm.as_array().to_owned();
        let penalty_flat = penalty_flat.as_array().to_owned();
        let penalty_offsets = penalty_offsets.as_array().to_owned();
        let penalty_ncat = penalty_ncat.as_array().to_owned();
        let distance_power = distance_power.as_array().to_owned();
        let cond_weight = cond_weight.as_array().to_owned();
        let zone_ti = zone_ti.as_array().to_owned();
        let zone_offsets = zone_offsets.as_array().to_owned();
        let zone_shapes = zone_shapes.as_array().to_owned();
        let zone_d_max = zone_d_max.as_array().to_owned();
        let node_domain = node_domain.as_array().to_owned();
        let active_mask = active_mask.map(|mask| mask.as_array().to_owned());

        let output = py
            .detach(move || {
                simulate_engine(
                    ti.view(),
                    ti_shape.view(),
                    initial_fields.view(),
                    conditioned.view(),
                    sim_shape.view(),
                    path.view(),
                    lag_matrices.view(),
                    u_start.view(),
                    u_fallback.view(),
                    offsets.view(),
                    n_neighbors.view(),
                    max_radius.view(),
                    variable_weights.view(),
                    metric_kinds.view(),
                    has_nan.view(),
                    d_max.view(),
                    p_norm.view(),
                    penalty_flat.view(),
                    penalty_offsets.view(),
                    penalty_ncat.view(),
                    threshold,
                    scan_fraction,
                    random_scan,
                    distance_power.view(),
                    cond_weight.view(),
                    partial_boundary,
                    num_threads,
                    zone_ti.view(),
                    zone_offsets.view(),
                    zone_shapes.view(),
                    zone_d_max.view(),
                    node_domain.view(),
                    active_mask.as_ref().map(|mask| mask.view()),
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            output.fields.into_pyarray(py),
            output.strict_fallback_count,
            output.level_count,
            output.max_ready_width,
            output.used_threads,
            output.collapsed_lag_count,
        ))
    }

    #[pyfunction(name = "mps_cleanup_pass")]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (ti, ti_shape, fields_in, conditioned, sim_shape, path, lag_matrices, u_start, u_fallback, offsets, n_neighbors, max_radius, variable_weights, metric_kinds, has_nan, d_max, p_norm, penalty_flat, penalty_offsets, penalty_ncat, threshold, scan_fraction, random_scan, distance_power, cond_weight, partial_boundary, num_threads, zone_ti, zone_offsets, zone_shapes, zone_d_max, node_domain, active_mask=None))]
    fn mps_cleanup_pass_py<'py>(
        py: Python<'py>,
        ti: PyReadonlyArray2<f64>,
        ti_shape: PyReadonlyArray1<i64>,
        fields_in: PyReadonlyArray2<f64>,
        conditioned: PyReadonlyArray2<u8>,
        sim_shape: PyReadonlyArray1<i64>,
        path: PyReadonlyArray2<i64>,
        lag_matrices: PyReadonlyArray3<f64>,
        u_start: PyReadonlyArray1<f64>,
        u_fallback: PyReadonlyArray2<f64>,
        offsets: PyReadonlyArray2<i64>,
        n_neighbors: PyReadonlyArray1<i64>,
        max_radius: PyReadonlyArray1<f64>,
        variable_weights: PyReadonlyArray1<f64>,
        metric_kinds: PyReadonlyArray1<i64>,
        has_nan: PyReadonlyArray1<u8>,
        d_max: PyReadonlyArray1<f64>,
        p_norm: PyReadonlyArray1<f64>,
        penalty_flat: PyReadonlyArray1<f64>,
        penalty_offsets: PyReadonlyArray1<i64>,
        penalty_ncat: PyReadonlyArray1<i64>,
        threshold: f64,
        scan_fraction: f64,
        random_scan: bool,
        distance_power: PyReadonlyArray1<f64>,
        cond_weight: PyReadonlyArray1<f64>,
        partial_boundary: bool,
        num_threads: usize,
        zone_ti: PyReadonlyArray2<f64>,
        zone_offsets: PyReadonlyArray1<i64>,
        zone_shapes: PyReadonlyArray2<i64>,
        zone_d_max: PyReadonlyArray2<f64>,
        node_domain: PyReadonlyArray1<i64>,
        active_mask: Option<PyReadonlyArray1<u8>>,
    ) -> PyResult<MpsCleanupPyOutput<'py>> {
        // Own every array before detaching (the GIL-safety boundary, mirroring
        // mps_simulate): Python may run concurrently and must not mutate buffers
        // borrowed by the Rust pass.
        let ti = ti.as_array().to_owned();
        let ti_shape = ti_shape.as_array().to_owned();
        let fields_in = fields_in.as_array().to_owned();
        let conditioned = conditioned.as_array().to_owned();
        let sim_shape = sim_shape.as_array().to_owned();
        let path = path.as_array().to_owned();
        let lag_matrices = lag_matrices.as_array().to_owned();
        let u_start = u_start.as_array().to_owned();
        let u_fallback = u_fallback.as_array().to_owned();
        let offsets = offsets.as_array().to_owned();
        let n_neighbors = n_neighbors.as_array().to_owned();
        let max_radius = max_radius.as_array().to_owned();
        let variable_weights = variable_weights.as_array().to_owned();
        let metric_kinds = metric_kinds.as_array().to_owned();
        let has_nan = has_nan.as_array().to_owned();
        let d_max = d_max.as_array().to_owned();
        let p_norm = p_norm.as_array().to_owned();
        let penalty_flat = penalty_flat.as_array().to_owned();
        let penalty_offsets = penalty_offsets.as_array().to_owned();
        let penalty_ncat = penalty_ncat.as_array().to_owned();
        let distance_power = distance_power.as_array().to_owned();
        let cond_weight = cond_weight.as_array().to_owned();
        let zone_ti = zone_ti.as_array().to_owned();
        let zone_offsets = zone_offsets.as_array().to_owned();
        let zone_shapes = zone_shapes.as_array().to_owned();
        let zone_d_max = zone_d_max.as_array().to_owned();
        let node_domain = node_domain.as_array().to_owned();
        let active_mask = active_mask.map(|mask| mask.as_array().to_owned());

        let (fields, strict_fallback_count, collapsed_lag_count, used_threads) = py
            .detach(move || {
                cleanup_pass_engine(
                    ti.view(),
                    ti_shape.view(),
                    fields_in.view(),
                    conditioned.view(),
                    sim_shape.view(),
                    path.view(),
                    lag_matrices.view(),
                    u_start.view(),
                    u_fallback.view(),
                    offsets.view(),
                    n_neighbors.view(),
                    max_radius.view(),
                    variable_weights.view(),
                    metric_kinds.view(),
                    has_nan.view(),
                    d_max.view(),
                    p_norm.view(),
                    penalty_flat.view(),
                    penalty_offsets.view(),
                    penalty_ncat.view(),
                    threshold,
                    scan_fraction,
                    random_scan,
                    distance_power.view(),
                    cond_weight.view(),
                    partial_boundary,
                    num_threads,
                    zone_ti.view(),
                    zone_offsets.view(),
                    zone_shapes.view(),
                    zone_d_max.view(),
                    node_domain.view(),
                    active_mask.as_ref().map(|mask| mask.view()),
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            fields.into_pyarray(py),
            strict_fallback_count,
            collapsed_lag_count,
            used_threads,
        ))
    }
}
