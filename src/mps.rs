//! MPS (Multiple Point Statistics) kernels for Direct Sampling.
//!
//! Fused gather and distance functions used by the Rust simulation engine and
//! retained Python-callable numerical helpers. Kernels read TI candidates via
//! flat-index arithmetic without allocating a full candidate/event matrix.
//! Python exports validate buffers before entering these trusted loops.

use ndarray::{s, Array1, ArrayView1, ArrayView2};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::collections::HashMap;

/// Validate a flattened candidate block once, before trusted numerical loops.
/// Signed bases/lags are allowed when their sums are valid. Bounds use i128
/// extrema, requiring O(B+n) index checks rather than the full candidate/event
/// product. Value validation also inspects the supplied TI buffer once.
pub(crate) fn validate_block_inputs(
    de: ArrayView1<'_, f64>,
    ti: ArrayView1<'_, f64>,
    bases: ArrayView1<'_, i64>,
    lags: ArrayView1<'_, i64>,
    weights: ArrayView1<'_, f64>,
    masked: bool,
    continuous: Option<(f64, f64)>,
) -> Result<(), String> {
    if de.len() != lags.len() || de.len() != weights.len() {
        return Err("MPS block event, lag and weight lengths must agree".into());
    }
    if de.iter().any(|v| !v.is_finite())
        || ti
            .iter()
            .any(|v| v.is_infinite() || (!masked && v.is_nan()))
    {
        return Err("MPS block values must be finite (masked TI values may be NaN)".into());
    }
    validate_weights(weights)?;
    if let Some((range, exponent)) = continuous {
        if !range.is_finite() || range <= 0.0 || !exponent.is_finite() || exponent <= 0.0 {
            return Err("MPS continuous range and exponent must be finite and positive".into());
        }
    }
    if let (Some(min_lag), Some(max_lag)) = (lags.iter().min(), lags.iter().max()) {
        for &base in bases.iter() {
            if (base as i128 + *min_lag as i128) < 0
                || (base as i128 + *max_lag as i128) >= ti.len() as i128
            {
                return Err("MPS flattened TI index out of bounds".into());
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_weights(weights: ArrayView1<'_, f64>) -> Result<(), String> {
    if weights.iter().any(|w| !w.is_finite() || *w < 0.0) || !weights.sum().is_finite() {
        return Err("MPS weights must be finite, nonnegative, with a finite sum".into());
    }
    Ok(())
}

/// Checked geometry for raw scans. The signature exposes flattened buffers,
/// not full TI axes; this proves flattened memory safety, not physical geometry.
pub(crate) fn validate_scan_geometry(
    lo: ArrayView1<'_, i64>,
    shape: ArrayView1<'_, i64>,
    strides: ArrayView1<'_, i64>,
    ti_size: usize,
    lags: ArrayView1<'_, i64>,
    threshold: f64,
) -> Result<(), String> {
    if lo.len() != shape.len() || lo.len() != strides.len() {
        return Err("MPS scan coordinate, window and stride dimensions must agree".into());
    }
    if !threshold.is_finite() || threshold < 0.0 {
        return Err("MPS scan threshold must be finite and nonnegative".into());
    }
    let mut size = 1_usize;
    let mut min_base = 0_i128;
    let mut max_base = 0_i128;
    for ((&lower, &extent), &stride) in lo.iter().zip(shape.iter()).zip(strides.iter()) {
        if extent < 0 || stride <= 0 {
            return Err("MPS scan sizes must be nonnegative and strides positive".into());
        }
        size = size
            .checked_mul(extent as usize)
            .filter(|&n| n <= i64::MAX as usize)
            .ok_or("MPS scan window size overflow")?;
        let upper = lower
            .checked_add(extent.saturating_sub(1))
            .ok_or("MPS scan coordinate overflow")?;
        let lower_term = lower as i128 * stride as i128;
        let upper_term = upper as i128 * stride as i128;
        // The trusted loop multiplies in i64 before adding terms.
        if lower_term < i64::MIN as i128 || upper_term > i64::MAX as i128 {
            return Err("MPS scan stride arithmetic overflow".into());
        }
        min_base = min_base
            .checked_add(lower_term)
            .ok_or("MPS scan index overflow")?;
        max_base = max_base
            .checked_add(upper_term)
            .ok_or("MPS scan index overflow")?;
    }
    if lo.is_empty() || size == 0 {
        return Ok(());
    }
    // Nonnegative coordinates prevent overflow in intermediate signed sums.
    if lo.iter().any(|&x| x < 0) || min_base < 0 || max_base >= ti_size as i128 {
        return Err("MPS scan candidate center out of bounds".into());
    }
    if let (Some(min_lag), Some(max_lag)) = (lags.iter().min(), lags.iter().max()) {
        if min_base + (*min_lag as i128) < 0 || max_base + (*max_lag as i128) >= ti_size as i128 {
            return Err("MPS scan event gather index out of bounds".into());
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_scan_inputs(
    lo: ArrayView1<'_, i64>,
    shape: ArrayView1<'_, i64>,
    threshold: f64,
    ti: ArrayView2<'_, f64>,
    strides: ArrayView1<'_, i64>,
    rows: ArrayView1<'_, i64>,
    offsets: ArrayView1<'_, i64>,
    de: ArrayView1<'_, f64>,
    lags: ArrayView1<'_, i64>,
    weights: ArrayView1<'_, f64>,
    variable_weights: ArrayView1<'_, f64>,
    metrics: ArrayView1<'_, i64>,
    has_nan: ArrayView1<'_, u8>,
    ranges: ArrayView1<'_, f64>,
    exponents: ArrayView1<'_, f64>,
    penalty: ArrayView1<'_, f64>,
    penalty_offsets: ArrayView1<'_, i64>,
    penalty_ncat: ArrayView1<'_, i64>,
    targets: ArrayView1<'_, i64>,
) -> Result<(), String> {
    validate_scan_geometry(lo, shape, strides, ti.ncols(), lags, threshold)?;
    let n = rows.len();
    if offsets.len() != n + 1
        || [
            variable_weights.len(),
            metrics.len(),
            has_nan.len(),
            ranges.len(),
            exponents.len(),
            penalty_offsets.len(),
            penalty_ncat.len(),
        ]
        .iter()
        .any(|&len| len != n)
        || de.len() != lags.len()
        || de.len() != weights.len()
    {
        return Err("MPS scan descriptor/event lengths must agree".into());
    }
    if offsets[0] != 0
        || offsets[n] != de.len() as i64
        || offsets.iter().any(|&x| x < 0 || x as usize > de.len())
        || offsets
            .iter()
            .zip(offsets.iter().skip(1))
            .any(|(a, b)| a > b)
    {
        return Err("MPS scan event offsets must partition the event arrays".into());
    }
    if rows
        .iter()
        .chain(targets.iter())
        .any(|&r| r < 0 || r as usize >= ti.nrows())
    {
        return Err("MPS scan TI row out of bounds".into());
    }
    if de.iter().any(|x| !x.is_finite()) || ti.iter().any(|x| x.is_infinite()) {
        return Err("MPS scan event values must be finite; TI may contain NaN".into());
    }
    validate_weights(weights)?;
    validate_weights(variable_weights)?;
    validate_metric_descriptors(
        metrics,
        ranges,
        exponents,
        penalty,
        penalty_offsets,
        penalty_ncat,
    )?;
    for (v, &row) in rows.iter().enumerate() {
        if has_nan[v] == 0 && ti.row(row as usize).iter().any(|x| x.is_nan()) {
            return Err("MPS scan missingness flag does not match TI row".into());
        }
    }
    Ok(())
}

pub(crate) fn validate_metric_descriptors(
    metrics: ArrayView1<'_, i64>,
    ranges: ArrayView1<'_, f64>,
    exponents: ArrayView1<'_, f64>,
    penalty: ArrayView1<'_, f64>,
    offsets: ArrayView1<'_, i64>,
    ncat: ArrayView1<'_, i64>,
) -> Result<(), String> {
    for v in 0..metrics.len() {
        if !(0..=3).contains(&metrics[v]) {
            return Err("MPS metric kind must be 0, 1, 2 or 3".into());
        }
        if !ranges[v].is_finite()
            || ranges[v] <= 0.0
            || !exponents[v].is_finite()
            || exponents[v] <= 0.0
        {
            return Err("MPS ranges and exponents must be finite and positive".into());
        }
        if offsets[v] < 0 || ncat[v] < 0 {
            return Err("MPS penalty descriptors must be nonnegative".into());
        }
        if metrics[v] == 3 {
            let count = ncat[v] as usize;
            let end = count
                .checked_mul(count)
                .and_then(|x| (offsets[v] as usize).checked_add(x))
                .ok_or("MPS penalty size overflow")?;
            if count == 0 || end > penalty.len() {
                return Err("MPS penalty block is empty or too short".into());
            }
            for i in 0..count {
                for j in 0..count {
                    let cost = penalty[offsets[v] as usize + i * count + j];
                    if !cost.is_finite() || !(0.0..=1.0).contains(&cost) || (i == j && cost != 0.0)
                    {
                        return Err(
                            "MPS penalty costs must be finite in [0,1], with zero diagonal".into(),
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[inline]
fn categorical_candidate_distance(
    base: i64,
    de_sim: ArrayView1<'_, f64>,
    ti_flat: ArrayView1<'_, f64>,
    lag_flat: ArrayView1<'_, i64>,
    node_weights: ArrayView1<'_, f64>,
) -> f64 {
    de_sim
        .iter()
        .zip(lag_flat.iter())
        .zip(node_weights.iter())
        .fold(0.0_f64, |distance, ((&de_val, &lag), &weight)| {
            let ti_val = ti_flat[(base + lag) as usize];
            distance + if de_val != ti_val { weight } else { 0.0 }
        })
}

#[inline]
fn categorical_masked_candidate_distance(
    base: i64,
    de_sim: ArrayView1<'_, f64>,
    ti_flat: ArrayView1<'_, f64>,
    lag_flat: ArrayView1<'_, i64>,
    node_weights: ArrayView1<'_, f64>,
) -> f64 {
    let (mismatch, weight_sum) = de_sim
        .iter()
        .zip(lag_flat.iter())
        .zip(node_weights.iter())
        .fold(
            (0.0_f64, 0.0_f64),
            |(mismatch, weight_sum), ((&de_val, &lag), &weight)| {
                let ti_val = ti_flat[(base + lag) as usize];
                if ti_val.is_nan() {
                    (mismatch, weight_sum)
                } else {
                    (
                        mismatch + if de_val != ti_val { weight } else { 0.0 },
                        weight_sum + weight,
                    )
                }
            },
        );
    if weight_sum > 0.0 {
        mismatch / weight_sum
    } else {
        f64::INFINITY
    }
}

/// Fused gather + weighted categorical distance over a block of TI candidates.
///
/// For each candidate `b`, reads TI values at flat positions
/// `base_flat[b] + lag_flat[i]` for each lag `i` and computes the weighted
/// categorical mismatch against `de_sim`:
///
/// `dist[b] = Σᵢ node_weights[i] · (de_sim[i] ≠ ti_flat[base_flat[b] + lag_flat[i]])`
///
/// The caller guarantees all `base_flat[b] + lag_flat[i]` indices are in-bounds.
/// Python exports validate this invariant; the production Rust engine
/// constructs valid TI search windows before calling the trusted kernels.
///
/// # Arguments
///
/// * `de_sim` – Data-event category labels at the current simulation node, shape `(n,)`.
///   Labels are stored as `f64` (the same representation as the simulation grid).
/// * `ti_flat` – Flattened C-order TI array, shape `(ti_size,)`.
///   Category labels are stored as `f64` (pre-cast by `_DirectSamplingEngine`).
/// * `base_flat` – Flat base index for each candidate anchor, shape `(B,)`.
///   Computed by Python as `(y_blk @ ti_strides).astype(np.int64)`.
/// * `lag_flat` – Flat lag offsets, shape `(n,)`.
///   Computed by Python as `(int_lags @ ti_strides).astype(np.int64)`.
/// * `node_weights` – Precomputed spatial-decay weights normalized to sum 1, shape `(n,)`.
///
/// # Returns
///
/// Distance array of shape `(B,)` with values in `[0.0, 1.0]`.
pub fn dist_block_categorical(
    de_sim: ArrayView1<'_, f64>,
    ti_flat: ArrayView1<'_, f64>,
    base_flat: ArrayView1<'_, i64>,
    lag_flat: ArrayView1<'_, i64>,
    node_weights: ArrayView1<'_, f64>,
) -> Array1<f64> {
    Array1::from_iter(
        base_flat.iter().map(|&base| {
            categorical_candidate_distance(base, de_sim, ti_flat, lag_flat, node_weights)
        }),
    )
}

/// Parallel (Rayon) variant of [`dist_block_categorical`] for standalone
/// benchmarking; not used by the production DS engine.
///
/// Candidate rows are independent and are evaluated in parallel. Rayon uses
/// an indexed iterator, so the returned array preserves the input candidate
/// order. The lag/weight accumulation within each candidate remains strictly
/// sequential, preserving the scalar summation order and thread-count-
/// independent results.
///
/// This function uses the currently installed Rayon pool. Set
/// `RAYON_NUM_THREADS` before the first Rayon call to configure that pool.
///
/// # Arguments
///
/// Arguments have the same meaning and shape contract as
/// [`dist_block_categorical`].
///
/// # Returns
///
/// Distance array of shape `(B,)` in the original candidate order.
pub fn dist_block_categorical_rayon(
    de_sim: ArrayView1<'_, f64>,
    ti_flat: ArrayView1<'_, f64>,
    base_flat: ArrayView1<'_, i64>,
    lag_flat: ArrayView1<'_, i64>,
    node_weights: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let distances: Vec<f64> = (0..base_flat.len())
        .into_par_iter()
        .map(|index| {
            categorical_candidate_distance(
                base_flat[index],
                de_sim,
                ti_flat,
                lag_flat,
                node_weights,
            )
        })
        .collect();
    Array1::from_vec(distances)
}

/// Fused gather + masked weighted categorical distance over TI candidates.
///
/// This is the masked counterpart of [`dist_block_categorical`]. For every
/// candidate, TI event positions whose value is `NaN` are excluded. The
/// surviving node weights are renormalized within that candidate:
///
/// `dist[b] = Σ(valid mismatch weights) / Σ(valid weights)`
///
/// A candidate with no positive surviving weight returns `+∞`, matching
/// the masked distance contract. TI NaN values denote missing support; the
/// public boundary rejects infinity and nonfinite simulation events.
///
/// The caller guarantees all flat indices are in bounds. The Rust production
/// engine validates candidate centers, selects the winner, handles fallback,
/// retrieves the selected value, and writes the simulation grid.
///
/// # Arguments
///
/// * `de_sim` – Data-event category labels, shape `(n,)`.
/// * `ti_flat` – Flattened C-order TI, including possible `NaN`, shape `(ti_size,)`.
/// * `base_flat` – Flat candidate-anchor indices, shape `(B,)`.
/// * `lag_flat` – Flat event-lag offsets, shape `(n,)`.
/// * `node_weights` – Precomputed node weights, shape `(n,)`.
///
/// # Returns
///
/// Distance array of shape `(B,)`; excluded candidates contain `+∞`.
pub fn dist_block_categorical_masked(
    de_sim: ArrayView1<'_, f64>,
    ti_flat: ArrayView1<'_, f64>,
    base_flat: ArrayView1<'_, i64>,
    lag_flat: ArrayView1<'_, i64>,
    node_weights: ArrayView1<'_, f64>,
) -> Array1<f64> {
    Array1::from_iter(base_flat.iter().map(|&base| {
        categorical_masked_candidate_distance(base, de_sim, ti_flat, lag_flat, node_weights)
    }))
}

/// Parallel (Rayon) masked categorical kernel for standalone benchmarking only.
///
/// It preserves candidate order and keeps each candidate's masked lag
/// reduction sequential. It is not wired into the production DS engine.
///
/// # Arguments
///
/// Arguments have the same meaning and shape contract as
/// [`dist_block_categorical_masked`].
///
/// # Returns
///
/// Distance array of shape `(B,)`; excluded candidates contain `+∞`.
pub fn dist_block_categorical_masked_rayon(
    de_sim: ArrayView1<'_, f64>,
    ti_flat: ArrayView1<'_, f64>,
    base_flat: ArrayView1<'_, i64>,
    lag_flat: ArrayView1<'_, i64>,
    node_weights: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let distances: Vec<f64> = (0..base_flat.len())
        .into_par_iter()
        .map(|index| {
            categorical_masked_candidate_distance(
                base_flat[index],
                de_sim,
                ti_flat,
                lag_flat,
                node_weights,
            )
        })
        .collect();
    Array1::from_vec(distances)
}

/// Fused gather + weighted L1 distance over a block of TI candidates.
///
/// `dist[b] = Σᵢ node_weights[i] · |de_sim[i] − ti_flat[base_flat[b] + lag_flat[i]]| / d_max`
///
/// # Arguments
///
/// * `de_sim` – Data-event values at the current simulation node, shape `(n,)`.
/// * `ti_flat` – Flattened C-order TI array, shape `(ti_size,)`.
/// * `base_flat` – Flat base index for each candidate anchor, shape `(B,)`.
/// * `lag_flat` – Flat lag offsets, shape `(n,)`.
/// * `node_weights` – Precomputed spatial-decay weights normalized to sum 1, shape `(n,)`.
/// * `d_max` – TI value range (max − min over finite cells); from `Variable.d_max`.
///   `Variable._data_range` guarantees this is positive.
///
/// # Returns
///
/// Distance array of shape `(B,)`, values nominally in `[0, 1]`.
pub fn dist_block_l1(
    de_sim: ArrayView1<'_, f64>,
    ti_flat: ArrayView1<'_, f64>,
    base_flat: ArrayView1<'_, i64>,
    lag_flat: ArrayView1<'_, i64>,
    node_weights: ArrayView1<'_, f64>,
    d_max: f64,
) -> Array1<f64> {
    Array1::from_iter(base_flat.iter().map(|&base| {
        de_sim
            .iter()
            .zip(lag_flat.iter())
            .zip(node_weights.iter())
            .fold(0.0_f64, |d, ((&de_val, &lag), &wt)| {
                let ti_val = ti_flat[(base + lag) as usize];
                d + wt * (de_val - ti_val).abs() / d_max
            })
    }))
}

/// Fused gather + masked weighted L1 distance over TI candidates.
///
/// This preserves `vec_l1_dist(..., has_nan=True)`: candidate TI event
/// positions containing `NaN` are excluded, surviving node weights are
/// renormalized per candidate, and a candidate with no positive surviving
/// weight returns `+∞`.
///
/// `dist[b] = Σ(valid weight · |SG − TI| / d_max) / Σ(valid weights)`
///
/// Missingness is deliberately `is_nan`, not `is_finite`, to match the current
/// Python oracle exactly. The existing [`dist_block_l1`] remains the separate,
/// overhead-free NaN-free path.
///
/// # Arguments
///
/// * `de_sim` – Simulation-grid event values, shape `(n,)`.
/// * `ti_flat` – Flattened C-order TI, including possible `NaN`, shape `(ti_size,)`.
/// * `base_flat` – Flat candidate-anchor indices, shape `(B,)`.
/// * `lag_flat` – Flat event-lag offsets, shape `(n,)`.
/// * `node_weights` – Precomputed node weights, shape `(n,)`.
/// * `d_max` – TI finite-value range used for normalization. Must be positive.
///
/// # Returns
///
/// Distance array of shape `(B,)`; excluded candidates contain `+∞`.
pub fn dist_block_l1_masked(
    de_sim: ArrayView1<'_, f64>,
    ti_flat: ArrayView1<'_, f64>,
    base_flat: ArrayView1<'_, i64>,
    lag_flat: ArrayView1<'_, i64>,
    node_weights: ArrayView1<'_, f64>,
    d_max: f64,
) -> Array1<f64> {
    Array1::from_iter(base_flat.iter().map(|&base| {
        let (error, weight_sum) = de_sim
            .iter()
            .zip(lag_flat.iter())
            .zip(node_weights.iter())
            .fold(
                (0.0_f64, 0.0_f64),
                |(error, weight_sum), ((&de_val, &lag), &weight)| {
                    let ti_val = ti_flat[(base + lag) as usize];
                    if ti_val.is_nan() {
                        (error, weight_sum)
                    } else {
                        (
                            error + weight * (de_val - ti_val).abs() / d_max,
                            weight_sum + weight,
                        )
                    }
                },
            );
        if weight_sum > 0.0 {
            error / weight_sum
        } else {
            f64::INFINITY
        }
    }))
}

/// Fused gather + weighted L2 distance over a block of TI candidates.
///
/// `dist[b] = √(Σᵢ node_weights[i] · ((de_sim[i] − ti_flat[base_flat[b] + lag_flat[i]]) / d_max)²)`
///
/// # Arguments
///
/// * `de_sim` – Data-event values at the current simulation node, shape `(n,)`.
/// * `ti_flat` – Flattened C-order TI array, shape `(ti_size,)`.
/// * `base_flat` – Flat base index for each candidate anchor, shape `(B,)`.
/// * `lag_flat` – Flat lag offsets, shape `(n,)`.
/// * `node_weights` – Precomputed spatial-decay weights normalized to sum 1, shape `(n,)`.
/// * `d_max` – TI value range for normalization. Must be positive.
///
/// # Returns
///
/// Distance array of shape `(B,)`.
pub fn dist_block_l2(
    de_sim: ArrayView1<'_, f64>,
    ti_flat: ArrayView1<'_, f64>,
    base_flat: ArrayView1<'_, i64>,
    lag_flat: ArrayView1<'_, i64>,
    node_weights: ArrayView1<'_, f64>,
    d_max: f64,
) -> Array1<f64> {
    Array1::from_iter(base_flat.iter().map(|&base| {
        de_sim
            .iter()
            .zip(lag_flat.iter())
            .zip(node_weights.iter())
            .fold(0.0_f64, |acc, ((&de_val, &lag), &wt)| {
                let diff = (de_val - ti_flat[(base + lag) as usize]) / d_max;
                acc + wt * diff * diff
            })
            .sqrt()
    }))
}

/// Fused gather + masked weighted L2 distance over TI candidates.
///
/// Candidate TI event positions containing `NaN` are excluded. Surviving
/// node weights are renormalized for each candidate before taking the square
/// root:
///
/// `dist[b] = √(Σ(valid weight · ((SG − TI) / d_max)²) / Σ(valid weights))`
///
/// A candidate with no positive surviving weight returns `+∞`. Missingness is
/// defined by `is_nan`; Python-callable boundaries reject infinite inputs.
///
/// # Arguments
///
/// * `de_sim` – Simulation-grid event values, shape `(n,)`.
/// * `ti_flat` – Flattened C-order TI, including possible `NaN`, shape `(ti_size,)`.
/// * `base_flat` – Flat candidate-anchor indices, shape `(B,)`.
/// * `lag_flat` – Flat event-lag offsets, shape `(n,)`.
/// * `node_weights` – Precomputed node weights, shape `(n,)`.
/// * `d_max` – TI finite-value range used for normalization. Must be positive.
///
/// # Returns
///
/// Distance array of shape `(B,)`; excluded candidates contain `+∞`.
pub fn dist_block_l2_masked(
    de_sim: ArrayView1<'_, f64>,
    ti_flat: ArrayView1<'_, f64>,
    base_flat: ArrayView1<'_, i64>,
    lag_flat: ArrayView1<'_, i64>,
    node_weights: ArrayView1<'_, f64>,
    d_max: f64,
) -> Array1<f64> {
    Array1::from_iter(base_flat.iter().map(|&base| {
        let (squared_error, weight_sum) = de_sim
            .iter()
            .zip(lag_flat.iter())
            .zip(node_weights.iter())
            .fold(
                (0.0_f64, 0.0_f64),
                |(squared_error, weight_sum), ((&de_val, &lag), &weight)| {
                    let ti_val = ti_flat[(base + lag) as usize];
                    if ti_val.is_nan() {
                        (squared_error, weight_sum)
                    } else {
                        let diff = (de_val - ti_val) / d_max;
                        (squared_error + weight * diff * diff, weight_sum + weight)
                    }
                },
            );
        if weight_sum > 0.0 {
            (squared_error / weight_sum).sqrt()
        } else {
            f64::INFINITY
        }
    }))
}

/// Fused gather + weighted Lp distance over a block of TI candidates.
///
/// `dist[b] = (Σᵢ node_weights[i] · |de_sim[i] − ti_flat[base_flat[b] + lag_flat[i]]|ᵖ / d_maxᵖ)^(1/p)`
///
/// For `p = 1` this is equivalent to `dist_block_l1`; for `p = 2` to `dist_block_l2`.
/// Prefer those specialized functions for `p ∈ {1, 2}` as they avoid `f64::powf`.
///
/// # Arguments
///
/// * `de_sim` – Data-event values at the current simulation node, shape `(n,)`.
/// * `ti_flat` – Flattened C-order TI array, shape `(ti_size,)`.
/// * `base_flat` – Flat base index for each candidate anchor, shape `(B,)`.
/// * `lag_flat` – Flat lag offsets, shape `(n,)`.
/// * `node_weights` – Precomputed spatial-decay weights normalized to sum 1, shape `(n,)`.
/// * `d_max` – TI value range for normalization. Must be positive.
/// * `p` – Lp norm exponent; from `Variable.p_norm`. Must be positive.
///
/// # Returns
///
/// Distance array of shape `(B,)`.
pub fn dist_block_lp(
    de_sim: ArrayView1<'_, f64>,
    ti_flat: ArrayView1<'_, f64>,
    base_flat: ArrayView1<'_, i64>,
    lag_flat: ArrayView1<'_, i64>,
    node_weights: ArrayView1<'_, f64>,
    d_max: f64,
    p: f64,
) -> Array1<f64> {
    let inv_p = p.recip();
    Array1::from_iter(base_flat.iter().map(|&base| {
        de_sim
            .iter()
            .zip(lag_flat.iter())
            .zip(node_weights.iter())
            .fold(0.0_f64, |acc, ((&de_val, &lag), &wt)| {
                let diff = (de_val - ti_flat[(base + lag) as usize]).abs() / d_max;
                acc + wt * diff.powf(p)
            })
            .powf(inv_p)
    }))
}

/// Fused gather + masked weighted Lp distance over TI candidates.
///
/// Candidate TI event positions containing `NaN` are excluded and surviving
/// weights are renormalized per candidate:
///
/// `dist[b] = (Σ(valid weight · |(SG − TI) / d_max|ᵖ) / Σ(valid weights))^(1/p)`
///
/// A candidate with no positive surviving weight returns `+∞`. The `p = 1`
/// and `p = 2` cases delegate to the specialized masked L1 and L2 kernels so
/// they retain the same arithmetic and avoid `powf`.
///
/// # Arguments
///
/// * `de_sim` – Simulation-grid event values, shape `(n,)`.
/// * `ti_flat` – Flattened C-order TI, including possible `NaN`, shape `(ti_size,)`.
/// * `base_flat` – Flat candidate-anchor indices, shape `(B,)`.
/// * `lag_flat` – Flat event-lag offsets, shape `(n,)`.
/// * `node_weights` – Precomputed node weights, shape `(n,)`.
/// * `d_max` – TI finite-value range used for normalization. Must be positive.
/// * `p` – Lp norm exponent. Must be positive.
///
/// # Returns
///
/// Distance array of shape `(B,)`; excluded candidates contain `+∞`.
#[allow(clippy::too_many_arguments)]
pub fn dist_block_lp_masked(
    de_sim: ArrayView1<'_, f64>,
    ti_flat: ArrayView1<'_, f64>,
    base_flat: ArrayView1<'_, i64>,
    lag_flat: ArrayView1<'_, i64>,
    node_weights: ArrayView1<'_, f64>,
    d_max: f64,
    p: f64,
) -> Array1<f64> {
    if p == 1.0 {
        return dist_block_l1_masked(de_sim, ti_flat, base_flat, lag_flat, node_weights, d_max);
    }
    if p == 2.0 {
        return dist_block_l2_masked(de_sim, ti_flat, base_flat, lag_flat, node_weights, d_max);
    }

    let inv_p = p.recip();
    Array1::from_iter(base_flat.iter().map(|&base| {
        let (powered_error, weight_sum) = de_sim
            .iter()
            .zip(lag_flat.iter())
            .zip(node_weights.iter())
            .fold(
                (0.0_f64, 0.0_f64),
                |(powered_error, weight_sum), ((&de_val, &lag), &weight)| {
                    let ti_val = ti_flat[(base + lag) as usize];
                    if ti_val.is_nan() {
                        (powered_error, weight_sum)
                    } else {
                        let diff = (de_val - ti_val).abs() / d_max;
                        (powered_error + weight * diff.powf(p), weight_sum + weight)
                    }
                },
            );
        if weight_sum > 0.0 {
            (powered_error / weight_sum).powf(inv_p)
        } else {
            f64::INFINITY
        }
    }))
}

/// Fused gather + weighted variation distance over a block of TI candidates.
///
/// Variation distance compares the *shape* of the pattern (deviation from the
/// local mean) rather than raw values, so a constant vertical shift between two
/// otherwise identical patterns has distance zero. For each candidate `b`, with
/// SG mean `S̄ = mean(de_sim)` and candidate TI mean `T̄_b = mean over the event
/// positions of that candidate`:
///
/// `dist[b] = min(1, [Σᵢ node_weights[i] · |((de_sim[i] − S̄) − (T_{b,i} − T̄_b)) / (2·d_max)|ᵖ]^(1/p))`
///
/// Both means are **unweighted** arithmetic means over the `n` event positions;
/// the node weights apply only to the centered error contributions. The `S̄`
/// mean is computed once per call; `T̄_b` is candidate-specific and needs one
/// extra pass over the event positions per candidate.
///
/// This kernel is the NaN-free counterpart of `vec_variation_dist` in
/// `gstools/mps/distance.py`. It returns only the distance vector — the
/// winning-anchor mean shift is applied separately by the Rust production
/// engine or the retained `TrainingImage.adjust_value` helper.
///
/// # Arguments
///
/// * `de_sim` – Data-event values at the current simulation node, shape `(n,)`.
/// * `ti_flat` – Flattened C-order TI array, shape `(ti_size,)`.
/// * `base_flat` – Flat base index for each candidate anchor, shape `(B,)`.
/// * `lag_flat` – Flat lag offsets, shape `(n,)`.
/// * `node_weights` – Precomputed spatial-decay weights normalized to sum 1, shape `(n,)`.
/// * `d_max` – TI value range for normalization. Must be positive.
/// * `p` – Variation Lp exponent; from `Variable.variation_p_norm`. Must be positive.
///
/// # Returns
///
/// Distance array of shape `(B,)` with values in `[0, 1]`.
#[allow(clippy::too_many_arguments)]
pub fn dist_block_variation(
    de_sim: ArrayView1<'_, f64>,
    ti_flat: ArrayView1<'_, f64>,
    base_flat: ArrayView1<'_, i64>,
    lag_flat: ArrayView1<'_, i64>,
    node_weights: ArrayView1<'_, f64>,
    d_max: f64,
    p: f64,
) -> Array1<f64> {
    let n = de_sim.len() as f64;
    // SG mean is the same for every candidate; compute it once.
    let de_mean = de_sim.sum() / n;
    let two_dmax = 2.0 * d_max;
    // `f64::min` ignores a NaN operand, whereas NumPy's `minimum` propagates
    // it. Preserve the Python oracle for non-finite direct-FFI inputs too.
    let clamp = |value: f64| {
        if value.is_nan() {
            value
        } else {
            value.min(1.0)
        }
    };

    // Candidate TI mean over the event positions (unweighted); one pass.
    let ti_mean = |base: i64| -> f64 {
        lag_flat
            .iter()
            .map(|&lag| ti_flat[(base + lag) as usize])
            .sum::<f64>()
            / n
    };

    // Specialize p == 2 (the `variation` default) and p == 1 to avoid a
    // per-element `powf`, which otherwise dominates runtime and makes the
    // fused kernel slower than NumPy's vectorized power. General `p` keeps the
    // `powf` path. `2*d_max` normalizes the common case to [0, 1]; SG values
    // are not bounded by the TI range (conditioning / mean-shifts), so clamp.
    if p == 2.0 {
        Array1::from_iter(base_flat.iter().map(|&base| {
            let tm = ti_mean(base);
            let inner = de_sim
                .iter()
                .zip(lag_flat.iter())
                .zip(node_weights.iter())
                .fold(0.0_f64, |acc, ((&de_val, &lag), &wt)| {
                    let ti_val = ti_flat[(base + lag) as usize];
                    let c = ((de_val - de_mean) - (ti_val - tm)) / two_dmax;
                    acc + wt * c * c
                });
            clamp(inner.sqrt())
        }))
    } else if p == 1.0 {
        Array1::from_iter(base_flat.iter().map(|&base| {
            let tm = ti_mean(base);
            let inner = de_sim
                .iter()
                .zip(lag_flat.iter())
                .zip(node_weights.iter())
                .fold(0.0_f64, |acc, ((&de_val, &lag), &wt)| {
                    let ti_val = ti_flat[(base + lag) as usize];
                    let c = ((de_val - de_mean) - (ti_val - tm)) / two_dmax;
                    acc + wt * c.abs()
                });
            clamp(inner)
        }))
    } else {
        let inv_p = p.recip();
        Array1::from_iter(base_flat.iter().map(|&base| {
            let tm = ti_mean(base);
            let inner = de_sim
                .iter()
                .zip(lag_flat.iter())
                .zip(node_weights.iter())
                .fold(0.0_f64, |acc, ((&de_val, &lag), &wt)| {
                    let ti_val = ti_flat[(base + lag) as usize];
                    let diff = (de_val - de_mean) - (ti_val - tm);
                    acc + wt * (diff / two_dmax).abs().powf(p)
                });
            clamp(inner.powf(inv_p))
        }))
    }
}

/// Fused gather + masked weighted variation distance over TI candidates.
///
/// For each candidate, TI event positions containing `NaN` are excluded. The
/// SG and TI means are both computed over that candidate's common defined
/// support, then the centered powered error is aggregated with the surviving
/// node weights and renormalized:
///
/// `dist[b] = min(1, [Σ(valid weight · |centered_error / (2·d_max)|ᵖ) / Σ(valid weights)]^(1/p))`
///
/// The support means are unweighted; weights apply only to the centered error.
/// Empty support, zero surviving weight, and non-finite aggregates return
/// `+∞`, so the candidate cannot be selected. The production engine applies
/// the accepted-value mean shift separately; this block kernel returns distances.
///
/// # Arguments
///
/// * `de_sim` – Simulation-grid event values, shape `(n,)`.
/// * `ti_flat` – Flattened C-order TI, including possible `NaN`, shape `(ti_size,)`.
/// * `base_flat` – Flat candidate-anchor indices, shape `(B,)`.
/// * `lag_flat` – Flat event-lag offsets, shape `(n,)`.
/// * `node_weights` – Precomputed node weights, shape `(n,)`.
/// * `d_max` – TI finite-value range used for normalization. Must be positive.
/// * `p` – Variation Lp exponent. Must be positive.
///
/// # Returns
///
/// Distance array of shape `(B,)` in `[0, 1]`; excluded candidates contain `+∞`.
#[allow(clippy::too_many_arguments)]
pub fn dist_block_variation_masked(
    de_sim: ArrayView1<'_, f64>,
    ti_flat: ArrayView1<'_, f64>,
    base_flat: ArrayView1<'_, i64>,
    lag_flat: ArrayView1<'_, i64>,
    node_weights: ArrayView1<'_, f64>,
    d_max: f64,
    p: f64,
) -> Array1<f64> {
    let two_dmax = 2.0 * d_max;
    let inv_p = p.recip();

    Array1::from_iter(base_flat.iter().map(|&base| {
        let (de_sum, ti_sum, support_count) = de_sim.iter().zip(lag_flat.iter()).fold(
            (0.0_f64, 0.0_f64, 0_usize),
            |(de_sum, ti_sum, count), (&de_val, &lag)| {
                let ti_val = ti_flat[(base + lag) as usize];
                if ti_val.is_nan() {
                    (de_sum, ti_sum, count)
                } else {
                    (de_sum + de_val, ti_sum + ti_val, count + 1)
                }
            },
        );
        if support_count == 0 {
            return f64::INFINITY;
        }
        let count = support_count as f64;
        let de_mean = de_sum / count;
        let ti_mean = ti_sum / count;

        let (aggregate, weight_sum) = de_sim
            .iter()
            .zip(lag_flat.iter())
            .zip(node_weights.iter())
            .fold(
                (0.0_f64, 0.0_f64),
                |(aggregate, weight_sum), ((&de_val, &lag), &weight)| {
                    let ti_val = ti_flat[(base + lag) as usize];
                    if ti_val.is_nan() {
                        (aggregate, weight_sum)
                    } else {
                        let centered = ((de_val - de_mean) - (ti_val - ti_mean)) / two_dmax;
                        let contribution = if p == 2.0 {
                            centered * centered
                        } else if p == 1.0 {
                            centered.abs()
                        } else {
                            centered.abs().powf(p)
                        };
                        (aggregate + weight * contribution, weight_sum + weight)
                    }
                },
            );
        if weight_sum <= 0.0 || !weight_sum.is_finite() {
            return f64::INFINITY;
        }
        let normalized = aggregate / weight_sum;
        let distance = if p == 2.0 {
            normalized.sqrt()
        } else if p == 1.0 {
            normalized
        } else {
            normalized.powf(inv_p)
        };
        if distance.is_finite() {
            distance.min(1.0)
        } else {
            f64::INFINITY
        }
    }))
}

/// Full DSBC/DS scan for one simulation node — univariate categorical case.
///
/// Combines the Python `_scan_window` outer loop with `_dist_block` into a single
/// Rust function. Iterates over candidate anchors in scan order (random start,
/// sequential wrap-around) and computes the weighted categorical distance for each.
/// Returns the TI coordinates of the best match, or `None` when no candidate is
/// selectable: an empty or zero-sized window, a zero scan budget, or every
/// visited candidate scoring `f64::INFINITY` (e.g. an all-undefined masked
/// window).
///
/// # Arguments
///
/// * `lo` – Lower-left anchor of the TI search window (TI coordinates), shape `(dim,)`.
/// * `win_shape` – Search window shape, shape `(dim,)`.
/// * `start` – Flat index into the window to start scanning from.
///   Computed in Python as `int(u_start_i * win_size)`.
/// * `max_scan` – Maximum candidates to evaluate, capped to the window size.
/// * `threshold` – DSBC: `0` (accept a zero-distance match early);
///   DS: `> 0` (accept first `d < threshold`).
/// * `de_sim` – Data-event categorical values at the sim node, shape `(n,)`.
/// * `ti_flat` – Flattened TI (categories as `f64`), shape `(ti_size,)`.
/// * `ti_strides` – C-order strides into the TI, shape `(dim,)`.
/// * `lag_flat` – Flat lag offsets, shape `(n,)`.
/// * `node_weights` – Precomputed weights, shape `(n,)`.
///
/// # Returns
///
/// `Some(coords)` where `coords` has shape `(dim,)` with the TI coordinates of the
/// best-match anchor, or `None` for an empty/zero-sized window, a zero scan
/// budget, or when every visited candidate has `f64::INFINITY` distance.
#[allow(clippy::too_many_arguments)]
pub fn scan_node_categorical(
    lo: ArrayView1<'_, i64>,
    win_shape: ArrayView1<'_, i64>,
    start: usize,
    max_scan: usize,
    threshold: f64,
    de_sim: ArrayView1<'_, f64>,
    ti_flat: ArrayView1<'_, f64>,
    ti_strides: ArrayView1<'_, i64>,
    lag_flat: ArrayView1<'_, i64>,
    node_weights: ArrayView1<'_, f64>,
) -> Option<Array1<i64>> {
    let dim = lo.len();
    // A degenerate or empty window would abort on `% win_size` or
    // `dim - 1`; return "no match" instead.
    if dim == 0 {
        return None;
    }
    let win_shape_u: Vec<usize> = win_shape.iter().map(|&s| s as usize).collect();
    let win_size: usize = win_shape_u.iter().product();
    if win_size == 0 {
        return None;
    }

    // C-order strides into the window (not the TI)
    let mut win_strides = vec![1usize; dim];
    for d in (0..dim - 1).rev() {
        win_strides[d] = win_strides[d + 1] * win_shape_u[d + 1];
    }

    let dsbc = threshold <= 0.0;
    let mut best_d = f64::INFINITY;
    let mut best_coords: Option<Array1<i64>> = None;

    for k in 0..max_scan.min(win_size) {
        let pos = ((start % win_size) + k) % win_size;
        // Decode flat window position into dim-D TI coordinates
        let mut coords = Array1::<i64>::zeros(dim);
        let mut rem = pos;
        for d in 0..dim {
            let c = (rem / win_strides[d]) as i64;
            coords[d] = lo[d] + c;
            rem %= win_strides[d];
        }
        // Flat base index in TI for this candidate anchor
        let base: i64 = coords
            .iter()
            .zip(ti_strides.iter())
            .map(|(&c, &s)| c * s)
            .sum();

        // Weighted categorical distance for this candidate
        let d = de_sim
            .iter()
            .zip(lag_flat.iter())
            .zip(node_weights.iter())
            .fold(0.0_f64, |acc, ((&de_val, &lag), &wt)| {
                let ti_val = ti_flat[(base + lag) as usize];
                acc + if de_val != ti_val { wt } else { 0.0 }
            });

        // DSBC: accept exact match immediately; DS: accept first under threshold
        if (dsbc && d == 0.0) || (!dsbc && d < threshold) {
            return Some(coords);
        }
        if d < best_d {
            best_d = d;
            best_coords = Some(coords);
        }
    }
    best_coords
}

const METRIC_CATEGORICAL: i64 = 0;
const METRIC_LP: i64 = 1;
const METRIC_VARIATION: i64 = 2;
const METRIC_CATEGORICAL_PENALTY: i64 = 3;

/// One step of a SplitMix64 PRNG; good avalanche, no low-bit structure.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform integer in `[0, bound)` via Lemire's method with rejection; unbiased
/// (no modulo bias), `bound > 0`.
#[inline]
fn bounded_u64(state: &mut u64, bound: u64) -> u64 {
    let mut product = (splitmix64(state) as u128) * (bound as u128);
    let mut low = product as u64;
    if low < bound {
        let threshold = bound.wrapping_neg() % bound;
        while low < threshold {
            product = (splitmix64(state) as u128) * (bound as u128);
            low = product as u64;
        }
    }
    (product >> 64) as u64
}

/// `count` DISTINCT window positions in `[0, win_size)`, uniformly sampled
/// WITHOUT replacement in a node-local order keyed on `scan_seed`.
///
/// Partial Fisher–Yates over the virtual window `[0, win_size)`, with a sparse
/// map holding only the entries that were swapped, so memory and time are
/// `O(count)` (the scan budget) — never a full-window permutation. Because the
/// first draw is uniform over the whole window and each subsequent draw is
/// uniform over the remaining positions, the generated *prefix* of any length
/// is an unbiased sample without replacement; every position can appear first.
/// Determinism is independent of thread scheduling.
fn random_scan_positions(win_size: usize, scan_seed: u64, count: usize) -> Vec<usize> {
    let count = count.min(win_size);
    // Mix the node seed once so a zero seed is not a degenerate SplitMix start.
    let mut state = scan_seed ^ 0xD1B5_4A32_D192_ED03;
    // Sparse view of the virtual array a[k] = value currently at slot k; a slot
    // absent from the map still holds its identity value k.
    let mut swapped: std::collections::HashMap<usize, usize> = HashMap::with_capacity(count);
    let value_at = |swapped: &HashMap<usize, usize>, slot: usize| -> usize {
        swapped.get(&slot).copied().unwrap_or(slot)
    };
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        // Draw slot j uniformly in [i, win_size); emit a[j], move a[i] into j.
        let j = i + bounded_u64(&mut state, (win_size - i) as u64) as usize;
        let drawn = value_at(&swapped, j);
        let displaced = value_at(&swapped, i);
        out.push(drawn);
        // Only slot j is read again (future i' > i never revisit slots < i').
        swapped.insert(j, displaced);
    }
    out
}

/// Penalty cost `T[sim_code, ti_code]` for one flattened `ncat x ncat` matrix.
///
/// Codes are read from `f64` event/TI values and are validated on the Python
/// side (`_check_category_codes`); an out-of-range code here is treated as a
/// full mismatch (cost `1.0`) instead of panicking (release profile aborts on
/// panic).
#[inline]
fn penalty_cost(penalty: &[f64], offset: usize, ncat: usize, sim_val: f64, ti_val: f64) -> f64 {
    let sim_code = sim_val as usize;
    let ti_code = ti_val as usize;
    if sim_val < 0.0 || ti_val < 0.0 || sim_code >= ncat || ti_code >= ncat {
        return 1.0;
    }
    penalty[offset + sim_code * ncat + ti_code]
}

/// Weighted categorical **penalty** distance for one candidate.
///
/// Like [`categorical_candidate_distance`] but replaces the 0/1 mismatch with
/// the asymmetric cost `T[sim_code, ti_code]` (zero diagonal, off-diagonal in
/// `(0, 1]`). `penalty` is the variable's flattened `ncat x ncat` matrix.
#[inline]
#[allow(clippy::too_many_arguments)]
fn categorical_penalty_candidate_distance(
    base: i64,
    de_sim: ArrayView1<'_, f64>,
    ti_flat: ArrayView1<'_, f64>,
    lag_flat: ArrayView1<'_, i64>,
    node_weights: ArrayView1<'_, f64>,
    penalty: &[f64],
    offset: usize,
    ncat: usize,
    masked: bool,
) -> f64 {
    if masked {
        let (cost, weight_sum) = de_sim
            .iter()
            .zip(lag_flat.iter())
            .zip(node_weights.iter())
            .fold(
                (0.0_f64, 0.0_f64),
                |(cost, weight_sum), ((&de_val, &lag), &weight)| {
                    let ti_val = ti_flat[(base + lag) as usize];
                    if ti_val.is_nan() {
                        (cost, weight_sum)
                    } else {
                        (
                            cost + weight * penalty_cost(penalty, offset, ncat, de_val, ti_val),
                            weight_sum + weight,
                        )
                    }
                },
            );
        if weight_sum > 0.0 {
            cost / weight_sum
        } else {
            f64::INFINITY
        }
    } else {
        de_sim
            .iter()
            .zip(lag_flat.iter())
            .zip(node_weights.iter())
            .fold(0.0_f64, |distance, ((&de_val, &lag), &weight)| {
                let ti_val = ti_flat[(base + lag) as usize];
                distance + weight * penalty_cost(penalty, offset, ncat, de_val, ti_val)
            })
    }
}

#[allow(clippy::too_many_arguments)]
fn continuous_candidate_distance(
    base: i64,
    de_sim: ArrayView1<'_, f64>,
    ti_flat: ArrayView1<'_, f64>,
    lag_flat: ArrayView1<'_, i64>,
    node_weights: ArrayView1<'_, f64>,
    d_max: f64,
    p: f64,
    masked: bool,
) -> f64 {
    let (aggregate, weight_sum) = de_sim
        .iter()
        .zip(lag_flat.iter())
        .zip(node_weights.iter())
        .fold(
            (0.0_f64, 0.0_f64),
            |(aggregate, weight_sum), ((&de_val, &lag), &weight)| {
                let ti_val = ti_flat[(base + lag) as usize];
                if masked && ti_val.is_nan() {
                    (aggregate, weight_sum)
                } else {
                    let weighted_contribution = if p == 1.0 {
                        weight * (de_val - ti_val).abs() / d_max
                    } else if p == 2.0 {
                        let diff = (de_val - ti_val) / d_max;
                        weight * diff * diff
                    } else {
                        let diff = (de_val - ti_val).abs() / d_max;
                        weight * diff.powf(p)
                    };
                    (aggregate + weighted_contribution, weight_sum + weight)
                }
            },
        );

    let normalized = if masked {
        if weight_sum <= 0.0 {
            return f64::INFINITY;
        }
        aggregate / weight_sum
    } else {
        aggregate
    };
    if p == 1.0 {
        normalized
    } else if p == 2.0 {
        normalized.sqrt()
    } else {
        normalized.powf(p.recip())
    }
}

#[allow(clippy::too_many_arguments)]
fn variation_candidate_distance(
    base: i64,
    de_sim: ArrayView1<'_, f64>,
    ti_flat: ArrayView1<'_, f64>,
    lag_flat: ArrayView1<'_, i64>,
    node_weights: ArrayView1<'_, f64>,
    d_max: f64,
    p: f64,
    masked: bool,
) -> f64 {
    let (de_sum, ti_sum, support_count) = de_sim.iter().zip(lag_flat.iter()).fold(
        (0.0_f64, 0.0_f64, 0_usize),
        |(de_sum, ti_sum, count), (&de_val, &lag)| {
            let ti_val = ti_flat[(base + lag) as usize];
            if masked && ti_val.is_nan() {
                (de_sum, ti_sum, count)
            } else {
                (de_sum + de_val, ti_sum + ti_val, count + 1)
            }
        },
    );
    if support_count == 0 {
        return f64::INFINITY;
    }
    let count = support_count as f64;
    let de_mean = de_sum / count;
    let ti_mean = ti_sum / count;
    let two_dmax = 2.0 * d_max;

    let (aggregate, weight_sum) = de_sim
        .iter()
        .zip(lag_flat.iter())
        .zip(node_weights.iter())
        .fold(
            (0.0_f64, 0.0_f64),
            |(aggregate, weight_sum), ((&de_val, &lag), &weight)| {
                let ti_val = ti_flat[(base + lag) as usize];
                if masked && ti_val.is_nan() {
                    (aggregate, weight_sum)
                } else {
                    let centered = ((de_val - de_mean) - (ti_val - ti_mean)) / two_dmax;
                    let contribution = if p == 1.0 {
                        centered.abs()
                    } else if p == 2.0 {
                        centered * centered
                    } else {
                        centered.abs().powf(p)
                    };
                    (aggregate + weight * contribution, weight_sum + weight)
                }
            },
        );

    let normalized = if masked {
        if weight_sum <= 0.0 || !weight_sum.is_finite() {
            return f64::INFINITY;
        }
        aggregate / weight_sum
    } else {
        aggregate
    };
    let distance = if p == 1.0 {
        normalized
    } else if p == 2.0 {
        normalized.sqrt()
    } else {
        normalized.powf(p.recip())
    };
    if masked {
        if distance.is_finite() {
            distance.min(1.0)
        } else {
            f64::INFINITY
        }
    } else if distance.is_nan() {
        distance
    } else {
        distance.min(1.0)
    }
}

/// Full serial DSBC/DS scan for one node with any supported variable mixture.
///
/// Candidate order, threshold acceptance, first-in-scan-order tie handling,
/// masked support, multivariate weighting, and target-centre exclusion are
/// evaluated inside Rust. This raw helper accepts prepared events. The production
/// engine also owns event gathering, fallback selection, value retrieval,
/// variation mean shifting, and simulation-grid writes; Python supplies run
/// controls, geometry and pre-generated random draws.
///
/// Per-variable events are stored in concatenated one-dimensional arrays.
/// `event_offsets[v]..event_offsets[v + 1]` selects variable `v` in
/// `de_sim`, `lag_flat`, and `node_weights`. `metric_kinds` uses `0` for
/// categorical, `1` for continuous Lp, `2` for variation distance, and `3` for
/// categorical with a per-variable penalty matrix.
///
/// # Arguments
///
/// * `lo` – Lower-left TI coordinate of the common search window, shape `(dim,)`.
/// * `win_shape` – Common search-window shape, shape `(dim,)`.
/// * `start` – First flat C-order position in the search window.
/// * `max_scan` – Maximum candidates to visit, capped to the window size.
/// * `threshold` – DSBC when zero; otherwise strict DS acceptance `d < threshold`.
/// * `ti_flat` – All TI variables as contiguous rows, shape `(n_variables, ti_size)`.
/// * `ti_strides` – C-order strides of the shared TI shape, shape `(dim,)`.
/// * `active_ti_rows` – TI row index for every active event variable, shape `(v,)`.
/// * `event_offsets` – Offsets into concatenated event arrays, shape `(v + 1,)`.
/// * `de_sim` – Concatenated simulation-grid event values.
/// * `lag_flat` – Concatenated flat TI lag offsets.
/// * `node_weights` – Concatenated normalized event-node weights.
/// * `variable_weights` – Global variable weights for the active variables.
/// * `metric_kinds` – Metric code per active variable (`0`, `1`, `2`, or `3`).
/// * `has_nan` – Nonzero when the active TI variable uses masked semantics.
/// * `d_max` – Continuous normalization range per active variable.
/// * `p_norm` – Lp or variation exponent per active variable.
/// * `random_scan` – When true, visit a bounded, distinct, node-local random
///   candidate prefix instead of the sequential wrap-around order.
/// * `scan_seed` – Node-local seed for the random-scan prefix.
/// * `penalty_flat` – Concatenated row-major `C x C` penalty matrices for the
///   penalty (`metric_kind == 3`) variables.
/// * `penalty_offsets` – Start offset into `penalty_flat` per active variable.
/// * `penalty_ncat` – Category count `C` per active variable (`0` when unused).
/// * `target_ti_rows` – TI rows whose candidate-centre values will be copied.
/// * `check_centers` – Whether target centres must all be finite.
///
/// # Returns
///
/// The first accepted or best TI anchor in scan order. Returns `None` when no
/// candidate has a selectable distance, including an all-invalid masked
/// window.
#[allow(clippy::too_many_arguments)]
pub fn scan_node(
    lo: ArrayView1<'_, i64>,
    win_shape: ArrayView1<'_, i64>,
    start: usize,
    max_scan: usize,
    threshold: f64,
    ti_flat: ArrayView2<'_, f64>,
    ti_strides: ArrayView1<'_, i64>,
    active_ti_rows: ArrayView1<'_, i64>,
    event_offsets: ArrayView1<'_, i64>,
    de_sim: ArrayView1<'_, f64>,
    lag_flat: ArrayView1<'_, i64>,
    node_weights: ArrayView1<'_, f64>,
    variable_weights: ArrayView1<'_, f64>,
    metric_kinds: ArrayView1<'_, i64>,
    has_nan: ArrayView1<'_, u8>,
    d_max: ArrayView1<'_, f64>,
    p_norm: ArrayView1<'_, f64>,
    random_scan: bool,
    scan_seed: u64,
    penalty_flat: ArrayView1<'_, f64>,
    penalty_offsets: ArrayView1<'_, i64>,
    penalty_ncat: ArrayView1<'_, i64>,
    target_ti_rows: ArrayView1<'_, i64>,
    check_centers: bool,
) -> Option<Array1<i64>> {
    let dim = lo.len();
    // Reject a degenerate or empty window before any `% win_size`
    // (division by zero) or `dim - 1` (underflow) can abort the process.
    if dim == 0 {
        return None;
    }
    let win_shape_u: Vec<usize> = win_shape.iter().map(|&size| size as usize).collect();
    let win_size: usize = win_shape_u.iter().product();
    if win_size == 0 {
        return None;
    }
    let mut win_strides = vec![1_usize; dim];
    for axis in (0..dim - 1).rev() {
        win_strides[axis] = win_strides[axis + 1] * win_shape_u[axis + 1];
    }

    let dsbc = threshold <= 0.0;
    let active_weight_total: f64 = variable_weights.sum();
    // Copy to an owned buffer so a non-contiguous (strided) penalty array does
    // not abort via `.as_slice().expect(...)`; the matrix is tiny.
    let penalty_owned = penalty_flat.to_vec();
    let penalty_all = penalty_owned.as_slice();
    // Every active penalty variable's CxC block must fit the supplied buffer.
    for variable in 0..active_ti_rows.len() {
        if metric_kinds[variable] == METRIC_CATEGORICAL_PENALTY {
            let ncat = penalty_ncat[variable] as usize;
            let needed =
                (penalty_offsets[variable] as usize).saturating_add(ncat.saturating_mul(ncat));
            if needed > penalty_all.len() {
                return None;
            }
        }
    }
    let mut best_distance = f64::INFINITY;
    let mut best_coords: Option<Vec<i64>> = None;

    let max_scan = max_scan.min(win_size);

    // Random scan: effective-budget distinct positions, capped to the window,
    // in a node-local order keyed on this node's seed.
    let random_positions = if random_scan {
        random_scan_positions(win_size, scan_seed, max_scan)
    } else {
        Vec::new()
    };

    // scan_index indexes random_positions in random mode and drives the
    // wrap-around walk in sequential mode, so the range loop is intentional.
    #[allow(clippy::needless_range_loop)]
    for scan_index in 0..max_scan {
        let position = if random_scan {
            random_positions[scan_index]
        } else {
            ((start % win_size) + scan_index) % win_size
        };
        let mut coords = vec![0_i64; dim];
        let mut remainder = position;
        for axis in 0..dim {
            let local = remainder / win_strides[axis];
            coords[axis] = lo[axis] + local as i64;
            remainder %= win_strides[axis];
        }
        let base: i64 = coords
            .iter()
            .zip(ti_strides.iter())
            .map(|(&coordinate, &stride)| coordinate * stride)
            .sum();

        if check_centers
            && target_ti_rows
                .iter()
                .any(|&row| !ti_flat[(row as usize, base as usize)].is_finite())
        {
            continue;
        }

        let mut distance = 0.0_f64;
        for variable in 0..active_ti_rows.len() {
            // Skip zero-weight variables before evaluating the metric: an
            // all-undefined masked event returns +inf, and 0 * inf is
            // NaN, which would poison the best-candidate/threshold comparison.
            if variable_weights[variable] == 0.0 {
                continue;
            }
            let event_start = event_offsets[variable] as usize;
            let event_end = event_offsets[variable + 1] as usize;
            let de_var = de_sim.slice(s![event_start..event_end]);
            let lag_var = lag_flat.slice(s![event_start..event_end]);
            let weight_var = node_weights.slice(s![event_start..event_end]);
            let ti_var = ti_flat.row(active_ti_rows[variable] as usize);
            let masked = has_nan[variable] != 0;
            let variable_distance = match metric_kinds[variable] {
                METRIC_CATEGORICAL => {
                    if masked {
                        categorical_masked_candidate_distance(
                            base, de_var, ti_var, lag_var, weight_var,
                        )
                    } else {
                        categorical_candidate_distance(base, de_var, ti_var, lag_var, weight_var)
                    }
                }
                METRIC_CATEGORICAL_PENALTY => categorical_penalty_candidate_distance(
                    base,
                    de_var,
                    ti_var,
                    lag_var,
                    weight_var,
                    penalty_all,
                    penalty_offsets[variable] as usize,
                    penalty_ncat[variable] as usize,
                    masked,
                ),
                METRIC_LP => continuous_candidate_distance(
                    base,
                    de_var,
                    ti_var,
                    lag_var,
                    weight_var,
                    d_max[variable],
                    p_norm[variable],
                    masked,
                ),
                METRIC_VARIATION => variation_candidate_distance(
                    base,
                    de_var,
                    ti_var,
                    lag_var,
                    weight_var,
                    d_max[variable],
                    p_norm[variable],
                    masked,
                ),
                _ => f64::INFINITY,
            };
            distance += variable_weights[variable] * variable_distance;
        }
        if active_weight_total > 0.0 && active_weight_total < 1.0 {
            distance /= active_weight_total;
        }

        if (dsbc && distance <= 0.0) || (!dsbc && distance < threshold) {
            return Some(Array1::from_vec(coords));
        }
        if distance < best_distance {
            best_distance = distance;
            best_coords = Some(coords);
        }
    }

    best_coords.map(Array1::from_vec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn sampler_prefixes_cover_permutations_and_cap_budgets() {
        for size in [1, 2, 3, 4, 7, 8, 10, 31] {
            for seed in [0, 1, u64::MAX, 0x8000_0000_0000_0000] {
                let full = random_scan_positions(size, seed, size);
                let mut sorted = full.clone();
                sorted.sort_unstable();
                assert_eq!(sorted, (0..size).collect::<Vec<_>>());
                assert_eq!(full, random_scan_positions(size, seed, size + 7));
                for count in 0..=size {
                    assert_eq!(full[..count], random_scan_positions(size, seed, count));
                }
            }
        }
        assert!(random_scan_positions(0, 0, 5).is_empty());
    }

    #[test]
    fn sampler_ordered_prefix_frequency_regression() {
        // Fixed sample and predeclared generous tolerance: every ordered triple
        // of four distinct positions has probability 1/24. This detects gross
        // prefix bias, not a proof of PRNG uniformity or stream independence.
        let mut counts = std::collections::HashMap::new();
        for seed in 0..24_000 {
            *counts.entry(random_scan_positions(4, seed, 3)).or_insert(0) += 1;
        }
        assert_eq!(counts.len(), 24);
        assert!(counts.values().all(|&n| (800..=1200).contains(&n)));
    }

    #[test]
    fn bounded_integer_rejects_low_product_region() {
        let bound = (1_u64 << 63) + 1;
        let threshold = bound.wrapping_neg() % bound;
        let initial = (0..100)
            .find(|&seed| {
                let mut state = seed;
                ((splitmix64(&mut state) as u128 * bound as u128) as u64) < threshold
            })
            .unwrap();
        let mut expected_state = initial;
        let mut draws = 0;
        let expected = loop {
            let product = splitmix64(&mut expected_state) as u128 * bound as u128;
            draws += 1;
            if product as u64 >= threshold {
                break (product >> 64) as u64;
            }
        };
        assert!(draws > 1);
        let mut state = initial;
        assert_eq!(bounded_u64(&mut state, bound), expected);
        assert_eq!(state, expected_state);
    }

    #[test]
    fn bounded_integer_edges_stay_in_range() {
        for bound in [1, 2, 3, 7, 8, u64::MAX / 2 + 1, u64::MAX] {
            let mut state = u64::MAX;
            for _ in 0..1000 {
                assert!(bounded_u64(&mut state, bound) < bound);
            }
        }
    }

    #[test]
    fn block_validator_checks_combined_signed_indices_and_lengths() {
        let event = array![0.];
        let ti = array![0.];
        let weight = array![1.];
        assert!(validate_block_inputs(
            event.view(),
            ti.view(),
            array![-1].view(),
            array![1].view(),
            weight.view(),
            false,
            None
        )
        .is_ok());
        assert!(validate_block_inputs(
            event.view(),
            ti.view(),
            array![i64::MAX].view(),
            array![1].view(),
            weight.view(),
            false,
            None
        )
        .is_err());
        assert!(validate_block_inputs(
            event.view(),
            ti.view(),
            array![0].view(),
            array![].view(),
            weight.view(),
            false,
            None
        )
        .is_err());
    }

    #[test]
    fn categorical_exact_match() {
        // TI: [3.0, 7.0], candidate base=0, lag=0 → ti[0]=3.0 == de[0]=3.0 → d=0.0
        let de_sim = array![3.0_f64];
        let ti_flat = array![3.0_f64, 7.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64];
        let weights = array![1.0_f64];
        let d = dist_block_categorical(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
        );
        assert_eq!(d.len(), 1);
        assert_abs_diff_eq!(d[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn categorical_full_mismatch() {
        // base=1: ti[1+0]=7.0 != de[0]=3.0 → d=1.0
        let de_sim = array![3.0_f64];
        let ti_flat = array![3.0_f64, 7.0_f64];
        let base_flat = array![1_i64];
        let lag_flat = array![0_i64];
        let weights = array![1.0_f64];
        let d = dist_block_categorical(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
        );
        assert_abs_diff_eq!(d[0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn categorical_partial_match() {
        // TI=[1,2,3], de=[1,9], base=0, lags=[0,1], weights=[0.5,0.5]
        // lag 0: ti[0]=1 == de[0]=1 → 0 mismatch; lag 1: ti[1]=2 != de[1]=9 → 1 mismatch
        // d = 0.0*0.5 + 1.0*0.5 = 0.5
        let de_sim = array![1.0_f64, 9.0_f64];
        let ti_flat = array![1.0_f64, 2.0_f64, 3.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64];
        let weights = array![0.5_f64, 0.5_f64];
        let d = dist_block_categorical(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
        );
        assert_abs_diff_eq!(d[0], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn categorical_close_distinct_labels_are_mismatch() {
        // Categorical labels need not be integer-spaced: 0.1 and 0.2 are distinct.
        let de_sim = array![0.1_f64];
        let ti_flat = array![0.2_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64];
        let weights = array![1.0_f64];
        let d = dist_block_categorical(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
        );
        assert_abs_diff_eq!(d[0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn categorical_two_candidates() {
        // TI=[5,5,7], de=[5], lag=[0], weights=[1.0]
        // base 0: ti[0]=5 == 5 → d=0.0; base 2: ti[2]=7 != 5 → d=1.0
        let de_sim = array![5.0_f64];
        let ti_flat = array![5.0_f64, 5.0_f64, 7.0_f64];
        let base_flat = array![0_i64, 2_i64];
        let lag_flat = array![0_i64];
        let weights = array![1.0_f64];
        let d = dist_block_categorical(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
        );
        assert_abs_diff_eq!(d[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d[1], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn categorical_rayon_matches_serial_for_thread_counts_and_sizes() {
        let n_lags = 16_usize;
        let de_sim = Array1::from_iter((0..n_lags).map(|index| (index % 3) as f64));
        let lag_flat = Array1::from_iter((0..n_lags).map(|index| index as i64));
        let weights = Array1::from_elem(n_lags, 1.0 / n_lags as f64);

        let mut thread_counts = vec![1_usize, 2, 4];
        thread_counts.push(
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
        );
        thread_counts.sort_unstable();
        thread_counts.dedup();

        for candidate_count in [1_usize, 7, 256, 4095, 4096, 4097] {
            let ti_flat = Array1::from_iter(
                (0..candidate_count * n_lags).map(|index| ((index / 5) % 3) as f64),
            );
            let base_flat =
                Array1::from_iter((0..candidate_count).map(|index| (index * n_lags) as i64));
            let expected = dist_block_categorical(
                de_sim.view(),
                ti_flat.view(),
                base_flat.view(),
                lag_flat.view(),
                weights.view(),
            );

            for &thread_count in &thread_counts {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(thread_count)
                    .build()
                    .expect("test Rayon pool");
                let actual = pool.install(|| {
                    dist_block_categorical_rayon(
                        de_sim.view(),
                        ti_flat.view(),
                        base_flat.view(),
                        lag_flat.view(),
                        weights.view(),
                    )
                });
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn categorical_masked_rayon_matches_serial_for_thread_counts() {
        let n_lags = 12_usize;
        let candidate_count = 513_usize;
        let de_sim = Array1::from_iter((0..n_lags).map(|index| (index % 4) as f64));
        let lag_flat = Array1::from_iter((0..n_lags).map(|index| index as i64));
        let weights = Array1::from_elem(n_lags, 1.0 / n_lags as f64);
        let ti_flat = Array1::from_iter((0..candidate_count * n_lags).map(|index| {
            if index % 17 == 0 {
                f64::NAN
            } else {
                ((index / 7) % 4) as f64
            }
        }));
        let base_flat =
            Array1::from_iter((0..candidate_count).map(|index| (index * n_lags) as i64));
        let expected = dist_block_categorical_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
        );

        let mut thread_counts = vec![1_usize, 2, 4];
        thread_counts.push(
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
        );
        thread_counts.sort_unstable();
        thread_counts.dedup();

        for thread_count in thread_counts {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build()
                .expect("test Rayon pool");
            let actual = pool.install(|| {
                dist_block_categorical_masked_rayon(
                    de_sim.view(),
                    ti_flat.view(),
                    base_flat.view(),
                    lag_flat.view(),
                    weights.view(),
                )
            });
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn categorical_negative_lag() {
        // TI=[1,2,3], de=[2], base=2, lag=-1 → ti[2-1]=ti[1]=2 == 2 → d=0.0
        let de_sim = array![2.0_f64];
        let ti_flat = array![1.0_f64, 2.0_f64, 3.0_f64];
        let base_flat = array![2_i64];
        let lag_flat = array![-1_i64];
        let weights = array![1.0_f64];
        let d = dist_block_categorical(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
        );
        assert_abs_diff_eq!(d[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn categorical_masked_renormalizes_each_candidate() {
        let de_sim = array![0.1_f64, 0.2_f64, 0.3_f64];
        let ti_flat = array![
            0.1_f64,
            f64::NAN,
            0.4_f64,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            0.2_f64,
            0.2_f64,
            0.3_f64,
        ];
        let base_flat = array![0_i64, 3_i64, 6_i64];
        let lag_flat = array![0_i64, 1_i64, 2_i64];
        let weights = array![0.2_f64, 0.3_f64, 0.5_f64];
        let d = dist_block_categorical_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
        );
        // Candidate 0: only positions 0 and 2 survive. Position 2 mismatches,
        // so distance = 0.5 / (0.2 + 0.5) = 5/7.
        assert_abs_diff_eq!(d[0], 5.0 / 7.0, epsilon = 1e-12);
        assert!(d[1].is_infinite());
        // Candidate 2 differs only at the first close numeric label.
        assert_abs_diff_eq!(d[2], 0.2, epsilon = 1e-12);
    }

    #[test]
    fn categorical_masked_zero_surviving_weight_is_infinite() {
        let de_sim = array![1.0_f64, 1.0_f64];
        let ti_flat = array![2.0_f64, f64::NAN];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64];
        let weights = array![0.0_f64, 1.0_f64];
        let d = dist_block_categorical_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
        );
        assert!(d[0].is_infinite());
    }

    #[test]
    fn categorical_masked_matches_unmasked_without_nan() {
        let de_sim = array![0.1_f64, 0.2_f64, 0.3_f64];
        let ti_flat = array![0.2_f64, 0.2_f64, 0.4_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64, 2_i64];
        let weights = array![0.25_f64, 0.25_f64, 0.5_f64];
        let plain = dist_block_categorical(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
        );
        let masked = dist_block_categorical_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
        );
        assert_abs_diff_eq!(masked[0], plain[0], epsilon = 1e-12);
    }

    #[test]
    fn random_scan_positions_are_distinct_and_in_range() {
        use std::collections::HashSet;
        // Full window: a permutation of [0, win_size).
        let full = random_scan_positions(10, 12345, 10);
        assert_eq!(full.len(), 10);
        assert_eq!(full.iter().copied().collect::<HashSet<_>>().len(), 10);
        assert!(full.iter().all(|&p| p < 10));
        // Non-power-of-two window, partial budget: still distinct + in range.
        let partial = random_scan_positions(7, 999, 5);
        assert_eq!(partial.len(), 5);
        assert_eq!(partial.iter().copied().collect::<HashSet<_>>().len(), 5);
        assert!(partial.iter().all(|&p| p < 7));
        // Deterministic for a fixed seed; different seeds generally differ.
        assert_eq!(
            random_scan_positions(64, 42, 8),
            random_scan_positions(64, 42, 8)
        );
        assert_ne!(
            random_scan_positions(64, 42, 8),
            random_scan_positions(64, 43, 8)
        );
    }

    #[test]
    fn random_scan_first_position_is_unbiased() {
        // A 4-cell window with a budget of 1 must be able to
        // pick every position first. The old power-of-two LCG produced only
        // 1 or 2 mod 4 first, never 0 or 3.
        use std::collections::HashSet;
        let firsts: HashSet<usize> = (0..64u64)
            .map(|seed| random_scan_positions(4, seed, 1)[0])
            .collect();
        assert_eq!(firsts, HashSet::from([0, 1, 2, 3]));
    }

    #[test]
    fn penalty_cost_uses_asymmetric_matrix_orientation() {
        // T[sim, ti]; asymmetric so orientation is observable.
        let penalty = [0.0, 0.5, 0.3, 0.0];
        // sim=1, ti=0 -> row 1, col 0 -> 0.3 (not 0.5).
        assert_abs_diff_eq!(penalty_cost(&penalty, 0, 2, 1.0, 0.0), 0.3);
        assert_abs_diff_eq!(penalty_cost(&penalty, 0, 2, 0.0, 1.0), 0.5);
        // diagonal / exact match is free.
        assert_abs_diff_eq!(penalty_cost(&penalty, 0, 2, 1.0, 1.0), 0.0);
        // out-of-range codes fall back to full mismatch 1.0 (no panic).
        assert_abs_diff_eq!(penalty_cost(&penalty, 0, 2, 5.0, 0.0), 1.0);
    }

    #[test]
    fn categorical_penalty_candidate_distance_hand_computed() {
        // de_sim=[0,1], matched TI values both 0 (ti_flat[base+lag]).
        let de_sim = array![0.0_f64, 1.0_f64];
        let ti_flat = array![0.0_f64, 0.0_f64];
        let lag_flat = array![0_i64, 1_i64];
        let weights = array![0.5_f64, 0.5_f64];
        let penalty = [0.0, 0.5, 0.3, 0.0]; // T[sim, ti], ncat=2
                                            // node0: 0.5 * T[0,0]=0 ; node1: 0.5 * T[1,0]=0.5*0.3=0.15 -> 0.15
        let d = categorical_penalty_candidate_distance(
            0,
            de_sim.view(),
            ti_flat.view(),
            lag_flat.view(),
            weights.view(),
            &penalty,
            0,
            2,
            false,
        );
        assert_abs_diff_eq!(d, 0.15, epsilon = 1e-12);
    }

    #[test]
    fn categorical_penalty_all_ones_matches_plain_mismatch() {
        // T with off-diagonal 1.0 reproduces the plain 0/1 categorical distance.
        let de_sim = array![0.0_f64, 1.0_f64, 1.0_f64];
        let ti_flat = array![0.0_f64, 0.0_f64, 1.0_f64];
        let lag_flat = array![0_i64, 1_i64, 2_i64];
        let weights = array![0.25_f64, 0.25_f64, 0.5_f64];
        let penalty = [0.0, 1.0, 1.0, 0.0];
        let pen = categorical_penalty_candidate_distance(
            0,
            de_sim.view(),
            ti_flat.view(),
            lag_flat.view(),
            weights.view(),
            &penalty,
            0,
            2,
            false,
        );
        let plain = categorical_candidate_distance(
            0,
            de_sim.view(),
            ti_flat.view(),
            lag_flat.view(),
            weights.view(),
        );
        assert_abs_diff_eq!(pen, plain, epsilon = 1e-12);
    }

    #[test]
    fn l1_single_lag() {
        // |5.0 - 3.0| / 4.0 * 1.0 = 0.5
        let de_sim = array![5.0_f64];
        let ti_flat = array![3.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64];
        let weights = array![1.0_f64];
        let d = dist_block_l1(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
        );
        assert_abs_diff_eq!(d[0], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn l1_two_candidates() {
        // TI=[0,2,4], de=[4,0], weights=[0.5,0.5], d_max=4.0
        // base 0, lags=[0,1]: ti=[0,2]; |4-0|/4*0.5 + |0-2|/4*0.5 = 0.5 + 0.25 = 0.75
        // base 1, lags=[0,1]: ti=[2,4]; |4-2|/4*0.5 + |0-4|/4*0.5 = 0.25 + 0.5 = 0.75
        let de_sim = array![4.0_f64, 0.0_f64];
        let ti_flat = array![0.0_f64, 2.0_f64, 4.0_f64];
        let base_flat = array![0_i64, 1_i64];
        let lag_flat = array![0_i64, 1_i64];
        let weights = array![0.5_f64, 0.5_f64];
        let d = dist_block_l1(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
        );
        assert_abs_diff_eq!(d[0], 0.75, epsilon = 1e-12);
        assert_abs_diff_eq!(d[1], 0.75, epsilon = 1e-12);
    }

    #[test]
    fn l2_single_lag() {
        // √(1.0 * ((5-3)/4)²) = √(0.25) = 0.5
        let de_sim = array![5.0_f64];
        let ti_flat = array![3.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64];
        let weights = array![1.0_f64];
        let d = dist_block_l2(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
        );
        assert_abs_diff_eq!(d[0], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn l2_masked_renormalizes_each_candidate() {
        let de_sim = array![0.0_f64, 2.0_f64, 4.0_f64];
        let ti_flat = array![2.0_f64, f64::NAN, 0.0_f64, f64::NAN, f64::NAN, f64::NAN];
        let base_flat = array![0_i64, 3_i64];
        let lag_flat = array![0_i64, 1_i64, 2_i64];
        let weights = array![0.2_f64, 0.3_f64, 0.5_f64];
        let d = dist_block_l2_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
        );
        // Candidate 0 keeps lags 0 and 2. Squared weighted error is
        // 0.2*(2/4)^2 + 0.5*(4/4)^2 = 0.55, renormalized by 0.7.
        assert_abs_diff_eq!(d[0], (0.55_f64 / 0.7).sqrt(), epsilon = 1e-12);
        assert!(d[1].is_infinite());
    }

    #[test]
    fn l2_masked_zero_surviving_weight_is_infinite() {
        let de_sim = array![0.0_f64, 2.0_f64];
        let ti_flat = array![1.0_f64, f64::NAN];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64];
        let weights = array![0.0_f64, 1.0_f64];
        let d = dist_block_l2_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
        );
        assert!(d[0].is_infinite());
    }

    #[test]
    fn l2_masked_matches_unmasked_without_nan() {
        let de_sim = array![0.0_f64, 2.0_f64, 4.0_f64];
        let ti_flat = array![2.0_f64, 1.0_f64, 0.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64, 2_i64];
        let weights = array![0.2_f64, 0.3_f64, 0.5_f64];
        let plain = dist_block_l2(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
        );
        let masked = dist_block_l2_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
        );
        assert_eq!(masked, plain);
    }

    #[test]
    fn lp_p1_matches_l1() {
        // lp(p=1) must equal l1 for same inputs
        let de_sim = array![5.0_f64];
        let ti_flat = array![3.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64];
        let weights = array![1.0_f64];
        let d_l1 = dist_block_l1(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
        );
        let d_lp = dist_block_lp(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
            1.0,
        );
        assert_abs_diff_eq!(d_l1[0], d_lp[0], epsilon = 1e-12);
    }

    #[test]
    fn lp_p2_matches_l2() {
        let de_sim = array![5.0_f64];
        let ti_flat = array![3.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64];
        let weights = array![1.0_f64];
        let d_l2 = dist_block_l2(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
        );
        let d_lp = dist_block_lp(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
            2.0,
        );
        assert_abs_diff_eq!(d_l2[0], d_lp[0], epsilon = 1e-10);
    }

    #[test]
    fn lp_masked_general_p_renormalizes_each_candidate() {
        let de_sim = array![0.0_f64, 2.0_f64, 4.0_f64];
        let ti_flat = array![2.0_f64, f64::NAN, 0.0_f64, f64::NAN, f64::NAN, f64::NAN];
        let base_flat = array![0_i64, 3_i64];
        let lag_flat = array![0_i64, 1_i64, 2_i64];
        let weights = array![0.2_f64, 0.3_f64, 0.5_f64];
        let d = dist_block_lp_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
            3.0,
        );
        // Candidate 0 keeps lags 0 and 2. Powered weighted error is
        // 0.2*(2/4)^3 + 0.5*(4/4)^3 = 0.525, renormalized by 0.7.
        assert_abs_diff_eq!(d[0], (0.525_f64 / 0.7).powf(1.0 / 3.0), epsilon = 1e-12);
        assert!(d[1].is_infinite());
    }

    #[test]
    fn lp_masked_p1_and_p2_match_specialized_kernels() {
        let de_sim = array![0.0_f64, 2.0_f64, 4.0_f64];
        let ti_flat = array![2.0_f64, f64::NAN, 0.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64, 2_i64];
        let weights = array![0.2_f64, 0.3_f64, 0.5_f64];
        let l1 = dist_block_l1_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
        );
        let lp1 = dist_block_lp_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
            1.0,
        );
        let l2 = dist_block_l2_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
        );
        let lp2 = dist_block_lp_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
            2.0,
        );
        assert_eq!(lp1, l1);
        assert_eq!(lp2, l2);
    }

    #[test]
    fn lp_masked_zero_surviving_weight_is_infinite() {
        let de_sim = array![0.0_f64, 2.0_f64];
        let ti_flat = array![1.0_f64, f64::NAN];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64];
        let weights = array![0.0_f64, 1.0_f64];
        let d = dist_block_lp_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
            3.0,
        );
        assert!(d[0].is_infinite());
    }

    #[test]
    fn lp_masked_matches_unmasked_without_nan() {
        let de_sim = array![0.0_f64, 2.0_f64, 4.0_f64];
        let ti_flat = array![2.0_f64, 1.0_f64, 0.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64, 2_i64];
        let weights = array![0.2_f64, 0.3_f64, 0.5_f64];
        let plain = dist_block_lp(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
            3.0,
        );
        let masked = dist_block_lp_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
            3.0,
        );
        assert_eq!(masked, plain);
    }

    #[test]
    fn variation_identical_pattern_zero() {
        // de == ti exactly → centered diffs are 0 → distance 0.
        let de_sim = array![0.0_f64, 1.0_f64, 2.0_f64, 1.0_f64];
        let ti_flat = array![0.0_f64, 1.0_f64, 2.0_f64, 1.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64, 2_i64, 3_i64];
        let weights = array![0.25_f64, 0.25_f64, 0.25_f64, 0.25_f64];
        let d = dist_block_variation(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            2.0,
            2.0,
        );
        assert_abs_diff_eq!(d[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn variation_constant_shift_zero() {
        // ti = de + 5 (constant vertical shift) → variation distance 0.
        let de_sim = array![1.0_f64, 2.0_f64, 3.0_f64];
        let ti_flat = array![6.0_f64, 7.0_f64, 8.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64, 2_i64];
        let weights = array![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let d = dist_block_variation(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            2.0,
            2.0,
        );
        assert_abs_diff_eq!(d[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn variation_p1_hand_calc() {
        // de=[0,2] mean=1 → de_c=[-1,1]; ti=[0,4] mean=2 → ti_c=[-2,2].
        // diff = de_c - ti_c = [1,-1]; d_max=1 → 2*d_max=2; |diff/2|=[0.5,0.5].
        // p=1, w=[0.5,0.5]: inner = 0.5*0.5 + 0.5*0.5 = 0.5 → d = 0.5.
        let de_sim = array![0.0_f64, 2.0_f64];
        let ti_flat = array![0.0_f64, 4.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64];
        let weights = array![0.5_f64, 0.5_f64];
        let d = dist_block_variation(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            1.0,
            1.0,
        );
        assert_abs_diff_eq!(d[0], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn variation_p2_hand_calc() {
        // Same inputs as p1 case but p=2:
        // inner = 0.5*0.5^2 + 0.5*0.5^2 = 0.25 → d = sqrt(0.25) = 0.5.
        let de_sim = array![0.0_f64, 2.0_f64];
        let ti_flat = array![0.0_f64, 4.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64];
        let weights = array![0.5_f64, 0.5_f64];
        let d = dist_block_variation(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            1.0,
            2.0,
        );
        assert_abs_diff_eq!(d[0], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn variation_nonuniform_weights() {
        // de=[0,3] mean=1.5 → de_c=[-1.5,1.5]; ti=[0,1] mean=0.5 → ti_c=[-0.5,0.5].
        // diff=[-1,1]; d_max=1 → 2*d_max=2; |diff/2|=[0.5,0.5]; p=1.
        // w=[0.25,0.75]: inner = 0.25*0.5 + 0.75*0.5 = 0.5 → d = 0.5.
        let de_sim = array![0.0_f64, 3.0_f64];
        let ti_flat = array![0.0_f64, 1.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64];
        let weights = array![0.25_f64, 0.75_f64];
        let d = dist_block_variation(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            1.0,
            1.0,
        );
        assert_abs_diff_eq!(d[0], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn variation_clamps_to_one() {
        // de=[0,10] mean=5 → de_c=[-5,5]; ti=[0,0] mean=0 → ti_c=[0,0].
        // diff=[-5,5]; d_max=1 → |diff/2|=[2.5,2.5]; p=1, w=[0.5,0.5]:
        // inner = 2.5 → raw d = 2.5, clamped to 1.0.
        let de_sim = array![0.0_f64, 10.0_f64];
        let ti_flat = array![0.0_f64, 0.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64];
        let weights = array![0.5_f64, 0.5_f64];
        let d = dist_block_variation(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            1.0,
            1.0,
        );
        assert_abs_diff_eq!(d[0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn variation_two_candidates() {
        // TI=[0,2,0,4], de=[0,2] mean=1. d_max=1, p=1, w=[0.5,0.5].
        // base 0, lags=[0,1]: ti=[0,2] mean=1 → ti_c=[-1,1]; de_c=[-1,1]; diff=0 → d=0.
        // base 2, lags=[0,1]: ti=[0,4] mean=2 → ti_c=[-2,2]; diff=[1,-1]; |.|/2=0.5 → d=0.5.
        let de_sim = array![0.0_f64, 2.0_f64];
        let ti_flat = array![0.0_f64, 2.0_f64, 0.0_f64, 4.0_f64];
        let base_flat = array![0_i64, 2_i64];
        let lag_flat = array![0_i64, 1_i64];
        let weights = array![0.5_f64, 0.5_f64];
        let d = dist_block_variation(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            1.0,
            1.0,
        );
        assert_abs_diff_eq!(d[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d[1], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn variation_masked_uses_candidate_common_support() {
        let de_sim = array![0.0_f64, 2.0_f64, 8.0_f64];
        let ti_flat = array![2.0_f64, f64::NAN, 6.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64, 2_i64];
        let weights = array![0.2_f64, 0.3_f64, 0.5_f64];
        let d = dist_block_variation_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            8.0,
            2.0,
        );
        // Common support is lags 0 and 2. SG mean=4, TI mean=4, so centered
        // differences are -2 and +2. Both normalized absolute errors are 1/8.
        assert_abs_diff_eq!(d[0], 0.125, epsilon = 1e-12);
    }

    #[test]
    fn variation_masked_constant_shift_on_common_support_is_zero() {
        let de_sim = array![0.0_f64, 10.0_f64, 20.0_f64];
        let ti_flat = array![5.0_f64, f64::NAN, 25.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64, 2_i64];
        let weights = array![0.2_f64, 0.3_f64, 0.5_f64];
        let d = dist_block_variation_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            25.0,
            2.0,
        );
        assert_abs_diff_eq!(d[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn variation_masked_p1_p2_and_fractional_agree_on_equal_errors() {
        let de_sim = array![0.0_f64, 2.0_f64, 8.0_f64];
        let ti_flat = array![2.0_f64, f64::NAN, 6.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64, 2_i64];
        let weights = array![0.2_f64, 0.3_f64, 0.5_f64];
        for p in [1.0_f64, 1.5, 2.0, 3.0] {
            let d = dist_block_variation_masked(
                de_sim.view(),
                ti_flat.view(),
                base_flat.view(),
                lag_flat.view(),
                weights.view(),
                8.0,
                p,
            );
            assert_abs_diff_eq!(d[0], 0.125, epsilon = 1e-12);
        }
    }

    #[test]
    fn variation_masked_empty_or_zero_weight_support_is_infinite() {
        let de_sim = array![0.0_f64, 2.0_f64];
        let ti_flat = array![f64::NAN, f64::NAN, 1.0_f64, f64::NAN];
        let base_flat = array![0_i64, 2_i64];
        let lag_flat = array![0_i64, 1_i64];
        let weights = array![0.0_f64, 1.0_f64];
        let d = dist_block_variation_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            2.0,
            2.0,
        );
        assert!(d[0].is_infinite());
        assert!(d[1].is_infinite());
    }

    #[test]
    fn variation_masked_matches_unmasked_without_nan() {
        let de_sim = array![0.0_f64, 2.0_f64, 8.0_f64];
        let ti_flat = array![2.0_f64, 3.0_f64, 6.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64, 2_i64];
        let weights = array![0.2_f64, 0.3_f64, 0.5_f64];
        for p in [1.0_f64, 1.5, 2.0] {
            let plain = dist_block_variation(
                de_sim.view(),
                ti_flat.view(),
                base_flat.view(),
                lag_flat.view(),
                weights.view(),
                8.0,
                p,
            );
            let masked = dist_block_variation_masked(
                de_sim.view(),
                ti_flat.view(),
                base_flat.view(),
                lag_flat.view(),
                weights.view(),
                8.0,
                p,
            );
            assert_abs_diff_eq!(masked[0], plain[0], epsilon = 1e-12);
        }
    }

    #[test]
    fn variation_masked_nonfinite_aggregate_is_excluded() {
        let de_sim = array![0.0_f64, 2.0_f64];
        let ti_flat = array![f64::INFINITY, 1.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64];
        let weights = array![0.5_f64, 0.5_f64];
        let d = dist_block_variation_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            2.0,
            2.0,
        );
        assert!(d[0].is_infinite());
    }

    #[test]
    fn l1_masked_renormalizes_each_candidate() {
        let de_sim = array![0.0_f64, 10.0_f64, 20.0_f64];
        let ti_flat = array![
            2.0_f64,
            f64::NAN,
            10.0_f64,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            0.0_f64,
            0.0_f64,
            20.0_f64,
        ];
        let base_flat = array![0_i64, 3_i64, 6_i64];
        let lag_flat = array![0_i64, 1_i64, 2_i64];
        let weights = array![0.2_f64, 0.3_f64, 0.5_f64];
        let d = dist_block_l1_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            20.0,
        );
        // Candidate 0 numerator = 0.2*(2/20) + 0.5*(10/20) = 0.27;
        // surviving weight = 0.7.
        assert_abs_diff_eq!(d[0], 0.27 / 0.7, epsilon = 1e-12);
        assert!(d[1].is_infinite());
        assert_abs_diff_eq!(d[2], 0.15, epsilon = 1e-12);
    }

    #[test]
    fn l1_masked_zero_surviving_weight_is_infinite() {
        let de_sim = array![1.0_f64, 1.0_f64];
        let ti_flat = array![2.0_f64, f64::NAN];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64];
        let weights = array![0.0_f64, 1.0_f64];
        let d = dist_block_l1_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            1.0,
        );
        assert!(d[0].is_infinite());
    }

    #[test]
    fn l1_masked_matches_unmasked_without_nan() {
        let de_sim = array![0.0_f64, 10.0_f64, 20.0_f64];
        let ti_flat = array![2.0_f64, 4.0_f64, 10.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64, 2_i64];
        let weights = array![0.25_f64, 0.25_f64, 0.5_f64];
        let plain = dist_block_l1(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            20.0,
        );
        let masked = dist_block_l1_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            20.0,
        );
        assert_abs_diff_eq!(masked[0], plain[0], epsilon = 1e-12);
    }

    #[test]
    fn variation_nonfinite_input_propagates_nan_like_numpy() {
        // A TI event containing infinity has an infinite mean; centering then
        // produces NaN. NumPy propagates that NaN through its final clamp.
        let de_sim = array![0.0_f64, 1.0_f64];
        let ti_flat = array![0.0_f64, f64::INFINITY];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64];
        let weights = array![0.5_f64, 0.5_f64];
        let d = dist_block_variation(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            1.0,
            2.0,
        );
        assert!(d[0].is_nan());
    }

    #[test]
    fn full_scan_multivariate_uses_one_shared_anchor() {
        let lo = array![0_i64];
        let win_shape = array![4_i64];
        let ti_flat = array![[0.0_f64, 1.0, 1.0, 0.0], [0.0_f64, 9.0, 2.0, 5.0],];
        let ti_strides = array![1_i64];
        let active_rows = array![0_i64, 1_i64];
        let offsets = array![0_i64, 1_i64, 2_i64];
        let de_sim = array![1.0_f64, 2.0_f64];
        let lag_flat = array![0_i64, 0_i64];
        let node_weights = array![1.0_f64, 1.0_f64];
        let variable_weights = array![0.5_f64, 0.5_f64];
        let metric_kinds = array![METRIC_CATEGORICAL, METRIC_LP];
        let has_nan = array![0_u8, 0_u8];
        let d_max = array![1.0_f64, 10.0_f64];
        let p_norm = array![1.0_f64, 1.0_f64];
        let target_rows = array![0_i64, 1_i64];

        let result = scan_node(
            lo.view(),
            win_shape.view(),
            0,
            4,
            0.0,
            ti_flat.view(),
            ti_strides.view(),
            active_rows.view(),
            offsets.view(),
            de_sim.view(),
            lag_flat.view(),
            node_weights.view(),
            variable_weights.view(),
            metric_kinds.view(),
            has_nan.view(),
            d_max.view(),
            p_norm.view(),
            false,
            0,
            Array1::<f64>::zeros(0).view(),
            Array1::<i64>::zeros(metric_kinds.len()).view(),
            Array1::<i64>::zeros(metric_kinds.len()).view(),
            target_rows.view(),
            false,
        )
        .unwrap();
        assert_eq!(result, array![2_i64]);
    }

    #[test]
    fn full_scan_ds_threshold_is_strict() {
        let lo = array![0_i64];
        let win_shape = array![2_i64];
        let ti_flat = array![[0.5_f64, 0.4_f64]];
        let ti_strides = array![1_i64];
        let active_rows = array![0_i64];
        let offsets = array![0_i64, 1_i64];
        let de_sim = array![0.0_f64];
        let lag_flat = array![0_i64];
        let node_weights = array![1.0_f64];
        let variable_weights = array![1.0_f64];
        let metric_kinds = array![METRIC_LP];
        let has_nan = array![0_u8];
        let d_max = array![1.0_f64];
        let p_norm = array![1.0_f64];
        let target_rows = array![0_i64];

        let result = scan_node(
            lo.view(),
            win_shape.view(),
            0,
            2,
            0.5,
            ti_flat.view(),
            ti_strides.view(),
            active_rows.view(),
            offsets.view(),
            de_sim.view(),
            lag_flat.view(),
            node_weights.view(),
            variable_weights.view(),
            metric_kinds.view(),
            has_nan.view(),
            d_max.view(),
            p_norm.view(),
            false,
            0,
            Array1::<f64>::zeros(0).view(),
            Array1::<i64>::zeros(metric_kinds.len()).view(),
            Array1::<i64>::zeros(metric_kinds.len()).view(),
            target_rows.view(),
            false,
        )
        .unwrap();
        assert_eq!(result, array![1_i64]);
    }

    #[test]
    fn full_scan_dsbc_tie_keeps_first_in_scan_order() {
        let lo = array![0_i64];
        let win_shape = array![3_i64];
        let ti_flat = array![[0.0_f64, 1.0_f64, 1.0_f64]];
        let ti_strides = array![1_i64];
        let active_rows = array![0_i64];
        let offsets = array![0_i64, 1_i64];
        let de_sim = array![1.0_f64];
        let lag_flat = array![0_i64];
        let node_weights = array![1.0_f64];
        let variable_weights = array![1.0_f64];
        let metric_kinds = array![METRIC_CATEGORICAL];
        let has_nan = array![0_u8];
        let d_max = array![1.0_f64];
        let p_norm = array![1.0_f64];
        let target_rows = array![0_i64];

        let result = scan_node(
            lo.view(),
            win_shape.view(),
            1,
            2,
            0.0,
            ti_flat.view(),
            ti_strides.view(),
            active_rows.view(),
            offsets.view(),
            de_sim.view(),
            lag_flat.view(),
            node_weights.view(),
            variable_weights.view(),
            metric_kinds.view(),
            has_nan.view(),
            d_max.view(),
            p_norm.view(),
            false,
            0,
            Array1::<f64>::zeros(0).view(),
            Array1::<i64>::zeros(metric_kinds.len()).view(),
            Array1::<i64>::zeros(metric_kinds.len()).view(),
            target_rows.view(),
            false,
        )
        .unwrap();
        assert_eq!(result, array![1_i64]);
    }

    #[test]
    fn full_scan_skips_invalid_target_centres_and_can_return_none() {
        let lo = array![0_i64];
        let win_shape = array![3_i64];
        let ti_flat = array![[1.0_f64, 1.0_f64, 1.0_f64], [f64::NAN, 7.0_f64, 8.0_f64],];
        let ti_strides = array![1_i64];
        let active_rows = array![0_i64];
        let offsets = array![0_i64, 1_i64];
        let de_sim = array![1.0_f64];
        let lag_flat = array![0_i64];
        let node_weights = array![1.0_f64];
        let variable_weights = array![1.0_f64];
        let metric_kinds = array![METRIC_CATEGORICAL];
        let has_nan = array![0_u8];
        let d_max = array![1.0_f64];
        let p_norm = array![1.0_f64];
        let target_rows = array![1_i64];

        let result = scan_node(
            lo.view(),
            win_shape.view(),
            0,
            3,
            0.0,
            ti_flat.view(),
            ti_strides.view(),
            active_rows.view(),
            offsets.view(),
            de_sim.view(),
            lag_flat.view(),
            node_weights.view(),
            variable_weights.view(),
            metric_kinds.view(),
            has_nan.view(),
            d_max.view(),
            p_norm.view(),
            false,
            0,
            Array1::<f64>::zeros(0).view(),
            Array1::<i64>::zeros(metric_kinds.len()).view(),
            Array1::<i64>::zeros(metric_kinds.len()).view(),
            target_rows.view(),
            true,
        )
        .unwrap();
        assert_eq!(result, array![1_i64]);

        let all_invalid_ti = array![[1.0_f64, 1.0_f64, 1.0_f64], [f64::NAN, f64::NAN, f64::NAN],];
        let no_result = scan_node(
            lo.view(),
            win_shape.view(),
            0,
            3,
            0.0,
            all_invalid_ti.view(),
            ti_strides.view(),
            active_rows.view(),
            offsets.view(),
            de_sim.view(),
            lag_flat.view(),
            node_weights.view(),
            variable_weights.view(),
            metric_kinds.view(),
            has_nan.view(),
            d_max.view(),
            p_norm.view(),
            false,
            0,
            Array1::<f64>::zeros(0).view(),
            Array1::<i64>::zeros(metric_kinds.len()).view(),
            Array1::<i64>::zeros(metric_kinds.len()).view(),
            target_rows.view(),
            true,
        );
        assert!(no_result.is_none());
    }

    #[test]
    fn full_scan_supports_l1_l2_lp_and_variation() {
        let lo = array![0_i64];
        let win_shape = array![3_i64];
        let ti_flat = array![[0.0_f64, 0.0_f64, 1.0_f64, 3.0_f64]];
        let ti_strides = array![1_i64];
        let active_rows = array![0_i64];
        let offsets = array![0_i64, 2_i64];
        let de_sim = array![1.0_f64, 3.0_f64];
        let lag_flat = array![0_i64, 1_i64];
        let node_weights = array![0.5_f64, 0.5_f64];
        let variable_weights = array![1.0_f64];
        let has_nan = array![0_u8];
        let d_max = array![3.0_f64];
        let target_rows = array![0_i64];

        for (metric, p) in [
            (METRIC_LP, 1.0_f64),
            (METRIC_LP, 2.0_f64),
            (METRIC_LP, 3.0_f64),
            (METRIC_VARIATION, 1.5_f64),
        ] {
            let metric_kinds = array![metric];
            let p_norm = array![p];
            let result = scan_node(
                lo.view(),
                win_shape.view(),
                0,
                3,
                0.0,
                ti_flat.view(),
                ti_strides.view(),
                active_rows.view(),
                offsets.view(),
                de_sim.view(),
                lag_flat.view(),
                node_weights.view(),
                variable_weights.view(),
                metric_kinds.view(),
                has_nan.view(),
                d_max.view(),
                p_norm.view(),
                false,
                0,
                Array1::<f64>::zeros(0).view(),
                Array1::<i64>::zeros(metric_kinds.len()).view(),
                Array1::<i64>::zeros(metric_kinds.len()).view(),
                target_rows.view(),
                false,
            )
            .unwrap();
            assert_eq!(result, array![2_i64]);
        }
    }

    #[test]
    fn full_scan_masked_event_renormalizes_common_support() {
        let lo = array![0_i64];
        let win_shape = array![2_i64];
        let ti_flat = array![[1.0_f64, f64::NAN, 0.0_f64]];
        let ti_strides = array![1_i64];
        let active_rows = array![0_i64];
        let offsets = array![0_i64, 2_i64];
        let de_sim = array![1.0_f64, 2.0_f64];
        let lag_flat = array![0_i64, 1_i64];
        let node_weights = array![0.25_f64, 0.75_f64];
        let variable_weights = array![1.0_f64];
        let metric_kinds = array![METRIC_LP];
        let has_nan = array![1_u8];
        let d_max = array![2.0_f64];
        let p_norm = array![1.0_f64];
        let target_rows = array![0_i64];

        let result = scan_node(
            lo.view(),
            win_shape.view(),
            0,
            2,
            0.0,
            ti_flat.view(),
            ti_strides.view(),
            active_rows.view(),
            offsets.view(),
            de_sim.view(),
            lag_flat.view(),
            node_weights.view(),
            variable_weights.view(),
            metric_kinds.view(),
            has_nan.view(),
            d_max.view(),
            p_norm.view(),
            false,
            0,
            Array1::<f64>::zeros(0).view(),
            Array1::<i64>::zeros(metric_kinds.len()).view(),
            Array1::<i64>::zeros(metric_kinds.len()).view(),
            target_rows.view(),
            false,
        )
        .unwrap();
        assert_eq!(result, array![0_i64]);
    }

    #[test]
    fn scan_node_cat_finds_exact_match() {
        // TI (1D): [1,2,3,2,1], window lo=[0], shape=[5], de=[2], lag=[0], w=[1.0]
        // DSBC: first exact match at scan position 1 (ti[1]=2) → return [1]
        let lo = array![0_i64];
        let win_shape = array![5_i64];
        let ti_flat = array![1.0_f64, 2.0_f64, 3.0_f64, 2.0_f64, 1.0_f64];
        let ti_strides = array![1_i64];
        let de_sim = array![2.0_f64];
        let lag_flat = array![0_i64];
        let weights = array![1.0_f64];
        let result = scan_node_categorical(
            lo.view(),
            win_shape.view(),
            0,
            5,
            0.0,
            de_sim.view(),
            ti_flat.view(),
            ti_strides.view(),
            lag_flat.view(),
            weights.view(),
        );
        assert!(result.is_some());
        assert_eq!(result.unwrap()[0], 1);
    }

    #[test]
    fn scan_node_cat_returns_best_when_no_exact() {
        // TI: [0,0,1], de=[2], DSBC → no exact match → best is any (all d=1.0) → first scanned
        let lo = array![0_i64];
        let win_shape = array![3_i64];
        let ti_flat = array![0.0_f64, 0.0_f64, 1.0_f64];
        let ti_strides = array![1_i64];
        let de_sim = array![2.0_f64];
        let lag_flat = array![0_i64];
        let weights = array![1.0_f64];
        let result = scan_node_categorical(
            lo.view(),
            win_shape.view(),
            0,
            3,
            0.0,
            de_sim.view(),
            ti_flat.view(),
            ti_strides.view(),
            lag_flat.view(),
            weights.view(),
        );
        assert!(result.is_some());
        assert_eq!(result.unwrap()[0], 0); // first scanned, all equal d=1.0
    }

    #[test]
    fn scan_node_cat_ds_mode_early_exit() {
        // DS mode: threshold=0.6. TI: [0,1,2], de=[1]
        // candidate 0: ti[0]=0 ≠ 1 → d=1.0 (≥0.6, no exit); candidate 1: ti[1]=1 == 1 → d=0.0 < 0.6 → exit
        let lo = array![0_i64];
        let win_shape = array![3_i64];
        let ti_flat = array![0.0_f64, 1.0_f64, 2.0_f64];
        let ti_strides = array![1_i64];
        let de_sim = array![1.0_f64];
        let lag_flat = array![0_i64];
        let weights = array![1.0_f64];
        let result = scan_node_categorical(
            lo.view(),
            win_shape.view(),
            0,
            3,
            0.6,
            de_sim.view(),
            ti_flat.view(),
            ti_strides.view(),
            lag_flat.view(),
            weights.view(),
        );
        assert!(result.is_some());
        assert_eq!(result.unwrap()[0], 1);
    }

    #[test]
    fn scan_node_cat_distinguishes_close_labels() {
        // Candidate 0 (0.2) differs from de=0.1; candidate 1 is the exact match.
        // The old tolerance-based comparison incorrectly accepted candidate 0.
        let lo = array![0_i64];
        let win_shape = array![2_i64];
        let ti_flat = array![0.2_f64, 0.1_f64];
        let ti_strides = array![1_i64];
        let de_sim = array![0.1_f64];
        let lag_flat = array![0_i64];
        let weights = array![1.0_f64];
        let result = scan_node_categorical(
            lo.view(),
            win_shape.view(),
            0,
            2,
            0.0,
            de_sim.view(),
            ti_flat.view(),
            ti_strides.view(),
            lag_flat.view(),
            weights.view(),
        );
        assert!(result.is_some());
        assert_eq!(result.unwrap()[0], 1);
    }
}
