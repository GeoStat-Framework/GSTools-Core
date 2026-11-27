"""Smoke-test the MPS exports of an installed gstools-core wheel.

Calls the current API 8 ``mps_simulate`` / ``mps_cleanup_pass`` signatures with
explicit keyword arguments so a future positional change fails loudly here.
Requires the versioned API export and the cleanup entry point in
addition to the distance/scan/simulate kernels.
"""

from __future__ import annotations

import numpy as np

import gstools_core


REQUIRED_EXPORTS = (
    "mps_dist_block_cat",
    "mps_dist_block_cat_masked",
    "mps_dist_block_cat_rayon",
    "mps_dist_block_l1",
    "mps_dist_block_l1_masked",
    "mps_dist_block_l2",
    "mps_dist_block_l2_masked",
    "mps_dist_block_lp",
    "mps_dist_block_lp_masked",
    "mps_dist_block_variation",
    "mps_dist_block_variation_masked",
    "mps_scan_node",
    "mps_scan_node_cat",
    "mps_simulate",
    "mps_cleanup_pass",
    "mps_api_version",
)

# Exact MPS interface revision supported by this paired smoke test.
REQUIRED_API = 8


def _engine_kwargs(num_threads=1):
    """Keyword arguments for a one-cell categorical ``mps_simulate`` call."""
    return dict(
        ti=np.array([[0.0, 1.0]]),
        ti_shape=np.array([2], dtype=np.int64),
        initial_fields=np.array([[np.nan]]),
        conditioned=np.zeros((1, 1), dtype=np.uint8),
        sim_shape=np.array([1], dtype=np.int64),
        path=np.array([[0]], dtype=np.int64),
        lag_matrices=np.empty((0, 1, 1)),
        u_start=np.array([0.0]),
        u_fallback=np.array([[0.1]]),
        offsets=np.empty((0, 1), dtype=np.int64),
        n_neighbors=np.array([1], dtype=np.int64),
        max_radius=np.array([np.nan]),
        variable_weights=np.array([1.0]),
        metric_kinds=np.array([0], dtype=np.int64),
        has_nan=np.array([0], dtype=np.uint8),
        d_max=np.array([1.0]),
        p_norm=np.array([1.0]),
        penalty_flat=np.empty(0),
        penalty_offsets=np.zeros(1, dtype=np.int64),
        penalty_ncat=np.zeros(1, dtype=np.int64),
        threshold=0.0,
        scan_fraction=1.0,
        random_scan=False,
        distance_power=np.zeros(1),
        cond_weight=np.ones(1),
        partial_boundary=False,
        num_threads=num_threads,
        zone_ti=np.empty((1, 0)),
        zone_offsets=np.array([0], dtype=np.int64),
        zone_shapes=np.empty((0, 1), dtype=np.int64),
        zone_d_max=np.empty((0, 1)),
        node_domain=np.zeros(1, dtype=np.int64),
    )


def main() -> None:
    missing = [name for name in REQUIRED_EXPORTS if not hasattr(gstools_core, name)]
    if missing:
        raise RuntimeError(f"installed wheel is missing MPS exports: {missing}")

    api = int(gstools_core.mps_api_version())
    if api != REQUIRED_API:
        raise RuntimeError(
            f"installed wheel reports MPS ABI {api}, need exactly {REQUIRED_API}"
        )

    distance = gstools_core.mps_dist_block_cat(
        np.array([0.1]),
        np.array([0.2]),
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.array([1.0]),
    )
    np.testing.assert_array_equal(distance, np.array([1.0]))

    field, strict_fallbacks, levels, ready_width, threads, collapsed = (
        gstools_core.mps_simulate(**_engine_kwargs())
    )
    np.testing.assert_array_equal(field, np.array([[0.0]]))
    assert (strict_fallbacks, levels, ready_width, threads, collapsed) == (
        0,
        1,
        1,
        1,
        0,
    )

    # Cleanup smoke check: a complete one-cell field is re-simulated in place.
    cleanup_kwargs = {
        key: value
        for key, value in _engine_kwargs().items()
        if key not in {"initial_fields"}
    }
    cleanup_kwargs["fields_in"] = np.array([[0.0]])
    cleaned, c_strict, c_collapsed, c_threads = gstools_core.mps_cleanup_pass(
        **cleanup_kwargs
    )
    np.testing.assert_array_equal(cleaned, np.array([[0.0]]))
    assert (c_strict, c_collapsed, c_threads) == (0, 0, 1)

    masked = _engine_kwargs()
    masked.update(
        active_mask=np.array([1, 0], dtype=np.uint8),
        initial_fields=np.full((1, 2), np.nan),
        conditioned=np.zeros((1, 2), dtype=np.uint8),
        sim_shape=np.array([2], dtype=np.int64),
        offsets=np.array([[1]], dtype=np.int64),
    )
    field = gstools_core.mps_simulate(**masked)[0]
    np.testing.assert_array_equal(field, [[0.0, np.nan]])
    masked.pop("initial_fields")
    masked["fields_in"] = field
    cleaned = gstools_core.mps_cleanup_pass(**masked)[0]
    np.testing.assert_array_equal(cleaned, [[0.0, np.nan]])

    print(f"gstools-core MPS wheel smoke test (API {api}): PASS")


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
