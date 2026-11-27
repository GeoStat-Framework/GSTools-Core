"""Release-wheel MPS boundary matrix, with one bounded process per case.

Run with the interpreter containing the intended installed release wheel. Each
child disables core dumps and must confirm its specified result or ValueError;
assertion failures, unexpected exceptions, signals and timeouts are failures.
The parent prints machine-readable JSON and exits nonzero if any case fails.

Boundary contract: block event values are finite, TI NaNs require a masked
kernel, and weights are finite/nonnegative (unmasked kernels expect callers to
normalize them). Empty candidate batches return empty arrays. Empty/zero-weight
events retain the numerical kernels' behavior: unmasked distances are zero;
masked kernels return infinity when no positive comparable weight remains.
Raw scans return None for valid empty windows/zero budgets/all-invalid masked
candidates, raise ValueError for malformed descriptors, and cap both candidate
orders to one visit per window cell. Strided and readonly buffers are supported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True

BLOCKS = (
    "cat",
    "cat_rayon",
    "cat_masked",
    "l1",
    "l1_masked",
    "l2",
    "l2_masked",
    "lp",
    "lp_masked",
    "variation",
    "variation_masked",
)
BLOCK_CASES = (
    "lower",
    "upper",
    "signed-overflow",
    "length",
    "weight-length",
    "negative-weight",
    "nan-weight",
    "infinite-weight",
    "weight-sum-overflow",
    "infinite-event",
    "infinite-ti",
    "empty-candidates",
    "empty-event",
    "negative-base-valid",
    "negative-lag-valid",
    "strided",
    "reversed",
    "readonly",
    "zero-weight",
)
SCAN_CASES = (
    "negative-window",
    "shape-overflow",
    "coordinate-overflow",
    "stride-overflow",
    "dimension",
    "stride-zero",
    "lower",
    "upper",
    "gather-lower",
    "gather-upper",
    "zero-window",
    "empty-dim",
    "zero-budget",
    "oversized",
    "exact-window",
    "start-overflow",
    "threshold-negative",
    "threshold-nan",
    "event-length",
    "weight-negative",
    "infinite-event",
    "infinite-ti",
    "strided",
)
MULTI_CASES = (
    "active-row",
    "target-row",
    "descriptor-length",
    "offset-empty",
    "offset-negative",
    "offset-nonzero-first",
    "offset-descending",
    "offset-final",
    "metric",
    "range",
    "exponent",
    "variable-weight",
    "penalty-short",
    "penalty-negative-offset",
    "penalty-negative-count",
    "penalty-empty",
    "penalty-overflow",
    "penalty-infinite",
    "penalty-diagonal",
    "penalty-strided",
    "all-missing",
    "missing-flag",
    "tie",
    "positive-strict",
)
ENGINE_CASES = (
    "shape-negative",
    "shape-zero",
    "shape-empty",
    "shape-overflow",
    "dimension",
    "neighbors-negative",
    "neighbors-zero",
    "neighbors-huge",
    "radius-infinite",
    "radius-negative",
    "weights-negative",
    "weights-infinite",
    "metric",
    "range",
    "exponent",
    "distance-power",
    "cond-weight",
    "threshold",
    "scan-fraction",
    "path-negative",
    "path-upper",
    "path-duplicate",
    "uniform",
    "ti-infinite",
    "field-infinite",
    "hard-nan",
    "hard-mask",
    "active-mask-length",
    "active-mask-value",
    "inactive-path",
    "inactive-hard",
    "offset-zero",
    "offset-unordered",
    "offset-overflow",
    "matrix-nan",
    "matrix-overflow",
    "zone-negative-offset",
    "zone-short-offset",
    "zone-negative-shape",
    "zone-shape-overflow",
    "zone-range",
    "domain",
    "penalty-short",
    "empty-path",
    "partial-hard",
    "primary-missing-flag",
)


def cases():
    for block in BLOCKS:
        for case in BLOCK_CASES:
            yield f"block/{block}/{case}"
        if block.startswith(("l1", "l2", "lp", "variation")):
            for case in ("range-zero", "range-infinite", "range-nan"):
                yield f"block/{block}/{case}"
        if block.startswith(("lp", "variation")):
            for case in ("exponent-zero", "exponent-infinite", "exponent-nan"):
                yield f"block/{block}/{case}"
        if "masked" in block:
            yield f"block/{block}/all-missing"
    for scan in ("cat", "multi"):
        for case in SCAN_CASES:
            yield f"scan/{scan}/{case}"
        for size in (1, 2, 3, 4, 7, 8, 10, 31):
            for random in (False, True) if scan == "multi" else (False,):
                yield f"scan/{scan}/budget-{size}-{int(random)}"
    for case in MULTI_CASES:
        yield f"scan/multi/{case}"
    for engine in ("simulate", "cleanup"):
        for case in ENGINE_CASES:
            yield f"engine/{engine}/{case}"


def probe(identifier):
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except ImportError:  # Windows has no POSIX resource module.
        pass
    import numpy as np
    import gstools_core as core
    from check_mps_wheel import _engine_kwargs

    def ints(v):
        return np.array(v, dtype=np.int64)

    def floats(v):
        return np.array(v, dtype=np.float64)

    def reject(call):
        try:
            call()
        except ValueError as error:
            assert str(error), "validation error must be descriptive"
            return
        raise AssertionError("malformed input did not raise ValueError")

    group, name, case = identifier.split("/")
    if group == "block":
        args = dict(
            de_sim=floats([0.0]),
            ti_flat=floats([1.0, 0.0]),
            base_flat=ints([0]),
            lag_flat=ints([0]),
            node_weights=floats([1.0]),
        )
        if name.startswith(("l1", "l2", "lp", "variation")):
            args["d_max"] = 1.0
        if name.startswith(("lp", "variation")):
            args["p"] = 2.0

        def call():
            return getattr(core, "mps_dist_block_" + name)(**args)

        if case == "lower":
            args["base_flat"][0] = -1
        elif case == "upper":
            args["base_flat"][0] = 2
        elif case == "signed-overflow":
            args["base_flat"][0] = 2**63 - 1
            args["lag_flat"][0] = 1
        elif case == "length":
            args["lag_flat"] = ints([])
        elif case == "weight-length":
            args["node_weights"] = floats([])
        elif case == "negative-weight":
            args["node_weights"][0] = -1
        elif case == "nan-weight":
            args["node_weights"][0] = np.nan
        elif case == "infinite-weight":
            args["node_weights"][0] = np.inf
        elif case == "weight-sum-overflow":
            args.update(
                de_sim=floats([0, 0]),
                lag_flat=ints([0, 0]),
                node_weights=floats([1e308, 1e308]),
            )
        elif case == "infinite-event":
            args["de_sim"][0] = np.inf
        elif case == "infinite-ti":
            args["ti_flat"][0] = np.inf
        elif case.startswith("range-"):
            args["d_max"] = {"zero": 0.0, "infinite": np.inf, "nan": np.nan}[case[6:]]
        elif case.startswith("exponent-"):
            args["p"] = {"zero": 0.0, "infinite": np.inf, "nan": np.nan}[case[9:]]
        elif case == "empty-candidates":
            args["base_flat"] = ints([])
            assert call().shape == (0,)
            return
        elif case == "empty-event":
            args.update(de_sim=floats([]), lag_flat=ints([]), node_weights=floats([]))
            result = call()
            assert result.shape == (1,)
            assert not np.isnan(result).any()
            return
        elif case == "negative-base-valid":
            args["base_flat"][0], args["lag_flat"][0] = -1, 1
            assert call().shape == (1,)
            return
        elif case == "negative-lag-valid":
            args["base_flat"][0], args["lag_flat"][0] = 1, -1
            assert call().shape == (1,)
            return
        elif case in ("strided", "reversed", "readonly"):
            expected = call()
            for key, value in list(args.items()):
                if not isinstance(value, np.ndarray):
                    continue
                if case == "strided":
                    buffer = np.empty(value.size * 2, dtype=value.dtype)
                    buffer[::2] = value
                    args[key] = buffer[::2]
                elif case == "reversed":
                    args[key] = value[::-1].copy()[::-1]
                else:
                    value.flags.writeable = False
            np.testing.assert_array_equal(call(), expected)
            return
        elif case == "all-missing":
            args["ti_flat"][:] = np.nan
            assert np.isinf(call()).all()
            return
        elif case == "zero-weight":
            args["node_weights"][0] = 0.0
            assert not np.isnan(call()).any()
            return
        else:
            raise AssertionError(case)
        reject(call)
        return

    if group == "scan":
        args = dict(
            lo=ints([0]),
            win_shape=ints([2]),
            start=0,
            max_scan=2,
            threshold=0.0,
            ti_flat=floats([[0.0, 1.0]]),
            ti_strides=ints([1]),
            active_ti_rows=ints([0]),
            event_offsets=ints([0, 1]),
            de_sim=floats([2.0]),
            lag_flat=ints([0]),
            node_weights=floats([1.0]),
            variable_weights=floats([1.0]),
            metric_kinds=ints([0]),
            has_nan=np.array([0], np.uint8),
            d_max=floats([1.0]),
            p_norm=floats([1.0]),
            random_scan=False,
            scan_seed=0,
            penalty_flat=floats([]),
            penalty_offsets=ints([0]),
            penalty_ncat=ints([0]),
            target_ti_rows=ints([0]),
            check_centers=True,
        )
        if name == "cat":
            args = {
                key: value
                for key, value in args.items()
                if key
                in (
                    "lo",
                    "win_shape",
                    "start",
                    "max_scan",
                    "threshold",
                    "ti_flat",
                    "ti_strides",
                    "de_sim",
                    "lag_flat",
                    "node_weights",
                )
            }
            args["ti_flat"] = args["ti_flat"][0]

        def call():
            return getattr(core, "mps_scan_node" + ("_cat" if name == "cat" else ""))(
                **args
            )

        if case == "negative-window":
            args["win_shape"][0] = -1
        elif case == "shape-overflow":
            args.update(
                lo=ints([0, 0]), win_shape=ints([2**62, 4]), ti_strides=ints([4, 1])
            )
        elif case == "coordinate-overflow":
            args["lo"][0] = 2**63 - 1
        elif case == "stride-overflow":
            args["ti_strides"][0] = 2**63 - 1
        elif case == "dimension":
            args["lo"] = ints([0, 0])
        elif case == "stride-zero":
            args["ti_strides"][0] = 0
        elif case == "lower":
            args["lo"][0] = -1
        elif case == "upper":
            args["lo"][0] = 2
        elif case == "gather-lower":
            args["lag_flat"][0] = -1
        elif case == "gather-upper":
            args["lag_flat"][0] = 1
        elif case == "zero-window":
            args["win_shape"][0] = 0
            assert call() is None
            return
        elif case == "empty-dim":
            args.update(lo=ints([]), win_shape=ints([]), ti_strides=ints([]))
            assert call() is None
            return
        elif case == "zero-budget":
            args["max_scan"] = 0
            assert call() is None
            return
        elif case in ("oversized", "exact-window", "start-overflow"):
            args["max_scan"] = 3 if case == "oversized" else 2
            if case == "start-overflow":
                args["start"] = 2**64 - 1
            np.testing.assert_array_equal(
                call(), [0] if case != "start-overflow" else [1]
            )
            return
        elif case.startswith("budget-"):
            _, size, random = case.split("-")
            size = int(size)
            args["win_shape"][0] = size
            args["ti_flat"] = np.zeros((1, size)) if name == "multi" else np.zeros(size)
            if name == "multi":
                args["random_scan"] = bool(int(random))
            args["max_scan"] = size
            expected = call()
            args["max_scan"] = size + 3
            np.testing.assert_array_equal(call(), expected)
            args["max_scan"] = 0
            assert call() is None
            return
        elif case == "threshold-negative":
            args["threshold"] = -1.0
        elif case == "threshold-nan":
            args["threshold"] = np.nan
        elif case == "event-length":
            args["de_sim"] = floats([])
        elif case == "weight-negative":
            args["node_weights"][0] = -1.0
        elif case == "infinite-event":
            args["de_sim"][0] = np.inf
        elif case == "infinite-ti":
            args["ti_flat"].flat[0] = np.inf
        elif case == "strided":
            expected = call()
            for key, value in list(args.items()):
                if isinstance(value, np.ndarray) and value.ndim == 1:
                    buffer = np.empty(value.size * 2, dtype=value.dtype)
                    buffer[::2] = value
                    args[key] = buffer[::2]
            np.testing.assert_array_equal(call(), expected)
            return
        elif case == "active-row":
            args["active_ti_rows"][0] = 1
        elif case == "target-row":
            args["target_ti_rows"][0] = -1
        elif case == "descriptor-length":
            args["metric_kinds"] = ints([])
        elif case == "offset-empty":
            args["event_offsets"] = ints([])
        elif case == "offset-negative":
            args["event_offsets"][0] = -1
        elif case == "offset-nonzero-first":
            args["event_offsets"][0] = 1
        elif case == "offset-descending":
            args["event_offsets"] = ints([1, 0])
        elif case == "offset-final":
            args["event_offsets"][1] = 2
        elif case == "metric":
            args["metric_kinds"][0] = 9
        elif case == "range":
            args["d_max"][0] = 0
        elif case == "exponent":
            args["p_norm"][0] = np.inf
        elif case == "variable-weight":
            args["variable_weights"][0] = -1
        elif case.startswith("penalty-"):
            args.update(
                metric_kinds=ints([3]),
                penalty_flat=floats([0.0, 1.0, 1.0, 0.0]),
                penalty_ncat=ints([2]),
                de_sim=floats([1.0]),
            )
            if case == "penalty-short":
                args["penalty_flat"] = floats([0.0])
            elif case == "penalty-negative-offset":
                args["penalty_offsets"][0] = -1
            elif case == "penalty-negative-count":
                args["penalty_ncat"][0] = -1
            elif case == "penalty-empty":
                args["penalty_ncat"][0] = 0
            elif case == "penalty-overflow":
                args["penalty_ncat"][0] = 2**63 - 1
            elif case == "penalty-infinite":
                args["penalty_flat"][1] = np.inf
            elif case == "penalty-diagonal":
                args["penalty_flat"][0] = 0.1
            elif case == "penalty-strided":
                expected = call()
                buffer = np.empty(8)
                buffer[::2] = args["penalty_flat"]
                args["penalty_flat"] = buffer[::2]
                np.testing.assert_array_equal(call(), expected)
                return
            else:
                raise AssertionError(case)
        elif case == "all-missing":
            args["ti_flat"][:] = np.nan
            args["has_nan"][0] = 1
            assert call() is None
            return
        elif case == "missing-flag":
            args["ti_flat"][0, 0] = np.nan
        elif case == "tie":
            np.testing.assert_array_equal(call(), [0])
            args["start"] = 1
            np.testing.assert_array_equal(call(), [1])
            return
        elif case == "positive-strict":
            args.update(
                metric_kinds=ints([1]),
                de_sim=floats([0.5]),
                threshold=0.5,
                ti_flat=floats([[0.0, 0.5]]),
            )
            np.testing.assert_array_equal(call(), [1])
            return
        else:
            raise AssertionError(case)
        reject(call)
        return

    args = _engine_kwargs()
    if name == "cleanup":
        args["fields_in"] = floats([[0.0]])
        del args["initial_fields"]
    field_key = "fields_in" if name == "cleanup" else "initial_fields"

    def call():
        return getattr(
            core, "mps_simulate" if name == "simulate" else "mps_cleanup_pass"
        )(**args)

    if case == "shape-negative":
        args["sim_shape"][0] = -1
    elif case == "shape-zero":
        args["sim_shape"][0] = 0
    elif case == "shape-empty":
        args["sim_shape"] = ints([])
    elif case == "shape-overflow":
        args["sim_shape"] = ints([2**62, 4])
    elif case == "dimension":
        args.update(ti_shape=ints([1, 2]))
    elif case == "neighbors-negative":
        args["n_neighbors"][0] = -1
    elif case == "neighbors-zero":
        args["n_neighbors"][0] = 0
    elif case == "neighbors-huge":
        args["n_neighbors"][0] = 2**63 - 1
        assert np.isfinite(call()[0]).all()
        return
    elif case == "radius-infinite":
        args["max_radius"][0] = np.inf
    elif case == "radius-negative":
        args["max_radius"][0] = -1
    elif case == "weights-negative":
        args["variable_weights"][0] = -1
    elif case == "weights-infinite":
        args["variable_weights"][0] = np.inf
    elif case == "metric":
        args["metric_kinds"][0] = 9
    elif case == "range":
        args["d_max"][0] = 0
    elif case == "exponent":
        args["p_norm"][0] = 0
    elif case == "distance-power":
        args["distance_power"][0] = np.inf
    elif case == "cond-weight":
        args["cond_weight"][0] = -1
    elif case == "threshold":
        args["threshold"] = np.nan
    elif case == "scan-fraction":
        args["scan_fraction"] = np.inf
    elif case == "path-negative":
        args["path"][0, 0] = -1
    elif case == "path-upper":
        args["path"][0, 0] = 1
    elif case == "path-duplicate":
        args.update(
            path=ints([[0], [0]]),
            u_start=floats([0.0, 0.0]),
            u_fallback=floats([[0.0], [0.0]]),
            node_domain=ints([0, 0]),
        )
    elif case == "uniform":
        args["u_start"][0] = 1.0
    elif case == "ti-infinite":
        args["ti"][0, 0] = np.inf
    elif case == "field-infinite":
        args[field_key][0, 0] = np.inf
    elif case == "hard-nan":
        args["conditioned"][0, 0] = 1
        args[field_key][0, 0] = np.nan
    elif case == "hard-mask":
        args["conditioned"][0, 0] = 2
    elif case == "active-mask-length":
        args["active_mask"] = np.ones(2, dtype=np.uint8)
    elif case == "active-mask-value":
        args["active_mask"] = np.array([2], dtype=np.uint8)
    elif case in {"inactive-path", "inactive-hard"}:
        args["active_mask"] = np.zeros(1, dtype=np.uint8)
        if case == "inactive-hard":
            args.update(
                path=ints([]).reshape(0, 1),
                u_start=floats([]),
                u_fallback=floats([]).reshape(0, 1),
                node_domain=ints([]),
            )
            args["conditioned"][0, 0] = 1
            args[field_key][0, 0] = 0.0
    elif case == "offset-zero":
        args["offsets"] = ints([[0]])
    elif case == "offset-unordered":
        args["offsets"] = ints([[2], [1]])
    elif case == "offset-overflow":
        args["offsets"] = ints([[2**63 - 1]])
    elif case == "matrix-nan":
        args["lag_matrices"] = floats([[[np.nan]]])
    elif case == "matrix-overflow":
        args.update(offsets=ints([[1]]), lag_matrices=floats([[[1e308]]]))
    elif case.startswith("zone-"):
        args.update(
            zone_ti=floats([[0.0, 1.0]]),
            zone_offsets=ints([0, 2]),
            zone_shapes=ints([[2]]),
            zone_d_max=floats([[1.0]]),
            node_domain=ints([1]),
        )
        if case == "zone-negative-offset":
            args["zone_offsets"][0] = -1
        elif case == "zone-short-offset":
            args["zone_offsets"][1] = 1
        elif case == "zone-negative-shape":
            args["zone_shapes"][0, 0] = -2
        elif case == "zone-shape-overflow":
            args["zone_shapes"] = ints([[2**62, 4]])
        elif case == "zone-range":
            args["zone_d_max"][0, 0] = np.inf
    elif case == "domain":
        args["node_domain"][0] = 1
    elif case == "penalty-short":
        args.update(metric_kinds=ints([3]), penalty_ncat=ints([2]))
    elif case == "empty-path":
        args.update(
            path=ints([]).reshape(0, 1),
            u_start=floats([]),
            u_fallback=floats([]).reshape(0, 1),
            node_domain=ints([]),
        )
        args["conditioned"][0, 0] = 1
        args[field_key][0, 0] = 1.0
        np.testing.assert_array_equal(call()[0], [[1.0]])
        return
    elif case == "partial-hard":
        args.update(
            ti=np.array([[0.0, 1.0], [3.0, 4.0]]),
            initial_fields=np.array([[2.0], [np.nan]]),
            conditioned=np.array([[1], [0]], np.uint8),
            n_neighbors=ints([1, 1]),
            max_radius=floats([np.nan, np.nan]),
            variable_weights=floats([0.5, 0.5]),
            metric_kinds=ints([0, 0]),
            has_nan=np.array([0, 0], np.uint8),
            d_max=floats([1.0, 1.0]),
            p_norm=floats([1.0, 1.0]),
            penalty_offsets=ints([0, 0]),
            penalty_ncat=ints([0, 0]),
            distance_power=floats([0.0, 0.0]),
            cond_weight=floats([1.0, 1.0]),
            zone_ti=np.empty((2, 0)),
            zone_d_max=np.empty((0, 2)),
        )
        if name == "cleanup":
            args["fields_in"] = floats([[2.0], [3.0]])
            del args["initial_fields"]
        result = call()[0]
        assert result[0, 0] == 2.0
        assert np.isfinite(result).all()
        return
    elif case == "primary-missing-flag":
        args.update(ti=floats([[np.nan, 1.0]]), has_nan=np.array([0], np.uint8))
        assert np.isfinite(call()[0]).all()
        return
    else:
        raise AssertionError(case)
    reject(call)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=30.0)
    options = parser.parse_args()
    if options.case:
        probe(options.case)
        print("confirmed", options.case)
        return 0
    results = []
    for identifier in cases():
        started = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--case", identifier],
                capture_output=True,
                text=True,
                timeout=options.timeout,
            )
            result = dict(
                case=identifier,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        except subprocess.TimeoutExpired as error:
            result = dict(
                case=identifier, returncode=None, timeout=True, stderr=str(error)
            )
        result["duration_seconds"] = time.monotonic() - started
        results.append(result)
    report = dict(
        results=results,
        passed=sum(r["returncode"] == 0 for r in results),
        failed=sum(r["returncode"] != 0 for r in results),
    )
    serialized = json.dumps(report, indent=2)
    if options.output:
        options.output.write_text(serialized + "\n")
    print(serialized)
    return int(report["failed"] != 0)


if __name__ == "__main__":
    sys.exit(main())
