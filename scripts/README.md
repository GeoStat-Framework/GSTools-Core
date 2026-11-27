# Installed-wheel Direct Sampling checks

Run these source-only scripts with the interpreter in which the intended native
wheel is installed. They require NumPy and use no network or temporary venv.

- `python scripts/check_mps_wheel.py` checks required exports, exact supported
  interface revision 8, and real main/cleanup calls with explicit masks and
  expected tuple shapes. Exit zero means these checks passed, not import alone.
- `python scripts/check_mps_boundaries.py --output /tmp/native-boundaries.json`
  runs the invalid-input/numerical matrix in bounded child processes. It checks
  valid empty inputs, strided/read-only buffers, exceptions, and numerical
  results, detecting signals and timeouts. The JSON records individual case IDs
  and outcomes. It uses `check_mps_wheel.py` helpers; keep both files together.
  An existing JSON output file is replaced; choose a new filename to preserve
  earlier evidence. Nonzero exits
  indicate failed cases. `--help` lists timeout and child-selection controls.

These checks complement Rust and frontend tests. They do not establish universal
statistical fidelity, cross-platform identity, or approval to publish a release.
An interface revision identifies callable signatures, not a unique binary build;
record source provenance and wheel/native hashes for exact artifact identity.
