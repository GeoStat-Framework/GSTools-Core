# How to Contribute to GSTools Core

We are happy about all contributions! :thumbsup:


## Did you find a bug?

- Ensure that the bug was not already reported under
[GitHub issues](https://github.com/GeoStat-Framework/GSTools-Core/issues)
- If the bug wasn't already reported, open a
[new issue](https://github.com/GeoStat-Framework/GSTools-Core/issues) with a clear
description of the problem and if possible with a
[minimal working example](https://en.wikipedia.org/wiki/Minimal_working_example).
- please add the version number to the issue:

```python
import gstools_core
print(gstools_core.__version__)
```


## Do you have suggestions for new features?

Open a [new issue](https://github.com/GeoStat-Framework/GSTools-Core/issues)
with your idea or suggestion and we'd love to discuss about it.


## Do you want to enhance GSTools or fix something?

- Fork the repo on [GitHub](https://github.com/GeoStat-Framework/GSTools-Core)
- Add yourself to AUTHORS.md (if you want to).
- Add tests to the bottom of the respective file.
- We use [rustfmt](https://github.com/rust-lang/rustfmt) format the code.
  Please use the script `cargo fmt` after you have written your code.
- We use [clippy](https://github.com/rust-lang/rust-clippy) as the linter.
  Please use the script `cargo clippy` after you have written your code.
- Push to your fork and submit a pull request.


## Tips on getting started

The steps shown here use [uv](https://docs.astral.sh/uv/), but every other Python package manager will work too.

- First, create a virtual environment with `uv venv .venv` or similar, maybe you have to source it with `source .venv/bin/activate`.
- Next, install maturin for building the Python package with `uv tool install maturin`.
- Now, you can build the crate and install it as a Python module into the current venv with `maturin develop`
- For perfomance testing, you can compile the code optimized with `maturin develop --release`.
- If you install GSTools into the same venv with `uv pip install gstools`, it will automatically use the local GSTools Core package installed with maturin.

Installed-wheel Direct Sampling smoke and bounded boundary checks are documented
in [scripts/README.md](scripts/README.md). Keep these source-only tools with the
source archive and run them against the exact artifact selected for release.
