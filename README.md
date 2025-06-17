# GSTools Core


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15684753.svg)](https://doi.org/10.5281/zenodo.15684753)
[![PyPI version](https://img.shields.io/pypi/v/gstools-core.svg)](https://pypi.org/project/gstools-core/)
[![Crates version](https://img.shields.io/crates/v/gstools-core.svg)](https://crates.io/crates/gstools-core)
[![Docs](https://docs.rs/gstools-core/badge.svg)](https://docs.rs/gstools-core)
[![Build Status](https://github.com/GeoStat-Framework/GSTools-Core/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/GeoStat-Framework/GSTools-Core/actions/workflows/main.yml)

A Rust implementation of the core algorithms of [GSTools][gstools_link].

This crate includes

- randomization methods for the random field generation
- the matrix operations of the kriging methods
- the variogram estimation

The documentation can be found [here][doc_link].


## Usage

You probably don't want to use this software directly, as it is the computational
backend of GSTools. See the [installation guide][gstools_installation] on how to
use this backend with GSTools.


## Contact

You can contact us via <info@geostat-framework.org>.


## License

[LGPLv3][license_link] © 2021-2025


[gstools_link]: https://github.com/GeoStat-Framework/GSTools
[gstools_installation]: https://geostat-framework.readthedocs.io/projects/gstools/en/latest/#installation
[doc_link]: https://docs.rs/gstools-core/latest/gstools_core/
[license_link]: https://github.com/GeoStat-Framework/GSTools-Core/blob/main/LICENSE
