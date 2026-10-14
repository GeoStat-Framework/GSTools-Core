//! Wire format for covariance models: exactly what comes in over JSON.
//!
//! Mirrors the parameter names gstools itself uses (`var`, `len_scale`,
//! `nugget`, `nu`), so a small `to_dict()` on the Python side would produce
//! compatible JSON directly. No validation happens here — a `CovModelSpec`
//! is just "what the wire said", turned into a usable [`crate::covmodel::CovModel`]
//! via [`crate::covmodel::CovModel::build`].

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CovModelSpec {
    Gaussian {
        var: f64,
        len_scale: f64,
        nugget: f64,
    },
    Exponential {
        var: f64,
        len_scale: f64,
        nugget: f64,
    },
    Matern {
        var: f64,
        len_scale: f64,
        nugget: f64,
        nu: f64,
    },
    Spherical {
        var: f64,
        len_scale: f64,
        nugget: f64,
    },
}
