//! Runtime covariance models.
//!
//! A [`CovModel`] is *built* from a [`CovModelSpec`] via [`CovModel::build`],
//! which validates the raw parameters and precomputes everything that would
//! otherwise be recomputed on every one of the O(n²) pairwise evaluations
//! needed to fill a kriging LHS/RHS: the rescaled length scale, and for
//! Matérn the polynomial coefficients of the half-integer closed form.
//!
//! Formulas follow gstools' own `CovModel.cor` convention exactly (see
//! `gstools/covmodel/models.py`): `correlation(r) = cor(r / len_rescaled)`,
//! `covariance(r) = var * correlation(r)`, with `len_rescaled = len_scale /
//! rescale` and a model-specific rescale factor.

use crate::covmodel_spec::CovModelSpec;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    InvalidVar(f64),
    InvalidLenScale(f64),
    InvalidNugget(f64),
    InvalidNu(f64),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidVar(v) => write!(f, "var must be >= 0, got {v}"),
            Self::InvalidLenScale(v) => write!(f, "len_scale must be > 0, got {v}"),
            Self::InvalidNugget(v) => write!(f, "nugget must be >= 0, got {v}"),
            Self::InvalidNu(v) => write!(f, "nu must be > 0, got {v}"),
        }
    }
}

impl std::error::Error for ModelError {}

/// Error building a [`CovModel`] from a JSON-encoded [`CovModelSpec`] via
/// [`CovModel::from_json`]: either the JSON itself is malformed/doesn't
/// match the wire format, or it parses fine but [`CovModel::build`]
/// rejects the parameters it describes.
#[derive(Debug)]
pub enum CovModelJsonError {
    Json(serde_json::Error),
    Model(ModelError),
}

impl fmt::Display for CovModelJsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(e) => write!(f, "invalid covariance model JSON: {e}"),
            Self::Model(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CovModelJsonError {}

impl From<serde_json::Error> for CovModelJsonError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

impl From<ModelError> for CovModelJsonError {
    fn from(e: ModelError) -> Self {
        Self::Model(e)
    }
}

/// Gaussian's standard rescale factor `s = sqrt(pi) / 2`, see
/// `Gaussian.default_rescale` in gstools.
fn gaussian_rescale() -> f64 {
    std::f64::consts::PI.sqrt() / 2.0
}

#[derive(Debug, Clone, PartialEq)]
pub enum CovModel {
    Gaussian {
        var: f64,
        len_rescaled: f64,
        nugget: f64,
    },
    Exponential {
        var: f64,
        len_rescaled: f64,
        nugget: f64,
    },
    Matern {
        var: f64,
        len_rescaled: f64,
        nugget: f64,
        nu: f64,
        sqrt_nu: f64,
    },
    Spherical {
        var: f64,
        len_rescaled: f64,
        nugget: f64,
    },
}

/// Matérn normalized correlation function, following gstools'
/// `Matern.cor` (`gstools/covmodel/models.py`) term for term:
/// `cor(h) = 2^(1-nu)/Gamma(nu) * (sqrt(nu) h)^nu * K_nu(sqrt(nu) h)`,
/// evaluated via a single `exp(...)` of the log-transformed prefactor to
/// avoid overflow in `Gamma(nu)` for large `nu`, and via the Gaussian
/// far-field approximation for `nu > 20` where the exact term underflows.
fn matern_cor(nu: f64, sqrt_nu: f64, h: f64) -> f64 {
    // for nu > 20 we just use the gaussian model
    if nu > 20.0 {
        return (-(h / 2.0).powi(2)).exp();
    }
    if h <= 0.0 {
        return 1.0;
    }
    // calculate by log-transformation to prevent numerical errors
    let x = sqrt_nu * h;
    let res = ((1.0 - nu) * std::f64::consts::LN_2 - scirs2_special::loggamma(nu) + nu * x.ln())
        .exp()
        * scirs2_special::kv(nu, x);
    // if nu >> 1 we get errors for the farfield, there 0 is approached;
    // covariance is positive
    if res.is_finite() {
        res.max(0.0)
    } else {
        0.0
    }
}

impl CovModel {
    /// Build a [`CovModel`] straight from the JSON wire format a
    /// [`CovModelSpec`] deserializes from -- the single entry point Python
    /// callers (via PyO3) go through, so the wire format's parsing and its
    /// parameter validation both stay colocated with the model types
    /// themselves rather than in the PyO3 binding layer.
    pub fn from_json(json: &str) -> Result<Self, CovModelJsonError> {
        let spec: CovModelSpec = serde_json::from_str(json)?;
        Ok(Self::build(spec)?)
    }

    pub fn build(spec: CovModelSpec) -> Result<Self, ModelError> {
        match spec {
            CovModelSpec::Gaussian {
                var,
                len_scale,
                nugget,
            } => {
                check_common(var, len_scale, nugget)?;
                Ok(Self::Gaussian {
                    var,
                    len_rescaled: len_scale / gaussian_rescale(),
                    nugget,
                })
            }
            CovModelSpec::Exponential {
                var,
                len_scale,
                nugget,
            } => {
                check_common(var, len_scale, nugget)?;
                Ok(Self::Exponential {
                    var,
                    len_rescaled: len_scale, // rescale = 1
                    nugget,
                })
            }
            CovModelSpec::Matern {
                var,
                len_scale,
                nugget,
                nu,
            } => {
                check_common(var, len_scale, nugget)?;
                if nu.is_nan() || nu <= 0.0 {
                    return Err(ModelError::InvalidNu(nu));
                }
                Ok(Self::Matern {
                    var,
                    len_rescaled: len_scale, // rescale = 1
                    nugget,
                    nu,
                    sqrt_nu: nu.sqrt(),
                })
            }
            CovModelSpec::Spherical {
                var,
                len_scale,
                nugget,
            } => {
                check_common(var, len_scale, nugget)?;
                Ok(Self::Spherical {
                    var,
                    len_rescaled: len_scale, // rescale = 1
                    nugget,
                })
            }
        }
    }

    fn var(&self) -> f64 {
        match self {
            Self::Gaussian { var, .. }
            | Self::Exponential { var, .. }
            | Self::Matern { var, .. }
            | Self::Spherical { var, .. } => *var,
        }
    }

    fn nugget(&self) -> f64 {
        match self {
            Self::Gaussian { nugget, .. }
            | Self::Exponential { nugget, .. }
            | Self::Matern { nugget, .. }
            | Self::Spherical { nugget, .. } => *nugget,
        }
    }

    /// Normalized correlation function, `correlation(0) = 1`.
    pub fn correlation(&self, r: f64) -> f64 {
        let r = r.abs();
        match self {
            Self::Gaussian { len_rescaled, .. } => {
                let h = r / len_rescaled;
                (-(h * h)).exp()
            }
            Self::Exponential { len_rescaled, .. } => {
                let h = r / len_rescaled;
                (-h).exp()
            }
            Self::Matern {
                len_rescaled,
                nu,
                sqrt_nu,
                ..
            } => {
                let h = r / len_rescaled;
                matern_cor(*nu, *sqrt_nu, h)
            }
            Self::Spherical { len_rescaled, .. } => {
                // compact support: the polynomial is exactly 0 at h = 1
                let h = (r / len_rescaled).min(1.0);
                1.0 - 1.5 * h + 0.5 * h * h * h
            }
        }
    }

    /// `covariance(r) = var * correlation(r)`.
    pub fn covariance(&self, r: f64) -> f64 {
        self.var() * self.correlation(r)
    }

    /// Isotropic variogram respecting the nugget at `r = 0` (matches
    /// gstools' `vario_nugget`: exactly 0 at the origin, `var + nugget -
    /// covariance(r)` elsewhere).
    pub fn variogram(&self, r: f64) -> f64 {
        if r == 0.0 {
            0.0
        } else {
            self.var() + self.nugget() - self.covariance(r)
        }
    }

    /// Sill: `var + nugget`, i.e. `covariance(0)` including the nugget
    /// (matches gstools' `cov_nugget` at `r = 0`).
    pub fn sill(&self) -> f64 {
        self.var() + self.nugget()
    }

    /// `covariance(r)`, except at `r = 0` where the nugget is added back
    /// in (matches gstools' `cov_nugget`: `sill()` at the origin,
    /// `covariance(r)` everywhere else). Used for the "exact" kriging
    /// variant, where a target point coinciding with a conditioning point
    /// must reproduce that point's value exactly.
    pub fn cov_nugget(&self, r: f64) -> f64 {
        if r == 0.0 {
            self.sill()
        } else {
            self.covariance(r)
        }
    }
}

fn check_common(var: f64, len_scale: f64, nugget: f64) -> Result<(), ModelError> {
    if var < 0.0 {
        return Err(ModelError::InvalidVar(var));
    }
    if len_scale <= 0.0 {
        return Err(ModelError::InvalidLenScale(len_scale));
    }
    if nugget < 0.0 {
        return Err(ModelError::InvalidNugget(nugget));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(spec: CovModelSpec) -> CovModel {
        CovModel::build(spec).expect("valid spec")
    }

    #[test]
    fn correlation_is_one_at_origin() {
        for spec in [
            CovModelSpec::Gaussian {
                var: 2.0,
                len_scale: 3.0,
                nugget: 0.1,
            },
            CovModelSpec::Exponential {
                var: 2.0,
                len_scale: 3.0,
                nugget: 0.1,
            },
            CovModelSpec::Matern {
                var: 2.0,
                len_scale: 3.0,
                nugget: 0.1,
                nu: 2.5,
            },
            CovModelSpec::Spherical {
                var: 2.0,
                len_scale: 3.0,
                nugget: 0.1,
            },
        ] {
            let model = build(spec);
            assert!((model.correlation(0.0) - 1.0).abs() < 1e-12);
            assert!((model.covariance(0.0) - model.var()).abs() < 1e-12);
        }
    }

    #[test]
    fn matern_half_integer_matches_hand_derived_closed_forms() {
        // cor(h) = P_n(x) * exp(-x), x = sqrt(nu) * h, len_scale = 1 so
        // len_rescaled = h directly.
        type ClosedForm = fn(f64) -> f64;
        let cases: [(f64, ClosedForm); 3] = [
            (0.5, |x| (-x).exp()),
            (1.5, |x| (1.0 + x) * (-x).exp()),
            (2.5, |x| (1.0 + x + x * x / 3.0) * (-x).exp()),
        ];
        for (nu, closed_form) in cases {
            let model = build(CovModelSpec::Matern {
                var: 1.0,
                len_scale: 1.0,
                nugget: 0.0,
                nu,
            });
            for h in [0.1, 0.5, 1.0, 2.0, 5.0] {
                let x = nu.sqrt() * h;
                let expected = closed_form(x);
                let got = model.correlation(h);
                assert!(
                    (got - expected).abs() < 1e-10,
                    "nu={nu} h={h}: got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn matern_matches_gstools_reference_for_arbitrary_nu() {
        // Reference values from gstools' own `Matern.cor` (scipy.special
        // loggamma/kv), computed independently in Python. `nu=25.0` exceeds
        // the nu > 20 cutoff, so it exercises the Gaussian far-field
        // fallback branch instead of loggamma/kv.
        let cases: [(f64, [(f64, f64); 5]); 4] = [
            (
                0.3,
                [
                    (0.1, 0.833946507020263),
                    (0.5, 0.5818747118092827),
                    (1.0, 0.40558180689489187),
                    (2.0, 0.21171026561635017),
                    (5.0, 0.0350803447153922),
                ],
            ),
            (
                1.3,
                [
                    (0.1, 0.9913055665037315),
                    (0.5, 0.8601699813844405),
                    (1.0, 0.6366803171828707),
                    (2.0, 0.2917074599714008),
                    (5.0, 0.01714790657015862),
                ],
            ),
            (
                3.7,
                [
                    (0.1, 0.996583355039885),
                    (0.5, 0.9196887797390918),
                    (1.0, 0.727966937380819),
                    (2.0, 0.3296887012120744),
                    (5.0, 0.00832512681464761),
                ],
            ),
            (
                25.0,
                [
                    (0.1, 0.9975031223974601),
                    (0.5, 0.9394130628134758),
                    (1.0, 0.7788007830714049),
                    (2.0, 0.36787944117144233),
                    (5.0, 0.0019304541362277093),
                ],
            ),
        ];
        for (nu, points) in cases {
            let model = build(CovModelSpec::Matern {
                var: 1.0,
                len_scale: 1.0,
                nugget: 0.0,
                nu,
            });
            for (h, expected) in points {
                let got = model.correlation(h);
                assert!(
                    (got - expected).abs() < 1e-9,
                    "nu={nu} h={h}: got {got}, expected {expected}"
                );
            }
        }
    }
        #[test]
    fn spherical_matches_reference() {
        // cor(h) = 1 - 1.5 h + 0.5 h^3 for h < 1, else 0; rescale = 1, so
        // with len_scale = 2 the support ends at r = 2.
        let model = build(CovModelSpec::Spherical {
            var: 1.0,
            len_scale: 2.0,
            nugget: 0.0,
        });
        let cases = [
            (0.5, 0.6328125),
            (1.0, 0.3125),
            (-1.0, 0.3125),
            (1.8, 0.0145),
            (2.0, 0.0),
            (4.0, 0.0),
            (100.0, 0.0),
        ];
        for (r, expected) in cases {
            let got = model.correlation(r);
            assert!(
                (got - expected).abs() < 1e-12,
                "r={r}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn spherical_covariance_and_variogram_reach_sill_at_range() {
        let model = build(CovModelSpec::Spherical {
            var: 2.0,
            len_scale: 3.0,
            nugget: 0.5,
        });
        assert!((model.covariance(1.5) - 2.0 * 0.3125).abs() < 1e-12);
        assert_eq!(model.covariance(3.0), 0.0);
        assert_eq!(model.covariance(10.0), 0.0);
        assert_eq!(model.variogram(0.0), 0.0);
        assert!((model.variogram(10.0) - model.sill()).abs() < 1e-12);
        assert!((model.cov_nugget(0.0) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn spherical_builds_from_json() {
        let model = CovModel::from_json(
            r#"{"type": "spherical", "var": 1.5, "len_scale": 4.0, "nugget": 0.2}"#,
        )
        .expect("valid spherical JSON");
        assert_eq!(
            model,
            CovModel::Spherical {
                var: 1.5,
                len_rescaled: 4.0,
                nugget: 0.2,
            }
        );
    }

    #[test]
    fn spherical_rejects_invalid_len_scale() {
        let err = CovModel::build(CovModelSpec::Spherical {
            var: 1.0,
            len_scale: 0.0,
            nugget: 0.0,
        })
        .unwrap_err();
        assert_eq!(err, ModelError::InvalidLenScale(0.0));
    }

    #[test]
    fn rejects_non_positive_nu() {
        for nu in [0.0, -1.5] {
            let err = CovModel::build(CovModelSpec::Matern {
                var: 1.0,
                len_scale: 1.0,
                nugget: 0.0,
                nu,
            })
            .unwrap_err();
            assert_eq!(err, ModelError::InvalidNu(nu));
        }
    }

    #[test]
    fn correlation_decays_with_distance() {
        let model = build(CovModelSpec::Gaussian {
            var: 1.0,
            len_scale: 5.0,
            nugget: 0.0,
        });
        assert!(model.correlation(1.0) > model.correlation(5.0));
        assert!(model.correlation(5.0) > model.correlation(20.0));
        assert!(model.correlation(100.0) < 1e-6);
    }

    #[test]
    fn variogram_is_zero_at_origin_even_with_nugget() {
        let model = build(CovModelSpec::Exponential {
            var: 1.0,
            len_scale: 2.0,
            nugget: 0.3,
        });
        assert_eq!(model.variogram(0.0), 0.0);
        assert!((model.sill() - 1.3).abs() < 1e-12);
    }
}
