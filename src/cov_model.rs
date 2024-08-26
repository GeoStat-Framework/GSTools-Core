//! Provide Covariance Models

use plotters::prelude::*;
use special::Gamma;

use plotters::{backend::BitMapBackend, drawing::IntoDrawingArea};

/// Minimum functionality for a covariance model.
pub trait CovModel {
    /// Isotropic correlation function of the model
    fn correlation(&self, h: f64) -> f64;
    /// Getter for the variance
    fn var(&self) -> f64;
    /// Getter for the length scale
    fn len_scale(&self) -> f64;
}

/// Spectral density of the model
pub trait SpectralDensity {
    /// Spectral density of the model
    fn spectral_density(&self, k: f64) -> f64;
}

/// Isotropic variogram of the model
pub trait Variogram {
    /// Isotropic variogram of the model
    fn variogram(&self, r: f64) -> f64;
}

/// Isotropic covariance of the model
pub trait Covariance {
    /// Isotropic covariance of the model
    fn covariance(&self, r: f64) -> f64;
}

/// Base variables of a covariance model
pub struct BaseCovModel {
    /// spatial dimension
    pub dim: usize,
    /// variance of the model, not including the nugget
    pub var: f64,
    /// length scale of the model
    pub len_scale: f64,
}
/// Gaussian covariance model
pub struct Gaussian {
    /// covariance base variables
    pub base: BaseCovModel,
    /// rescale factor resulting in integral length scale
    pub rescale: f64,
}
/// Stable covariance model
pub struct Stable {
    /// covariance base variables
    pub base: BaseCovModel,
    /// shape parameter, valid values ɑ ∊ (0, 2]
    pub alpha: f64,
    /// rescale factor resulting in integral length scale
    pub rescale: f64,
}

impl Default for BaseCovModel {
    fn default() -> Self {
        Self {
            dim: 3,
            var: 1.0,
            len_scale: 1.0,
        }
    }
}
impl Default for Gaussian {
    fn default() -> Self {
        Self {
            base: BaseCovModel {
                dim: 3,
                var: 1.0,
                len_scale: 1.0,
            },
            rescale: std::f64::consts::PI.sqrt() / 2.0,
        }
    }
}
impl Default for Stable {
    fn default() -> Self {
        Self {
            base: BaseCovModel::default(),
            alpha: 1.5,
            rescale: 1.0,
        }
    }
}

impl CovModel for Gaussian {
    fn correlation(&self, h: f64) -> f64 {
        (-(h / self.len_scale()).powi(2)).exp()
    }
    fn var(&self) -> f64 {
        self.base.var
    }
    fn len_scale(&self) -> f64 {
        self.base.len_scale / self.rescale
    }
}
impl CovModel for Stable {
    #![allow(unstable_name_collisions)]
    fn correlation(&self, h: f64) -> f64 {
        (-(h / self.len_scale()).powf(self.alpha) * (1.0 + 1.0 / self.alpha).gamma()).exp()
    }
    fn var(&self) -> f64 {
        self.base.var
    }
    fn len_scale(&self) -> f64 {
        self.base.len_scale / self.rescale
    }
}

impl SpectralDensity for Gaussian {
    fn spectral_density(&self, k: f64) -> f64 {
        // TODO use std::f64::consts::FRAC_1_SQRT_PI, once it's stable
        self.base.len_scale / 2.0 / std::f64::consts::PI.sqrt().powi(self.base.dim as i32)
            * (-(k * self.base.len_scale / 2.0).powi(2)).exp()
    }
}

impl Variogram for Gaussian {
    fn variogram(&self, r: f64) -> f64 {
        // TODO nugget missing, add `+ self.nugget`
        self.base.var * (1.0 - self.correlation(r))
    }
}

impl Variogram for Stable {
    fn variogram(&self, r: f64) -> f64 {
        // TODO nugget missing, add `+ self.nugget`
        self.base.var * (1.0 - self.correlation(r))
    }
}

impl Covariance for Gaussian {
    fn covariance(&self, r: f64) -> f64 {
        self.base.var * self.correlation(r)
    }
}

impl Covariance for Stable {
    fn covariance(&self, r: f64) -> f64 {
        self.base.var * self.correlation(r)
    }
}

/// Plot the correlation function of a given correlation model.
pub fn plot_cor(model: Box<dyn CovModel>) {
    let root_drawing_area = BitMapBackend::new("cor.png", (1024, 768)).into_drawing_area();

    root_drawing_area.fill(&WHITE).unwrap();

    let x_max = model.len_scale() * 3.0;
    let mut chart = ChartBuilder::on(&root_drawing_area)
        .build_cartesian_2d(0.0..x_max, 0.0..1.2)
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            (0..1000)
                .map(|x| x as f64 / 1000.0 * x_max)
                .map(|x| (x, model.correlation(x))),
            &RED,
        ))
        .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_ulps_eq;

    #[test]
    fn test_plotting() {
        let gau = Box::new(Gaussian::default());
        plot_cor(gau);
    }
    #[test]
    fn test_covmodels() {
        let gau = Gaussian {
            base: BaseCovModel {
                dim: 1,
                var: 0.5,
                len_scale: 2.0,
            },
            ..Default::default()
        };
        assert_ulps_eq!(gau.correlation(0.0), 1.0, max_ulps = 6,);
        assert_ulps_eq!(gau.correlation(0.1), 0.9980384309875864, max_ulps = 6,);
        assert_ulps_eq!(gau.correlation(1.0), 0.8217249580338771, max_ulps = 6,);
        assert_ulps_eq!(gau.correlation(10.0), 2.96925699656481e-09, max_ulps = 6,);
        assert_ulps_eq!(gau.correlation(-0.5), 0.9520979267837046, max_ulps = 6,);

        assert_ulps_eq!(gau.covariance(0.0), 0.5, max_ulps = 6,);
        assert_ulps_eq!(gau.covariance(0.1), 0.4990192154937932, max_ulps = 6,);
        assert_ulps_eq!(gau.covariance(1.0), 0.41086247901693856, max_ulps = 6,);
        assert_ulps_eq!(gau.covariance(10.0), 1.4846284982824e-09, max_ulps = 6,);
        assert_ulps_eq!(gau.covariance(-0.5), 0.4760489633918523, max_ulps = 6,);

        // TODO test Stable model
    }
}
