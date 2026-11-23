//! Kriging: the global kriging matrix operations and local
//! (moving-neighborhood) kriging.
//!
//! ```text
//! global.rs        global kriging: contract a pre-inverted kriging matrix
//! local.rs         local kriging of one or many fields (weights cached
//!                  per target and neighbor mask)
//!
//! used by local kriging:
//! neighbors.rs     k-d tree radius search
//! solve.rs         linear algebra: Cholesky + Schur complement,
//!                  dense LU / SVD fallback
//!
//! test_util.rs     test helpers, incl. the reference system builders
//!                  build_lhs / build_rhs
//! ```
//!
//! Dependencies only point downwards: local kriging uses `neighbors` and
//! `solve`; `solve` knows nothing about kriging.
//!
//! The public API is re-exported here, so callers use
//! `crate::krige::calc_field_krige_local` etc.

use std::fmt;

pub mod global;
pub mod local;
mod neighbors;
pub mod solve;
#[cfg(test)]
mod test_util;

pub use global::{calculator_field_krige, calculator_field_krige_and_variance};
pub use local::{calc_field_krige_local, drift_terms_used, LocalKrigeResult, LocalKrigeStats};
pub use solve::{solve_dense, solve_kriging, SingularSystem};

/// Errors of local kriging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KrigeError {
    /// Inconsistent array shapes or non-finite positions / radius.
    InvalidInput(String),
    /// A local system could not be solved, not even by the least-squares
    /// fallback of [`solve_dense`] (in practice essentially unreachable).
    SingularSystem {
        /// index of the target point
        target_index: usize,
        /// index of the field (e.g. day)
        field_index: usize,
    },
}

impl fmt::Display for KrigeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::SingularSystem {
                target_index,
                field_index,
            } => write!(
                f,
                "local kriging matrix for target point {target_index} and field \
                 {field_index} is singular even for the least-squares fallback"
            ),
        }
    }
}

impl std::error::Error for KrigeError {}
