use std::array::IntoIter;
use std::ops::{Index, IndexMut};

use ndarray::ArrayView1;

/// Represents an element of the vector space R^N with N fixed at compile time.
///
/// The increased alignment to 16 bytes makes these amenable for SIMD processing through auto-vectorization.
#[derive(Clone, Copy)]
#[repr(align(16))]
pub struct ShortVec<const N: usize>([f64; N]);

impl<const N: usize> ShortVec<N> {
    /// Create an instance of the origin, i.e. a vector with all components equal to zero.
    pub fn zero() -> Self {
        Self([0.0; N])
    }

    /// Create an instance whose components are the first N values of the given one-dimensional array view.
    pub fn from_array(array: ArrayView1<'_, f64>) -> Self {
        let mut this = Self::zero();

        (0..N).for_each(|idx| {
            this.0[idx] = array[idx];
        });

        this
    }

    /// Compute the inner product of `self` with `other`.
    pub fn dot(&self, other: &Self) -> f64 {
        (0..N).map(|idx| self.0[idx] * other.0[idx]).sum()
    }

    /// Add `other` to `self` which is updated in-place.
    pub fn add(&mut self, other: &Self) {
        (0..N).for_each(|idx| {
            self.0[idx] += other.0[idx];
        });
    }
}

impl<const N: usize> IntoIterator for ShortVec<N> {
    type Item = f64;
    type IntoIter = IntoIter<f64, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const N: usize> Index<usize> for ShortVec<N> {
    type Output = f64;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.0[idx]
    }
}

impl<const N: usize> IndexMut<usize> for ShortVec<N> {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.0[idx]
    }
}
