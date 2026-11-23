//! Radius search for local kriging: a k-d tree over the conditioning points.

use kdtree::{distance::squared_euclidean, KdTree};
use ndarray::{ArrayView1, ArrayView2};

/// k-d tree over the conditioning points with a fixed search radius.
pub(crate) struct Neighbors {
    tree: KdTree<f64, usize, Vec<f64>>,
    /// squared radius: the tree compares squared distances (no square roots)
    radius_sq: f64,
}

impl Neighbors {
    /// Build the tree over the columns of `cond_pos` (shape `(dim, n)`).
    pub(crate) fn new(cond_pos: ArrayView2<'_, f64>, radius: f64) -> Self {
        let mut tree = KdTree::new(cond_pos.nrows());
        for i in 0..cond_pos.ncols() {
            tree.add(cond_pos.column(i).to_vec(), i)
                .expect("cond_pos columns are finite and match `dim`");
        }
        Self {
            tree,
            radius_sq: radius * radius,
        }
    }

    /// Conditioning points within the radius of `target`, sorted by
    /// distance: `(indices, distances)`, both empty if there are none.
    pub(crate) fn within(&self, target: ArrayView1<'_, f64>) -> (Vec<usize>, Vec<f64>) {
        self.tree
            .within(&target.to_vec(), self.radius_sq, &squared_euclidean)
            .expect("target point is finite and matches `dim`")
            .into_iter()
            .map(|(sq_dist, &idx)| (idx, sq_dist.sqrt()))
            .unzip()
    }
}
