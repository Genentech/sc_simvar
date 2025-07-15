use ndarray::prelude::*;
use petal_neighbors::BallTree;
use rayon::prelude::*;

/// Returns the k nearest neighbors and distances for each point in latent.
///
/// # Arguments
///
/// * `latent` - A 2D array of shape (n_points, n_features) containing
///   the latent space coordinates.
/// * `k` - The number of nearest neighbors to return.
///
/// # Returns
///
/// * `indices` - A 2D array of shape (n_points, k) containing the indices
///   of the k nearest neighbors for each point in latent.
/// * `distances` - A 2D array of shape (n_points, k) containing the
///   distances to the k nearest neighbors for each point in latent.
pub fn knn_from_latent(latent: &ArrayView2<f64>, k: usize) -> (Array2<usize>, Array2<f64>) {
    let tree = match BallTree::euclidean(latent) {
        Ok(bt) => bt,
        Err(e) => panic!("Error: {}", e),
    };

    let n_points = latent.shape()[0];
    let mut indices = Vec::with_capacity(n_points * k);
    let mut distances = Vec::with_capacity(n_points * k);

    latent
        .outer_iter()
        .par_bridge()
        .map(|point| tree.query(&point, k))
        .collect::<Vec<_>>()
        .into_iter()
        .for_each(|(point_indices, point_distances)| {
            indices.extend(point_indices);
            distances.extend(point_distances);
        });

    let n_rows_cols = (n_points, k);

    (
        Array2::from_shape_vec(n_rows_cols, indices).unwrap(),
        Array2::from_shape_vec(n_rows_cols, distances).unwrap(),
    )
}

/// Returns the k nearest neighbors and distances for each point in distances.
///
/// # Arguments
///
/// * `distances` - A 2D array of shape (n_points, n_points) containing
///   the distances between each point.
/// * `k` - The number of nearest neighbors to return.
///
/// # Returns
///
/// * `indices` - A 2D array of shape (n_points, k) containing the indices
///   of the k nearest neighbors for each point in distances.
/// * `distances` - A 2D array of shape (n_points, k) containing the
///   distances to the k nearest neighbors for each point in distances.
pub fn knn_from_distances(distances: &ArrayView2<f64>, k: usize) -> (Array2<usize>, Array2<f64>) {
    let n_points = distances.shape()[0];
    let mut indices = Vec::with_capacity(n_points * k);
    let mut knn_distances = Vec::with_capacity(n_points * k);

    for i in 0..n_points {
        // Get distances from point i to all other points
        let row = distances.row(i);

        // Create vector of (distance, index) pairs
        let mut dist_idx: Vec<(f64, usize)> = row
            .iter()
            .enumerate()
            .map(|(idx, &dist)| (dist, idx))
            .collect();

        // Sort by distance, using stable sort for consistent ordering
        // When distances are equal, sklearn maintains original index order
        dist_idx.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        // Take the k nearest neighbors (including self if k allows)
        for (distance, index) in dist_idx.into_iter().take(k.min(n_points)) {
            indices.push(index);
            knn_distances.push(distance);
        }
    }

    let n_rows_cols = (n_points, k);

    (
        Array2::from_shape_vec(n_rows_cols, indices).unwrap(),
        Array2::from_shape_vec(n_rows_cols, knn_distances).unwrap(),
    )
}
