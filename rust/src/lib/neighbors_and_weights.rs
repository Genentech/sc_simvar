use ndarray::prelude::*;

use crate::knn::{knn_from_distances, knn_from_latent};
use float_cmp::approx_eq;

/// Returns the k nearest neighbors and distances for each point in data.
/// The data can be either the latent space coordinates or the distances
/// between each point in the latent space.
///
/// # Arguments
///
/// * `data` - A 2D array of shape (n_points, n_features) containing the type of data specified by kind.
/// * `k` - The number of nearest neighbors to get with distances.
/// * `n_factor` - The number of neighbors to use for calculating weights using a Gaussian kernel.
/// * `kind` - The kind of data. Either `Kind::Latent` or `Kind::Distances`.
///
/// # Returns
///
/// * `neighbors` - A 2D array of shape (n_points, k) containing the indices of the k nearest neighbors for
///   each point in data.
/// * `weights` - A 2D array of shape (n_points, k) containing the weights for each point in data.
pub fn make_neighbors_and_weights(
    data: &ArrayView2<f64>,
    k: usize,
    n_factor: usize,
    kind: &str,
) -> (Array2<usize>, Array2<f64>) {
    match kind {
        "latent" => make_neighbors_and_weights_from_latent(data, k, n_factor),
        "distances" => make_neighbors_and_weights_from_distances(data, k, n_factor),
        _ => panic!("Invalid kind"),
    }
}

/// Returns the k nearest neighbors and distances for each point in latent.
///
/// # Arguments
///
/// * `latent` - A 2D array of shape (n_points, n_features) containing the latent space coordinates.
/// * `k` - The number of nearest neighbors to get with distances.
/// * `n_factor` - The number of neighbors to use for calculating weights using a Gaussian kernel.
///
/// # Returns
///
/// * `neighbors` - A 2D array of shape (n_points, k) containing the indices of the k nearest neighbors for each point in latent.
/// * `weights` - A 2D array of shape (n_points, k) containing the weights for each point in latent.
fn make_neighbors_and_weights_from_latent(
    latent: &ArrayView2<f64>,
    k: usize,
    n_factor: usize,
) -> (Array2<usize>, Array2<f64>) {
    let (neighbors, distances) = knn_from_latent(latent, k);

    let weights = compute_weights(&distances.view(), n_factor);

    (neighbors, weights)
}

/// Returns the k nearest neighbors and distances for each point in distances.
///
/// # Arguments
///
/// * `distances` - A 2D array of shape (n_points, n_points) containing the distances between each point in
///   latent.
/// * `k` - The number of nearest neighbors to get with distances.
/// * `n_factor` - The number of neighbors to use for calculating weights
///
/// # Returns
///
/// * `neighbors` - A 2D array of shape (n_points, k) containing the indices of the k nearest neighbors for
///   each point in distances.
fn make_neighbors_and_weights_from_distances(
    distances: &ArrayView2<f64>,
    k: usize,
    n_factor: usize,
) -> (Array2<usize>, Array2<f64>) {
    let (neighbors, distances) = knn_from_distances(distances, k);

    let weights = compute_weights(&distances.view(), n_factor);

    (neighbors, weights)
}

/// Computes the weights for each point in latent.
///
/// The weights are computed as a Gaussian kernel of the distances
/// to the k nearest neighbors.
///
/// The weights are normalized such that the sum of the weights
/// for each point is 1.
///
/// # Arguments
///
/// * `distances` - A 2D array of shape (n_points, n_neighbors) containing the distances to the k nearest
///   neighbors for each point in latent.
/// * `n_factor` - The number of neighbors to use for the Gaussian kernel.
/// * `neighbors` - A 2D array of shape (n_points, n_neighbors) containing the indices of the k nearest
///   neighbors for each point in latent.
///
/// # Returns
///
/// * `weights` - A 2D array of shape (n_points, n_neighbors) containing the weights for each point in latent.
pub fn compute_weights(distances: &ArrayView2<f64>, n_factor: usize) -> Array2<f64> {
    let (n_points, n_neighbors) = distances.dim();
    let rad_ii = n_neighbors / n_factor + if n_neighbors % n_factor > 0 { 1 } else { 0 };

    let mut weights = Array2::<f64>::zeros((n_points, n_neighbors));

    for (i, d_row) in distances.outer_iter().enumerate() {
        // Calculate sigma for this row
        let sigma = if approx_eq!(f64, d_row[rad_ii - 1].abs(), 0.0, ulps = 2) {
            1.0
        } else {
            d_row[rad_ii - 1].powi(2)
        };

        // Calculate weights for this row
        let mut row_sum = 0.0;
        for (j, &d) in d_row.iter().enumerate() {
            let w = (-d.powi(2) / sigma).exp();
            weights[(i, j)] = w;
            row_sum += w;
        }

        // Normalize weights for this row
        let norm_factor = if approx_eq!(f64, row_sum, 0.0, ulps = 2) {
            1.0
        } else {
            1.0 / row_sum
        };

        for j in 0..n_neighbors {
            weights[(i, j)] *= norm_factor;
        }
    }

    weights
}

/// Makes the weights non-redundant by summing the weights of the
/// symmetric neighbors.
///
/// # Arguments
///
/// * `weights` - A 2D array of shape (n_points, n_neighbors) containing the weights for each point in latent.
/// * `neighbors` - A 2D array of shape (n_points, n_neighbors) containing the indices of the k nearest
///   neighbors for each point in latent.
///
/// # Returns
///
/// * `weights` - A 2D array of shape (n_points, n_neighbors) containing the weights for each point in latent.
pub fn make_weights_non_redundant(
    mut weights: Array2<f64>,
    neighbors: &ArrayView2<usize>,
) -> Array2<f64> {
    for (i, n_row) in neighbors.outer_iter().enumerate() {
        for (j, &nbr) in n_row.iter().enumerate() {
            if nbr < i {
                continue;
            }

            // Find the index where nbr points back to i
            if let Some(j2) = neighbors.row(nbr).iter().position(|&x| x == i) {
                let w_nbr_j2 = weights[(nbr, j2)];
                weights[(nbr, j2)] = 0.0;
                weights[(i, j)] += w_nbr_j2;
            }
        }
    }

    weights
}
