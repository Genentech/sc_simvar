use crate::models::{fit_bernoulli_model, fit_danb_model, fit_none_model, fit_normal_model};
use crate::utils::_reorder;
use crate::utils::square_iterable;
use adjustp::{Procedure, adjust};
use float_cmp::approx_eq;
use indicatif::ParallelProgressIterator;
use ndarray::{Array1, ArrayView1, ArrayView2, Zip};
use numpy::PyFixedUnicode;
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};

/// Computes the node degree for each node in the graph.
///
/// # Arguments
///
/// * `neighbors` - A 2D array of shape (n_points, n_neighbors) containing the indices of the neighbors of
///   each point.
/// * `weights` - A 2D array of shape (n_points, n_neighbors) containing the weights of the edges between each
///   point and its neighbors.
///
/// # Returns
///
/// * `node_degree` - A 1D array of shape (n_points,) containing the node degree for each node in the graph.
pub fn compute_node_degree(
    neighbors: &ArrayView2<usize>,
    weights: &ArrayView2<f64>,
) -> Array1<f64> {
    let n_points = neighbors.nrows();

    let mut node_degree = Array1::zeros(n_points);

    for i in 0..n_points {
        for k in 0..neighbors.ncols() {
            let j = neighbors[[i, k]];
            let w_ij = weights[[i, k]];
            node_degree[i] += w_ij;
            node_degree[j] += w_ij;
        }
    }

    node_degree
}

/// Centers the provided values using mu and var.
///
/// # Arguments
///
/// * `vals` - A 1D array of shape (n_points,) containing the values to be centered.
/// * `mu` - A 1D array of shape (n_points,) containing the means of the values to be centered.
/// * `var` - A 1D array of shape (n_points,) containing the variances of the values to be centered.
///
/// # Returns
///
/// * `centered_vals` - A 1D array of shape (n_points,) containing the centered values.
pub fn center_values(
    vals: &ArrayView1<f64>,
    mu: &ArrayView1<f64>,
    var: &ArrayView1<f64>,
) -> Array1<f64> {
    let mut centered = Array1::<f64>::zeros(vals.len());
    Zip::from(&mut centered)
        .and(vals)
        .and(mu)
        .and(var)
        .for_each(|c, &v, &m, &va| {
            *c = if approx_eq!(f64, va, 0.0, ulps = 2) {
                0.0
            } else {
                (v - m) / va.sqrt()
            }
        });
    centered
}

/// Computes the local covariance between each node and its neighbors.
///
/// # Arguments
///
/// * `vals` - A 1D array of shape (n_points,) containing the values to calculate the local covariance for.
/// * `neighbors` - A 2D array of shape (n_points, n_neighbors) containing the indices of the neighbors of each
///   point.
/// * `weights` - A 2D array of shape (n_points, n_neighbors) containing the weights of the edges between each
///   point and its neighbors.
///
/// # Returns
///
/// * `local_cov` - The local covariance between a node and its neighbors.
pub fn local_cov_weights(
    vals: &ArrayView1<f64>,
    neighbors: &ArrayView2<usize>,
    weights: &ArrayView2<f64>,
) -> f64 {
    let n_points = neighbors.nrows();
    let n_neighbors = neighbors.ncols();

    // Use direct memory access for better performance
    let vals_ptr = vals.as_ptr();
    let neighbors_ptr = neighbors.as_ptr();
    let weights_ptr = weights.as_ptr();

    let mut sum = 0.0;

    unsafe {
        for i in 0..n_points {
            let x = *vals_ptr.add(i);

            // Skip early if x is zero
            if x == 0.0 {
                continue;
            }

            let base = i * n_neighbors;
            for k in 0..n_neighbors {
                let idx = base + k;
                let w_ij = *weights_ptr.add(idx);

                // Skip if weight is zero
                if w_ij == 0.0 {
                    continue;
                }

                let j = *neighbors_ptr.add(idx);
                let y = *vals_ptr.add(j);

                // Skip if y is zero
                if y == 0.0 {
                    continue;
                }

                sum += x * y * w_ij;
            }
        }
    }

    sum
}

/// Compute the moments of the weights.
///
/// # Arguments
///
/// * `mu` - A 1D array of shape (n_points,) containing the means of the values to calculate the moments for.
/// * `x2` - A 1D array of shape (n_points,) containing a combination of the means and variance.
/// * `neighbors` - A 2D array of shape (n_points, n_neighbors) containing the indices of the neighbors of
///   each point.
/// * `weights` - A 2D array of shape (n_points, n_neighbors) containing the weights of the edges between each
///   point and its neighbors.
///
/// # Returns
///
/// * `eg` - The first moment of the weights.
/// * `eg2` - The second moment of the weights.
pub fn compute_moments_weights(
    mu: &ArrayView1<f64>,
    x2: &ArrayView1<f64>,
    neighbors: &ArrayView2<usize>,
    weights: &ArrayView2<f64>,
) -> (f64, f64) {
    let n_points = neighbors.nrows();
    let mu_2 = square_iterable(mu);

    // Preallocate accumulators
    let mut eg = 0.0;
    let mut eg2 = 0.0;
    let mut t1 = vec![0.0; n_points];
    let mut t2 = vec![0.0; n_points];

    // Iterate efficiently over all edges
    for i in 0..n_points {
        let mu_i = mu[i];
        let mu2_i = mu_2[i];
        let x2_i = x2[i];
        for k in 0..neighbors.ncols() {
            let j = neighbors[[i, k]];
            let w_ij = weights[[i, k]];
            if approx_eq!(f64, w_ij, 0.0, ulps = 2) {
                continue;
            }
            let mu_j = mu[j];
            let mu2_j = mu_2[j];
            let x2_j = x2[j];
            let w_ij2 = w_ij * w_ij;

            // eg and eg2
            eg += w_ij * mu_i * mu_j;
            eg2 += w_ij2 * (x2_i * x2_j - mu2_i * mu2_j);

            // t1 and t2 accumulators
            t1[i] += w_ij * mu_j;
            t2[i] += w_ij2 * mu2_j;
            t1[j] += w_ij * mu_i;
            t2[j] += w_ij2 * mu2_i;
        }
    }

    // Square t1 in place
    for v in &mut t1 {
        *v = *v * *v;
    }

    // Final eg2 correction
    for i in 0..n_points {
        eg2 += (x2[i] - mu_2[i]) * (t1[i] - t2[i]);
    }

    (eg, eg2 + eg * eg)
}

/// Computes the max local covariance between each node and its neighbors.
///
/// # Arguments
///
/// * `node_degrees` - A 1D array of shape (n_points,) containing the node degree for each node in the graph.
/// * `row` - A 1D array of shape (n_points,) containing the values to calculate the local covariance for.
///
/// # Returns
///
/// * `local_cov_max` - The max local covariance between a node and its neighbors.
pub fn compute_local_cov_max(node_degrees: &ArrayView1<f64>, row: &ArrayView1<f64>) -> f64 {
    node_degrees
        .iter()
        .zip(row.iter())
        .filter_map(|(x, y)| {
            if approx_eq!(f64, *x, 0.0, ulps = 2) || approx_eq!(f64, *y, 0.0, ulps = 2) {
                None
            } else {
                Some(x * y.powi(2))
            }
        })
        .sum::<f64>()
        / 2.0
}

/// Compute the similar variance for each feature.
///
/// # Arguments
///
/// * `counts` - A 2D array of shape (n_points, n_features) containing the counts.
/// * `neighbors` - A 2D array of shape (n_points, n_neighbors) containing the indices of the neighbors of
///   each point.
/// * `weights` - A 2D array of shape (n_points, n_neighbors) containing the weights of the edges between each
///   point and its neighbors.
/// * `umi_counts` - A 1D array of shape (n_points,) containing the UMI counts.
/// * `model` - The model to use for fitting the data.
/// * `features` - A 1D array of shape (n_features,) containing the names of the features.
/// * `centered` - Whether to center the data.
///
/// # Returns
///
/// * `features` - A 1D array of shape (n_features,) containing the names of the features.
/// * `c` - A 1D array of shape (n_features,) containing the local covariance between a node and its neighbors.
/// * `z` - A 1D array of shape (n_features,) containing the z-scores.
/// * `p_values` - A 1D array of shape (n_features,) containing the p-values.
/// * `fdr` - A 1D array of shape (n_features,) containing the FDR corrected p-values.
#[allow(clippy::type_complexity)]
pub fn compute_simvar(
    counts: &ArrayView2<f64>,
    neighbors: &ArrayView2<usize>,
    weights: &ArrayView2<f64>,
    umi_counts: &ArrayView1<f64>,
    model: &str,
    features: &ArrayView1<PyFixedUnicode<25>>,
    centered: bool,
) -> (
    Array1<PyFixedUnicode<25>>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let node_degrees = compute_node_degree(neighbors, weights);
    let w_2_total = weights.iter().map(|x| x.powi(2)).sum();

    let mut index_c_z: Vec<_> = counts
        .outer_iter()
        .enumerate()
        .par_bridge()
        .progress_count(counts.nrows() as u64)
        .map(|(i, row)| {
            let (mu, var, x2) = match model {
                "bernoulli" => fit_bernoulli_model(&row, umi_counts),
                "danb" => fit_danb_model(&row, umi_counts),
                "normal" => fit_normal_model(&row, umi_counts),
                "none" => fit_none_model(&row),
                _ => panic!("Invalid model"),
            };

            // TODO: implement other compute moments weights and expose as options
            let (row, (eg, eg_2)) = if centered {
                (
                    center_values(&row.view(), &mu.view(), &var.view()),
                    (0.0, w_2_total),
                )
            } else {
                (
                    row.to_owned(), // TODO: figure out how to get around this clone
                    compute_moments_weights(&mu.view(), &x2.view(), neighbors, weights),
                )
            };

            let g = local_cov_weights(&row.view(), neighbors, weights);

            let std_g = (eg_2 - eg.powi(2)).sqrt();

            let c = (g - eg) / compute_local_cov_max(&node_degrees.view(), &row.view());

            (i, (c, (g - eg) / std_g))
        })
        .collect();

    index_c_z.sort_by_key(|&(i, _)| i);

    let (_, c_z): (Vec<_>, Vec<_>) = index_c_z.into_iter().unzip();
    let (c, z): (Vec<_>, Vec<_>) = c_z.into_iter().unzip();

    let dist = Normal::new(0.0, 1.0).unwrap();
    let p_values: Vec<_> = z.iter().map(|x| dist.sf(*x)).collect();
    let fdr = adjust(&p_values, Procedure::BenjaminiHochberg);

    let mut order: Vec<_> = (0..z.len()).collect();
    order.sort_by(|a, b| z[*a].partial_cmp(&z[*b]).unwrap().reverse());

    (
        _reorder(&order, features.to_vec()),
        _reorder(&order, c),
        _reorder(&order, z),
        _reorder(&order, p_values),
        _reorder(&order, fdr),
    )
}
