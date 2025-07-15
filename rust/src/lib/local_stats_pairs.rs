use crate::local_stats::compute_local_cov_max;
use crate::local_stats::compute_node_degree;
use crate::models::{fit_bernoulli_model, fit_danb_model, fit_none_model, fit_normal_model};
use float_cmp::approx_eq;
use itertools::Combinations;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use std::ops::Range;

// TODO: combine with calculate conditional eg2

/// Creates the centered counts.
///
/// # Arguments
///
/// * `counts` - A 2D array of shape (n_points, n_features) containing the counts.
/// * `model` - The model to use for fitting the data.
/// * `umi_counts` - A 1D array of shape (n_points,) containing the UMI counts.
///
/// # Returns
///
/// * `centered_counts` - A 2D array of shape (n_points, n_features) containing the centered counts.
pub fn create_centered_counts(
    counts: &ArrayView2<f64>,
    model: &str,
    umi_counts: &ArrayView1<f64>,
) -> Array2<f64> {
    let (n_genes, n_cells) = counts.dim();
    let mut centered_counts = Array2::zeros((n_genes, n_cells));

    centered_counts
        .outer_iter_mut()
        .into_par_iter()
        .zip(counts.outer_iter())
        .for_each(|(mut centered_row, row)| {
            let (mu, var, _) = match model {
                "bernoulli" => fit_bernoulli_model(&row.view(), umi_counts),
                "danb" => fit_danb_model(&row.view(), umi_counts),
                "normal" => fit_normal_model(&row.view(), umi_counts),
                "none" => fit_none_model(&row.view()),
                _ => panic!("Invalid model"),
            };

            // Compute centered values directly into the output array
            centered_row
                .iter_mut()
                .zip(row.iter())
                .zip(mu.iter())
                .zip(var.iter())
                .for_each(|(((centered_val, &count), &mean), &variance)| {
                    let std_dev = if approx_eq!(f64, variance, 0.0, ulps = 2) {
                        1.0
                    } else {
                        variance.sqrt()
                    };
                    *centered_val = (count - mean) / std_dev;
                });
        });

    centered_counts
}

/// Calculate the conditional E(G^2) for each feature.
///
/// # Arguments
///
/// * `counts` - A 2D array of shape (n_points, n_features) containing the counts.
/// * `neighbors` - A 2D array of shape (n_points, n_neighbors) containing the indices of the neighbors of
///   each point.
/// * `weights` - A 2D array of shape (n_points, n_neighbors) containing the weights of the edges between each
///   point and its neighbors.
///
/// # Returns
///
/// * `eg2s` - A 1D array of shape (n_features,) containing the conditional E(G^2) for each feature.
pub fn calculate_conditional_eg2(
    counts: &ArrayView2<f64>,
    neighbors: &ArrayView2<usize>,
    weights: &ArrayView2<f64>,
) -> Array1<f64> {
    let results: Vec<f64> = counts
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|x| {
            let n = neighbors.nrows();
            let k = neighbors.ncols();
            let mut t1x = Array1::zeros(n);

            for i in 0..n {
                for k_idx in 0..k {
                    let j = neighbors[(i, k_idx)];
                    let wij = weights[(i, k_idx)];

                    if approx_eq!(f64, wij, 0.0, ulps = 2) {
                        continue;
                    }

                    t1x[i] += wij * x[j];
                    t1x[j] += wij * x[i];
                }
            }

            t1x.mapv(|val: f64| val.powi(2)).sum()
        })
        .collect();

    Array1::from(results)
}

/// Compute the similar variance on centered values using conditional symmetry.
/// This combines local covariance calculation with z-score computation for better performance.
///
/// # Arguments
///
/// * `combo` - The combo of features to calculate the similar variance for.
/// * `counts` - A 2D array of shape (n_points, n_features) containing the counts.
/// * `neighbors` - A 2D array of shape (n_points, n_neighbors) containing the indices of the neighbors of
///   each point.
/// * `weights` - A 2D array of shape (n_points, n_neighbors) containing the weights of the edges between each
///   point and its neighbors.
/// * `eg2s` - A 1D array of shape (n_features,) containing the conditional E(G^2) for each feature.
///
/// # Returns
///
/// * `lcp` - The local covariance between feature i and feature j.
/// * `z` - The z-score for the local covariance between feature i and feature j.
pub fn compute_simvar_pairs_inner_centered_cond_sym(
    combo: &[usize],
    counts: &ArrayView2<f64>,
    neighbors: &ArrayView2<usize>,
    weights: &ArrayView2<f64>,
    eg2s: &ArrayView1<f64>,
) -> (f64, f64) {
    let (i, j) = (combo[0], combo[1]);

    // Get raw pointers for direct memory access
    let counts_ptr = counts.as_ptr();
    let neighbors_ptr = neighbors.as_ptr();
    let weights_ptr = weights.as_ptr();
    let (n, stride_counts) = (counts.ncols(), counts.strides()[0]);
    let (k, stride_neighbors, stride_weights) = (
        neighbors.ncols(),
        neighbors.strides()[0],
        weights.strides()[0],
    );

    // Pre-compute denominators
    let sqrt_eg2_i = eg2s[i].sqrt();
    let sqrt_eg2_j = eg2s[j].sqrt();

    let mut sum = 0.0;

    unsafe {
        // Cache row pointers for i and j
        let row_i_ptr = counts_ptr.offset(i as isize * stride_counts);
        let row_j_ptr = counts_ptr.offset(j as isize * stride_counts);

        for node_i in 0..n {
            let x_i = *row_i_ptr.add(node_i);
            let y_i = *row_j_ptr.add(node_i);

            // Skip if both values are effectively zero
            if x_i.abs() < f64::EPSILON && y_i.abs() < f64::EPSILON {
                continue;
            }

            let neighbors_row_ptr = neighbors_ptr.offset(node_i as isize * stride_neighbors);
            let weights_row_ptr = weights_ptr.offset(node_i as isize * stride_weights);

            // Unroll inner loop for better performance
            let mut k_idx = 0;
            while k_idx + 3 < k {
                for unroll_i in 0..4 {
                    let neighbor_j = *neighbors_row_ptr.add(k_idx + unroll_i);
                    let w_ij = *weights_row_ptr.add(k_idx + unroll_i);

                    if w_ij.abs() >= f64::EPSILON {
                        let x_j = *row_i_ptr.add(neighbor_j);
                        let y_j = *row_j_ptr.add(neighbor_j);

                        if x_j.abs() >= f64::EPSILON || y_j.abs() >= f64::EPSILON {
                            sum += w_ij * (x_i * y_j + y_i * x_j);
                        }
                    }
                }
                k_idx += 4;
            }

            // Handle remaining elements
            while k_idx < k {
                let neighbor_j = *neighbors_row_ptr.add(k_idx);
                let w_ij = *weights_row_ptr.add(k_idx);

                if w_ij.abs() >= f64::EPSILON {
                    let x_j = *row_i_ptr.add(neighbor_j);
                    let y_j = *row_j_ptr.add(neighbor_j);

                    if x_j.abs() >= f64::EPSILON || y_j.abs() >= f64::EPSILON {
                        sum += w_ij * (x_i * y_j + y_i * x_j);
                    }
                }
                k_idx += 1;
            }
        }
    }

    let lcp = sum;

    // Use faster absolute value comparison
    let z_xy = lcp / sqrt_eg2_i;
    let z_yx = lcp / sqrt_eg2_j;
    let z = if sqrt_eg2_i > sqrt_eg2_j { z_xy } else { z_yx };

    (lcp, z)
}

/// Expand the row combos into a 2D array.
///
/// # Arguments
///
/// * `row_combos` - The row combos to expand.
/// * `vals` - The values to insert into the 2D array.
/// * `n` - The number of rows and columns in the 2D array.
///
/// # Returns
///
/// * `out` - The expanded 2D array.
fn _expand_row_combos(
    row_combos: Combinations<Range<usize>>,
    vals: Vec<f64>,
    n: usize,
) -> Array2<f64> {
    let mut out: Array2<f64> = Array2::zeros((n, n));

    // Use unsafe access for better performance
    unsafe {
        let out_ptr = out.as_mut_ptr();
        let vals_ptr = vals.as_ptr();

        for (i, combo) in row_combos.into_iter().enumerate() {
            let x = combo[0];
            let y = combo[1];
            let val = *vals_ptr.add(i);

            // Direct memory access to avoid bounds checking
            *out_ptr.add(x * n + y) = val;
            *out_ptr.add(y * n + x) = val;
        }
    }

    out
}

/// Compute the maximum local covariance between each pair of features.
///
/// # Arguments
///
/// * `node_degrees` - A 1D array of shape (n_points,) containing the node degrees.
/// * `counts` - A 2D array of shape (n_points, n_features) containing the counts.
///
/// # Returns
///
/// * `lcps` - A 2D array of shape (n_features, n_features) containing the local covariance between each pair
///   of features.
pub fn compute_local_cov_pairs_max(
    node_degrees: &ArrayView1<f64>,
    counts: &ArrayView2<f64>,
) -> Array2<f64> {
    let n_genes = counts.nrows();

    // Pre-compute gene maximums in parallel
    let gene_maximums: Vec<f64> = counts
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| compute_local_cov_max(node_degrees, &row))
        .collect();

    // Pre-allocate output array and fill it directly
    let mut result = Array2::zeros((n_genes, n_genes));

    // Use parallel iteration with safe indexing
    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let max_i = gene_maximums[i];
            for (j, val) in row.iter_mut().enumerate() {
                let max_j = gene_maximums[j];
                *val = (max_i + max_j) * 0.5;
            }
        });

    result
}

fn _count_combinations(n: u64, r: u64) -> u64 {
    if r > n {
        0
    } else {
        (1..=r.min(n - r)).fold(1, |acc, val| acc * (n - val + 1) / val)
    }
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
///
/// # Returns
///
/// * `lcps` - A 2D array of shape (n_features, n_features) containing the local covariance between each pair
///   of features.
/// * `zs` - A 2D array of shape (n_features, n_features) containing the z-scores.
pub fn compute_simvar_pairs_centered_cond(
    counts: &ArrayView2<f64>,
    neighbors: &ArrayView2<usize>,
    weights: &ArrayView2<f64>,
    umi_counts: &ArrayView1<f64>,
    model: &str,
) -> (Array2<f64>, Array2<f64>) {
    let node_degrees = compute_node_degree(neighbors, weights);
    let counts = create_centered_counts(counts, model, umi_counts);
    let eg2s = calculate_conditional_eg2(&counts.view(), neighbors, weights);
    let n = counts.nrows();

    // Pre-compute max values for normalization
    let max_values = compute_local_cov_pairs_max(&node_degrees.view(), &counts.view());

    // Create views that can be safely shared across threads
    let counts_view = counts.view();
    let eg2s_view = eg2s.view();
    let max_values_view = max_values.view();

    // Collect results in parallel
    let results: Vec<((usize, usize), (f64, f64))> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            ((i + 1)..n).into_par_iter().map(move |j| {
                let combo = [i, j];
                let (lcp, z) = compute_simvar_pairs_inner_centered_cond_sym(
                    &combo,
                    &counts_view,
                    neighbors,
                    weights,
                    &eg2s_view,
                );

                // Normalize lcp directly
                let normalized_lcp = lcp / max_values_view[[i, j]];
                ((i, j), (normalized_lcp, z))
            })
        })
        .collect();

    // Pre-allocate output arrays
    let mut lcps = Array2::zeros((n, n));
    let mut zs = Array2::zeros((n, n));

    // Fill output arrays with results
    for ((i, j), (normalized_lcp, z)) in results {
        lcps[[i, j]] = normalized_lcp;
        lcps[[j, i]] = normalized_lcp;
        zs[[i, j]] = z;
        zs[[j, i]] = z;
    }

    (lcps, zs)
}
