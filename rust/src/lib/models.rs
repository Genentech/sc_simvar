use ndarray::prelude::*;
use ndarray_stats::{QuantileExt, interpolate};
use noisy_float::prelude::N64;
use ordered_float::OrderedFloat;

fn _calc_x2(mu: &Array1<f64>, var: &Array1<f64>) -> Array1<f64> {
    mu.mapv(|x| x.powi(2)) + var
}

/// Fits no model to the data.
///
/// # Arguments
///
/// * `gene_counts` - A 1D array of shape (n_genes,) containing the gene counts.
///
/// # Returns
///
/// * `mu` - A 1D array of shape (n_genes,) containing zeros.
/// * `var` - A 1D array of shape (n_genes,) containing ones.
/// * `x2` - A 1D array of shape (n_genes,) containing ones.
pub fn fit_none_model(gene_counts: &ArrayView1<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    (
        Array1::zeros(gene_counts.len()),
        Array1::ones(gene_counts.len()),
        Array1::ones(gene_counts.len()),
    )
}

/// Fits a Gaussian model to the data.
///
/// # Arguments
///
/// * `gene_counts` - A 1D array of shape (n_genes,) containing the gene counts.
/// * `umi_counts` - A 1D array of shape (n_genes,) containing the UMI counts.
///
/// # Returns
///
/// * `mu` - A 1D array of shape (n_genes,) containing the means of the Gaussian.
/// * `var` - A 1D array of shape (n_genes,) containing the variances of the Gaussian.
/// * `x2` - A 1D array of shape (n_genes,) containing the sum of the squared means and variances.
pub fn fit_normal_model(
    gene_counts: &ArrayView1<f64>,
    umi_counts: &ArrayView1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let n = gene_counts.len();

    if umi_counts.var(0.0).abs() < 1e-6 {
        let mean_val = gene_counts.mean().unwrap();
        let var_val = gene_counts.var(0.0);
        let mu = Array1::from_elem(n, mean_val);
        let var = Array1::from_elem(n, var_val);
        let x2 = _calc_x2(&mu, &var);
        (mu, var, x2)
    } else {
        // Create design matrix X = [1, umi_counts] more efficiently
        let mut x = Array2::uninit((n, 2));
        for i in 0..n {
            x[[i, 0]] = std::mem::MaybeUninit::new(1.0);
            x[[i, 1]] = std::mem::MaybeUninit::new(umi_counts[i]);
        }

        // Compute X^T * X manually (2x2 matrix)
        let xt_x_00 = n as f64;
        let xt_x_01 = umi_counts.sum();
        let xt_x_11 = umi_counts.mapv(|x| x * x).sum();

        // Compute X^T * y manually
        let xt_y_0 = gene_counts.sum();
        let xt_y_1 = umi_counts
            .iter()
            .zip(gene_counts.iter())
            .map(|(u, g)| u * g)
            .sum::<f64>();

        // Solve 2x2 system manually instead of matrix inversion
        let det = xt_x_00 * xt_x_11 - xt_x_01 * xt_x_01;
        let b0 = (xt_x_11 * xt_y_0 - xt_x_01 * xt_y_1) / det;
        let b1 = (xt_x_00 * xt_y_1 - xt_x_01 * xt_y_0) / det;

        // Compute mu = X * B
        let mu = Array1::from_iter((0..n).map(|i| b0 + b1 * umi_counts[i]));

        // Compute residual variance
        let var_val = gene_counts
            .iter()
            .zip(mu.iter())
            .map(|(g, m)| (g - m).powi(2))
            .sum::<f64>()
            / (n as f64);
        let var = Array1::from_elem(n, var_val);

        let x2 = _calc_x2(&mu, &var);

        (mu, var, x2)
    }
}

/// Fits a danb model to the data.
///
/// # Arguments
///
/// * `gene_counts` - A 1D array of shape (n_genes,) containing the gene counts.
/// * `umi_counts` - A 1D array of shape (n_genes,) containing the UMI counts.
///
/// # Returns
///
/// * `mu` - A 1D array of shape (n_genes,) containing the means of the danb.
/// * `var` - A 1D array of shape (n_genes,) containing the variances of the danb.
/// * `x2` - A 1D array of shape (n_genes,) containing the sum of the squared means and variances.
pub fn fit_danb_model(
    gene_counts: &ArrayView1<f64>,
    umi_counts: &ArrayView1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let n = gene_counts.len();
    let n_f64 = n as f64;

    // Actually compute sums correctly
    let tj = gene_counts.sum();
    let total = umi_counts.sum();
    let umi_sum_sq = umi_counts.mapv(|x| x * x).sum();

    // Pre-allocate and compute mu efficiently
    let mut mu = Array1::uninit(n);
    let tj_over_total = tj / total;
    for i in 0..n {
        mu[i] = std::mem::MaybeUninit::new(tj_over_total * umi_counts[i]);
    }
    let mu = unsafe { mu.assume_init() };

    // Compute variance manually to avoid temporary allocation
    let mut sum_sq_diff = 0.0;
    for i in 0..n {
        let diff = gene_counts[i] - mu[i];
        sum_sq_diff += diff * diff;
    }
    let vv = sum_sq_diff / (n_f64 - 1.0);

    // Compute size with fewer operations
    let numerator = (tj * tj / total) * (umi_sum_sq / total);
    let denominator = (n_f64 - 1.0) * vv - tj;

    let size = if denominator <= 0.0 {
        1e9
    } else {
        let s = numerator / denominator;
        if s < 1e-10 { 1e-10 } else { s }
    };

    // Compute var and x2 in single pass
    let mut var = Array1::uninit(n);
    let mut x2 = Array1::uninit(n);
    let inv_size = 1.0 / size;

    for i in 0..n {
        let mu_val = mu[i];
        let var_val = mu_val * (1.0 + mu_val * inv_size);
        var[i] = std::mem::MaybeUninit::new(var_val);
        x2[i] = std::mem::MaybeUninit::new(mu_val * mu_val + var_val);
    }

    let var = unsafe { var.assume_init() };
    let x2 = unsafe { x2.assume_init() };

    (mu, var, x2)
}

/// Fits a bernoulli model to the data.
///
/// # Arguments
///
/// * `gene_counts` - A 1D array of shape (n_genes,) containing the gene counts.
/// * `umi_counts` - A 1D array of shape (n_genes,) containing the UMI counts.
///
/// # Returns
///
/// * `mu` - A 1D array of shape (n_genes,) containing the means of the bernoulli.
/// * `var` - A 1D array of shape (n_genes,) containing the variances of the bernoulli.
/// * `x2` - A 1D array of shape (n_genes,) containing the sum of the squared means and variances.
pub fn fit_bernoulli_model(
    gene_counts: &ArrayView1<f64>,
    umi_counts: &ArrayView1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let n = umi_counts.len();
    let n_bins = if n < 30 { n } else { 30 };

    // Create quantiles and bin limits more efficiently
    let mut qs = Array1::uninit(n_bins + 1);
    let step = 1.0 / n_bins as f64;
    for i in 0..n_bins {
        qs[i] = std::mem::MaybeUninit::new(N64::new(i as f64 * step));
    }
    qs[n_bins] = std::mem::MaybeUninit::new(N64::new(1.0));
    let qs = unsafe { qs.assume_init() };

    // Pre-compute log10 values and convert to OrderedFloat
    let mut log_umi_counts = Array1::uninit(n);
    for i in 0..n {
        log_umi_counts[i] = std::mem::MaybeUninit::new(OrderedFloat(umi_counts[i].log10()));
    }
    let log_umi_counts = unsafe { log_umi_counts.assume_init() };

    let bin_limits = log_umi_counts
        .clone()
        .quantiles_axis_mut(Axis(0), &qs, &interpolate::Linear)
        .unwrap();

    // Bin assignment and bin centers computation combined
    let mut bin_detects = vec![1.0; n_bins];
    let mut bin_totals = vec![2.0; n_bins];
    let mut bin_centers = Vec::with_capacity(n_bins);

    // Compute bin centers
    for i in 0..n_bins {
        bin_centers.push((bin_limits[i] + bin_limits[i + 1]) / 2.0);
    }

    // Assign to bins and accumulate detection stats in one pass
    for i in 0..n {
        let log_val = log_umi_counts[i];
        // Find bin efficiently
        let mut bin_idx = 0;
        for j in 0..n_bins {
            if log_val >= bin_limits[j] && log_val <= bin_limits[j + 1] {
                bin_idx = j;
                break;
            }
        }

        bin_detects[bin_idx] += gene_counts[i];
        bin_totals[bin_idx] += 1.0;
    }

    // Compute logit detection rates
    let mut lbin_detects = Array1::uninit(n_bins);
    for i in 0..n_bins {
        let rate = bin_detects[i] / bin_totals[i];
        lbin_detects[i] = std::mem::MaybeUninit::new((rate / (1.0 - rate)).ln());
    }
    let lbin_detects = unsafe { lbin_detects.assume_init() };

    // Solve 2x2 linear system manually instead of matrix operations
    let mut sum_x = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;

    for i in 0..n_bins {
        let x = f64::from(bin_centers[i]);
        let y = lbin_detects[i];
        sum_x += x;
        sum_x2 += x * x;
        sum_y += y;
        sum_xy += x * y;
    }

    let n_bins_f64 = n_bins as f64;
    let det = n_bins_f64 * sum_x2 - sum_x * sum_x;
    let b0 = (sum_x2 * sum_y - sum_x * sum_xy) / det;
    let b1 = (n_bins_f64 * sum_xy - sum_x * sum_y) / det;

    // Compute mu, var, and x2 in single pass
    let mut mu = Array1::uninit(n);
    let mut var = Array1::uninit(n);

    for i in 0..n {
        let logit = b0 + b1 * f64::from(log_umi_counts[i]);
        let exp_logit = logit.exp();
        let mu_val = exp_logit / (1.0 + exp_logit);
        let var_val = mu_val * (1.0 - mu_val);

        mu[i] = std::mem::MaybeUninit::new(mu_val);
        var[i] = std::mem::MaybeUninit::new(var_val);
    }

    let mu = unsafe { mu.assume_init() };
    let var = unsafe { var.assume_init() };
    let x2 = mu.clone(); // For bernoulli, x2 = mu

    (mu, var, x2)
}
