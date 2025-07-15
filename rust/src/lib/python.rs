use crate::knn::{knn_from_distances as kfd, knn_from_latent as kfl};
use crate::local_stats::{
    center_values as cv, compute_local_cov_max as clcm, compute_moments_weights as cmw,
    compute_node_degree as cnd, compute_simvar as comp_sv, local_cov_weights as lcw,
};
use crate::local_stats_pairs::{
    calculate_conditional_eg2 as cce, compute_local_cov_pairs_max as clcpm,
    compute_simvar_pairs_centered_cond as comp_sv_pcc,
    compute_simvar_pairs_inner_centered_cond_sym as chpicccs, create_centered_counts as ccc,
};
use crate::models::{
    fit_bernoulli_model as fbernm, fit_danb_model as fdanbm, fit_none_model as fnm,
    fit_normal_model as fnormalm,
};
use crate::neighbors_and_weights::{
    compute_weights as cw, make_neighbors_and_weights as make_n_w,
    make_weights_non_redundant as make_w_nr,
};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyFixedUnicode, PyReadonlyArray1, PyReadonlyArray2, ToPyArray,
};
use pyo3::prelude::*;

#[pyfunction]
fn make_neighbors_and_weights<'py>(
    py: Python<'py>,
    latent: PyReadonlyArray2<'py, f64>,
    k: usize,
    n_factor: usize,
    kind: &str,
) -> (Bound<'py, PyArray2<usize>>, Bound<'py, PyArray2<f64>>) {
    let latent = latent.as_array();

    let (neighbors, weights) = make_n_w(&latent, k, n_factor, kind);

    (neighbors.into_pyarray(py), weights.into_pyarray(py))
}

#[pyfunction]
fn make_weights_non_redundant<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray2<'py, f64>,
    neighbors: PyReadonlyArray2<'py, usize>,
) -> Bound<'py, PyArray2<f64>> {
    let weights = weights.as_array();

    let neighbors = neighbors.as_array();

    make_w_nr(weights.to_owned(), &neighbors).into_pyarray(py)
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
#[pyfunction]
fn compute_simvar<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray2<'py, f64>,
    neighbors: PyReadonlyArray2<'py, usize>,
    weights: PyReadonlyArray2<'py, f64>,
    umi_counts: PyReadonlyArray1<'py, f64>,
    features: PyReadonlyArray1<'py, PyFixedUnicode<25>>,
    model: &'py str,
    centered: bool,
) -> (
    Bound<'py, PyArray1<PyFixedUnicode<25>>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
) {
    let counts = counts.as_array();
    let neighbors = neighbors.as_array();
    let weights = weights.as_array();
    let umi_counts = umi_counts.as_array();
    let features = features.as_array();

    let (features, c, z, p_values, fdr) = comp_sv(
        &counts,
        &neighbors,
        &weights,
        &umi_counts,
        model,
        &features,
        centered,
    );

    (
        features.into_pyarray(py),
        c.into_pyarray(py),
        z.into_pyarray(py),
        p_values.into_pyarray(py),
        fdr.into_pyarray(py),
    )
}

#[pyfunction]
fn compute_simvar_pairs_centered_cond<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray2<'py, f64>,
    neighbors: PyReadonlyArray2<'py, usize>,
    weights: PyReadonlyArray2<'py, f64>,
    umi_counts: PyReadonlyArray1<'py, f64>,
    model: &'py str,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>) {
    let counts = counts.as_array();
    let neighbors = neighbors.as_array();
    let weights = weights.as_array();
    let umi_counts = umi_counts.as_array();

    let (lcps, zs) = comp_sv_pcc(&counts, &neighbors, &weights, &umi_counts, model);

    (lcps.into_pyarray(py), zs.into_pyarray(py))
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
#[pyfunction]
fn compute_simvar_and_pairs<'py>(
    py: Python<'py>,
    all_counts: PyReadonlyArray2<'py, f64>,
    sub_counts: PyReadonlyArray2<'py, f64>,
    neighbors: PyReadonlyArray2<'py, usize>,
    weights: PyReadonlyArray2<'py, f64>,
    umi_counts: PyReadonlyArray1<'py, f64>,
    features: PyReadonlyArray1<'py, PyFixedUnicode<25>>,
    model: &'py str,
    centered: bool,
) -> (
    Bound<'py, PyArray1<PyFixedUnicode<25>>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
) {
    let all_counts = all_counts.as_array();
    let sub_counts = sub_counts.as_array();
    let neighbors = neighbors.as_array();
    let weights = weights.as_array();
    let umi_counts = umi_counts.as_array();
    let features = features.as_array();

    let (features, c, z, p_values, fdr) = comp_sv(
        &all_counts,
        &neighbors,
        &weights,
        &umi_counts,
        model,
        &features,
        centered,
    );

    let (lcps, zs) = comp_sv_pcc(&sub_counts, &neighbors, &weights, &umi_counts, model);

    (
        features.into_pyarray(py),
        c.into_pyarray(py),
        z.into_pyarray(py),
        p_values.into_pyarray(py),
        fdr.into_pyarray(py),
        lcps.into_pyarray(py),
        zs.into_pyarray(py),
    )
}

#[pyfunction]
fn knn_from_latent<'py>(
    py: Python<'py>,
    latent: PyReadonlyArray2<'py, f64>,
    k: usize,
) -> (Bound<'py, PyArray2<usize>>, Bound<'py, PyArray2<f64>>) {
    let latent = latent.as_array();

    let (indices, distances) = kfl(&latent, k);

    (indices.into_pyarray(py), distances.into_pyarray(py))
}

#[pyfunction]
fn knn_from_distances<'py>(
    py: Python<'py>,
    distances: PyReadonlyArray2<'py, f64>,
    k: usize,
) -> (Bound<'py, PyArray2<usize>>, Bound<'py, PyArray2<f64>>) {
    let distances = distances.as_array();

    let (indices, distances) = kfd(&distances, k);

    (indices.into_pyarray(py), distances.into_pyarray(py))
}

#[pyfunction]
fn create_centered_counts<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray2<'py, f64>,
    model: &str,
    umi_counts: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray2<f64>> {
    let counts = counts.as_array();
    let umi_counts = umi_counts.as_array();

    ccc(&counts, model, &umi_counts).into_pyarray(py)
}

#[pyfunction]
fn calculate_conditional_eg2<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray2<'py, f64>,
    neighbors: PyReadonlyArray2<'py, usize>,
    weights: PyReadonlyArray2<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let counts = counts.as_array();
    let neighbors = neighbors.as_array();
    let weights = weights.as_array();

    cce(&counts, &neighbors, &weights).into_pyarray(py)
}

#[pyfunction]
fn compute_simvar_pairs_inner_centered_cond_sym<'py>(
    _py: Python<'py>,
    combo: Vec<usize>,
    counts: PyReadonlyArray2<'py, f64>,
    neighbors: PyReadonlyArray2<'py, usize>,
    weights: PyReadonlyArray2<'py, f64>,
    eg2s: PyReadonlyArray1<'py, f64>,
) -> (f64, f64) {
    let counts = counts.as_array();
    let neighbors = neighbors.as_array();
    let weights = weights.as_array();
    let eg2s = eg2s.as_array();

    chpicccs(&combo, &counts, &neighbors, &weights, &eg2s)
}

#[pyfunction]
fn compute_local_cov_pairs_max<'py>(
    py: Python<'py>,
    node_degrees: PyReadonlyArray1<'py, f64>,
    counts: PyReadonlyArray2<'py, f64>,
) -> Bound<'py, PyArray2<f64>> {
    let node_degrees = node_degrees.as_array();
    let counts = counts.as_array();

    clcpm(&node_degrees, &counts).into_pyarray(py)
}

#[pyfunction]
fn compute_node_degree<'py>(
    py: Python<'py>,
    neighbors: PyReadonlyArray2<'py, usize>,
    weights: PyReadonlyArray2<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let neighbors = neighbors.as_array();
    let weights = weights.as_array();

    cnd(&neighbors, &weights).into_pyarray(py)
}

#[pyfunction]
fn center_values<'py>(
    py: Python<'py>,
    vals: PyReadonlyArray1<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
    var: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let vals = vals.as_array();
    let mu = mu.as_array();
    let var = var.as_array();

    cv(&vals.view(), &mu, &var).into_pyarray(py)
}

#[pyfunction]
fn local_cov_weights<'py>(
    _py: Python<'py>,
    vals: PyReadonlyArray1<'py, f64>,
    neighbors: PyReadonlyArray2<'py, usize>,
    weights: PyReadonlyArray2<'py, f64>,
) -> f64 {
    let vals = vals.as_array();
    let neighbors = neighbors.as_array();
    let weights = weights.as_array();

    lcw(&vals, &neighbors, &weights)
}

#[pyfunction]
fn compute_moments_weights<'py>(
    _py: Python<'py>,
    mu: PyReadonlyArray1<'py, f64>,
    x2: PyReadonlyArray1<'py, f64>,
    neighbors: PyReadonlyArray2<'py, usize>,
    weights: PyReadonlyArray2<'py, f64>,
) -> (f64, f64) {
    let mu = mu.as_array();
    let x2 = x2.as_array();
    let neighbors = neighbors.as_array();
    let weights = weights.as_array();

    cmw(&mu, &x2, &neighbors, &weights)
}

#[pyfunction]
fn compute_local_cov_max<'py>(
    _py: Python<'py>,
    node_degrees: PyReadonlyArray1<'py, f64>,
    row: PyReadonlyArray1<'py, f64>,
) -> f64 {
    let node_degrees = node_degrees.as_array();
    let row = row.as_array();

    clcm(&node_degrees, &row.view())
}

#[allow(clippy::type_complexity)]
#[pyfunction]
fn fit_none_model<'py>(
    py: Python<'py>,
    gene_counts: PyReadonlyArray1<'py, f64>,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
) {
    let gene_counts = gene_counts.as_array();

    let (mu, var, x2) = fnm(&gene_counts.view());

    (mu.to_pyarray(py), var.to_pyarray(py), x2.to_pyarray(py))
}

#[allow(clippy::type_complexity)]
#[pyfunction]
fn fit_bernoulli_model<'py>(
    py: Python<'py>,
    gene_counts: PyReadonlyArray1<'py, f64>,
    umi_counts: PyReadonlyArray1<'py, f64>,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
) {
    let gene_counts = gene_counts.as_array();
    let umi_counts = umi_counts.as_array();

    let (mu, var, x2) = fbernm(&gene_counts.view(), &umi_counts);

    (mu.to_pyarray(py), var.to_pyarray(py), x2.to_pyarray(py))
}

#[allow(clippy::type_complexity)]
#[pyfunction]
fn fit_danb_model<'py>(
    py: Python<'py>,
    gene_counts: PyReadonlyArray1<'py, f64>,
    umi_counts: PyReadonlyArray1<'py, f64>,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
) {
    let gene_counts = gene_counts.as_array();
    let umi_counts = umi_counts.as_array();

    let (mu, var, x2) = fdanbm(&gene_counts.view(), &umi_counts);

    (mu.to_pyarray(py), var.to_pyarray(py), x2.to_pyarray(py))
}

#[allow(clippy::type_complexity)]
#[pyfunction]
fn fit_normal_model<'py>(
    py: Python<'py>,
    gene_counts: PyReadonlyArray1<'py, f64>,
    umi_counts: PyReadonlyArray1<'py, f64>,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
) {
    let gene_counts = gene_counts.as_array();
    let umi_counts = umi_counts.as_array();

    let (mu, var, x2) = fnormalm(&gene_counts.view(), &umi_counts);

    (mu.to_pyarray(py), var.to_pyarray(py), x2.to_pyarray(py))
}

#[pyfunction]
fn compute_weights<'py>(
    py: Python<'py>,
    distances: PyReadonlyArray2<'py, f64>,
    n_factor: usize,
) -> Bound<'py, PyArray2<f64>> {
    let distances = distances.as_array();

    cw(&distances, n_factor).into_pyarray(py)
}

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // top level
    m.add_function(wrap_pyfunction!(compute_simvar, m)?)?;
    m.add_function(wrap_pyfunction!(compute_simvar_pairs_centered_cond, m)?)?;
    m.add_function(wrap_pyfunction!(compute_simvar_and_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(make_neighbors_and_weights, m)?)?;
    m.add_function(wrap_pyfunction!(make_weights_non_redundant, m)?)?;
    // testing
    m.add_function(wrap_pyfunction!(knn_from_latent, m)?)?;
    m.add_function(wrap_pyfunction!(knn_from_distances, m)?)?;
    m.add_function(wrap_pyfunction!(create_centered_counts, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_conditional_eg2, m)?)?;
    m.add_function(wrap_pyfunction!(
        compute_simvar_pairs_inner_centered_cond_sym,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(compute_local_cov_pairs_max, m)?)?;
    m.add_function(wrap_pyfunction!(compute_node_degree, m)?)?;
    m.add_function(wrap_pyfunction!(center_values, m)?)?;
    m.add_function(wrap_pyfunction!(local_cov_weights, m)?)?;
    m.add_function(wrap_pyfunction!(compute_moments_weights, m)?)?;
    m.add_function(wrap_pyfunction!(compute_local_cov_max, m)?)?;
    m.add_function(wrap_pyfunction!(fit_none_model, m)?)?;
    m.add_function(wrap_pyfunction!(fit_bernoulli_model, m)?)?;
    m.add_function(wrap_pyfunction!(fit_danb_model, m)?)?;
    m.add_function(wrap_pyfunction!(fit_normal_model, m)?)?;
    m.add_function(wrap_pyfunction!(compute_weights, m)?)?;
    Ok(())
}
