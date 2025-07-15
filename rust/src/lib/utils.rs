use ndarray::Array1;

/// Square the values in the provided vector.
///
/// # Arguments
///
/// * `vec` - A vector of values to be squared.
///
/// # Returns
///
/// * `squared_vec` - A vector of squared values.
pub fn square_iterable<'a, I: IntoIterator<Item = &'a f64>>(vec: I) -> Vec<f64> {
    vec.into_iter().map(|x| x * x).collect()
}

/// Reorder the provided values according to the provided order.
///
/// # Arguments
///
/// * `order` - A vector of indices to reorder the values by.
/// * `vals` - A vector of values to be reordered.
///
/// # Returns
///
/// * `reordered_vals` - A vector of reordered values.
pub fn _reorder<T>(order: &[usize], vals: Vec<T>) -> Array1<T>
where
    T: Clone,
{
    order.iter().map(|x| vals[*x].clone()).collect()
}
