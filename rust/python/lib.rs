use pyo3::prelude::*;

mod example;

pub use example::Example;

fn map_err(error: String) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(error)
}

#[pyfunction]
fn equal_weights(count: usize) -> PyResult<Vec<f64>> {
    ::finance_portfolio::equal_weights(count).map_err(map_err)
}

#[pyfunction]
fn rank_weights(values: Vec<f64>, ascending: bool) -> PyResult<Vec<f64>> {
    ::finance_portfolio::rank_weights(values, ascending).map_err(map_err)
}

#[pyfunction]
fn signal_proportional_weights(values: Vec<f64>) -> PyResult<Vec<f64>> {
    ::finance_portfolio::signal_proportional_weights(values).map_err(map_err)
}

#[pyfunction]
fn target_volatility_weights(volatility: Vec<f64>) -> PyResult<Vec<f64>> {
    ::finance_portfolio::target_volatility_weights(volatility).map_err(map_err)
}

#[pyfunction]
fn minimum_variance_weights(covariance: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    ::finance_portfolio::minimum_variance_weights(covariance).map_err(map_err)
}

#[pyfunction]
fn mean_variance_weights(
    expected_returns: Vec<f64>,
    covariance: Vec<Vec<f64>>,
    risk_aversion: f64,
) -> PyResult<Vec<f64>> {
    ::finance_portfolio::mean_variance_weights(expected_returns, covariance, risk_aversion)
        .map_err(map_err)
}

#[pyfunction]
fn risk_parity_weights(
    covariance: Vec<Vec<f64>>,
    max_iter: usize,
    tolerance: f64,
) -> PyResult<Vec<f64>> {
    ::finance_portfolio::risk_parity_weights(covariance, max_iter, tolerance).map_err(map_err)
}

#[pyfunction]
fn hierarchical_risk_parity_weights(covariance: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    ::finance_portfolio::hierarchical_risk_parity_weights(covariance).map_err(map_err)
}

#[pyfunction]
fn portfolio_variance(weights: Vec<f64>, covariance: Vec<Vec<f64>>) -> PyResult<f64> {
    ::finance_portfolio::portfolio_variance(weights, covariance).map_err(map_err)
}

#[pyfunction]
fn portfolio_volatility(weights: Vec<f64>, covariance: Vec<Vec<f64>>) -> PyResult<f64> {
    ::finance_portfolio::portfolio_volatility(weights, covariance).map_err(map_err)
}

#[pyfunction]
fn risk_contribution(
    weights: Vec<f64>,
    covariance: Vec<Vec<f64>>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    ::finance_portfolio::risk_contribution(weights, covariance).map_err(map_err)
}

#[pyfunction]
fn tracking_error(active_weights: Vec<f64>, covariance: Vec<Vec<f64>>) -> PyResult<f64> {
    ::finance_portfolio::tracking_error(active_weights, covariance).map_err(map_err)
}

#[pyfunction]
fn active_share(active_weights: Vec<f64>) -> f64 {
    ::finance_portfolio::active_share(active_weights)
}

#[pyfunction]
fn brinson_attribution(
    portfolio_weights: Vec<f64>,
    benchmark_weights: Vec<f64>,
    portfolio_returns: Vec<f64>,
    benchmark_returns: Vec<f64>,
) -> PyResult<(f64, f64, f64, f64)> {
    ::finance_portfolio::brinson_attribution(
        portfolio_weights,
        benchmark_weights,
        portfolio_returns,
        benchmark_returns,
    )
    .map_err(map_err)
}

#[pyfunction]
fn factor_return_decomposition(
    exposures: Vec<f64>,
    factor_returns: Vec<f64>,
) -> PyResult<(Vec<f64>, f64)> {
    ::finance_portfolio::factor_return_decomposition(exposures, factor_returns).map_err(map_err)
}

#[pymodule]
fn finance_portfolio(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // Example
    m.add_class::<Example>().unwrap();
    m.add_function(wrap_pyfunction!(equal_weights, m)?)?;
    m.add_function(wrap_pyfunction!(rank_weights, m)?)?;
    m.add_function(wrap_pyfunction!(signal_proportional_weights, m)?)?;
    m.add_function(wrap_pyfunction!(target_volatility_weights, m)?)?;
    m.add_function(wrap_pyfunction!(minimum_variance_weights, m)?)?;
    m.add_function(wrap_pyfunction!(mean_variance_weights, m)?)?;
    m.add_function(wrap_pyfunction!(risk_parity_weights, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchical_risk_parity_weights, m)?)?;
    m.add_function(wrap_pyfunction!(portfolio_variance, m)?)?;
    m.add_function(wrap_pyfunction!(portfolio_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(risk_contribution, m)?)?;
    m.add_function(wrap_pyfunction!(tracking_error, m)?)?;
    m.add_function(wrap_pyfunction!(active_share, m)?)?;
    m.add_function(wrap_pyfunction!(brinson_attribution, m)?)?;
    m.add_function(wrap_pyfunction!(factor_return_decomposition, m)?)?;
    Ok(())
}
