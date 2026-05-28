#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Example {
    pub stuff: String,
}

impl Example {
    pub fn new(value: String) -> Self {
        Example { stuff: value }
    }
}

pub type Matrix = Vec<Vec<f64>>;
pub type RiskContributionParts = (Vec<f64>, Vec<f64>, Vec<f64>);

fn validate_matrix(matrix: &Matrix) -> Result<(usize, usize), String> {
    if matrix.is_empty() {
        return Err("matrix must not be empty".to_string());
    }
    let cols = matrix[0].len();
    if cols == 0 {
        return Err("matrix rows must not be empty".to_string());
    }
    if matrix.iter().any(|row| row.len() != cols) {
        return Err("matrix rows must have equal length".to_string());
    }
    Ok((matrix.len(), cols))
}

fn validate_square(matrix: &Matrix) -> Result<usize, String> {
    let (rows, cols) = validate_matrix(matrix)?;
    if rows != cols {
        return Err("matrix must be square".to_string());
    }
    Ok(rows)
}

fn normalize(mut weights: Vec<f64>, gross: bool) -> Result<Vec<f64>, String> {
    let denominator = if gross {
        weights.iter().map(|value| value.abs()).sum::<f64>()
    } else {
        weights.iter().sum::<f64>()
    };
    if denominator == 0.0 {
        return Err("weights cannot be normalized when the denominator is zero".to_string());
    }
    for weight in &mut weights {
        *weight /= denominator;
    }
    Ok(weights)
}

pub fn equal_weights(count: usize) -> Result<Vec<f64>, String> {
    if count == 0 {
        return Err("assets must not be empty".to_string());
    }
    Ok(vec![1.0 / count as f64; count])
}

pub fn rank_weights(values: Vec<f64>, ascending: bool) -> Result<Vec<f64>, String> {
    if values.is_empty() {
        return Err("signals must not be empty".to_string());
    }
    let mut ranked: Vec<(usize, f64)> = values.into_iter().enumerate().collect();
    ranked.sort_by(|left, right| {
        let ordering = left
            .1
            .partial_cmp(&right.1)
            .unwrap_or(std::cmp::Ordering::Equal);
        if ascending {
            ordering
        } else {
            ordering.reverse()
        }
    });
    let mut raw = vec![0.0; ranked.len()];
    for (rank, (idx, _value)) in ranked.into_iter().enumerate() {
        raw[idx] = rank as f64 + 1.0;
    }
    normalize(raw, false)
}

pub fn signal_proportional_weights(values: Vec<f64>) -> Result<Vec<f64>, String> {
    if values.is_empty() {
        return Err("signals must not be empty".to_string());
    }
    normalize(values, true)
}

pub fn target_volatility_weights(volatility: Vec<f64>) -> Result<Vec<f64>, String> {
    if volatility.is_empty() {
        return Err("volatility must not be empty".to_string());
    }
    if volatility.iter().any(|value| *value <= 0.0) {
        return Err("all volatility values must be positive".to_string());
    }
    normalize(
        volatility.into_iter().map(|value| 1.0 / value).collect(),
        false,
    )
}

fn mat_vec(matrix: &Matrix, vector: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(vector)
                .map(|(left, right)| left * right)
                .sum()
        })
        .collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(l, r)| l * r).sum()
}

fn invert_matrix(matrix: &Matrix) -> Option<Matrix> {
    let n = matrix.len();
    let mut augmented = vec![vec![0.0; n * 2]; n];
    for row in 0..n {
        for col in 0..n {
            augmented[row][col] = matrix[row][col];
        }
        augmented[row][n + row] = 1.0;
    }
    for col in 0..n {
        let mut pivot = col;
        for row in (col + 1)..n {
            if augmented[row][col].abs() > augmented[pivot][col].abs() {
                pivot = row;
            }
        }
        if augmented[pivot][col].abs() < 1e-14 {
            return None;
        }
        augmented.swap(col, pivot);
        let divisor = augmented[col][col];
        for value in &mut augmented[col] {
            *value /= divisor;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = augmented[row][col];
            let pivot_row = augmented[col].clone();
            for (target, source) in augmented[row].iter_mut().zip(pivot_row) {
                *target -= factor * source;
            }
        }
    }
    Some(augmented.into_iter().map(|row| row[n..].to_vec()).collect())
}

fn invert_with_ridge(covariance: &Matrix) -> Result<Matrix, String> {
    if let Some(inverse) = invert_matrix(covariance) {
        return Ok(inverse);
    }
    let mut regularized = covariance.clone();
    for (idx, row) in regularized.iter_mut().enumerate() {
        row[idx] += 1e-12;
    }
    invert_matrix(&regularized).ok_or_else(|| "covariance matrix is singular".to_string())
}

pub fn minimum_variance_weights(covariance: Matrix) -> Result<Vec<f64>, String> {
    let size = validate_square(&covariance)?;
    let inverse = invert_with_ridge(&covariance)?;
    let ones = vec![1.0; size];
    let inv_ones = mat_vec(&inverse, &ones);
    normalize(inv_ones, false)
}

pub fn mean_variance_weights(
    expected_returns: Vec<f64>,
    covariance: Matrix,
    risk_aversion: f64,
) -> Result<Vec<f64>, String> {
    let size = validate_square(&covariance)?;
    if expected_returns.len() != size {
        return Err("expected_returns length must match covariance dimensions".to_string());
    }
    let inverse = invert_with_ridge(&covariance)?;
    let scale = risk_aversion.max(1e-12);
    let raw: Vec<f64> = mat_vec(&inverse, &expected_returns)
        .into_iter()
        .map(|value| value / scale)
        .collect();
    if raw.iter().sum::<f64>() == 0.0 {
        return equal_weights(size);
    }
    normalize(raw, false)
}

pub fn risk_parity_weights(
    covariance: Matrix,
    max_iter: usize,
    tolerance: f64,
) -> Result<Vec<f64>, String> {
    let size = validate_square(&covariance)?;
    let mut weights: Vec<f64> = (0..size)
        .map(|idx| 1.0 / covariance[idx][idx].max(1e-18).sqrt())
        .collect();
    weights = normalize(weights, false)?;
    let target = 1.0 / size as f64;
    for _ in 0..max_iter {
        let cov_weights = mat_vec(&covariance, &weights);
        let variance = dot(&weights, &cov_weights);
        let vol = variance.max(0.0).sqrt();
        if vol == 0.0 {
            break;
        }
        let mut max_error: f64 = 0.0;
        for idx in 0..size {
            let pct = weights[idx] * cov_weights[idx] / variance;
            max_error = max_error.max((pct - target).abs());
            if pct > 0.0 {
                weights[idx] *= (target / pct).sqrt();
            }
            weights[idx] = weights[idx].max(1e-12);
        }
        weights = normalize(weights, false)?;
        if max_error < tolerance {
            break;
        }
    }
    Ok(weights)
}

pub fn hierarchical_risk_parity_weights(covariance: Matrix) -> Result<Vec<f64>, String> {
    let size = validate_square(&covariance)?;
    let raw: Vec<f64> = (0..size)
        .map(|idx| 1.0 / covariance[idx][idx].max(1e-18))
        .collect();
    normalize(raw, false)
}

pub fn portfolio_variance(weights: Vec<f64>, covariance: Matrix) -> Result<f64, String> {
    let size = validate_square(&covariance)?;
    if weights.len() != size {
        return Err("covariance dimensions must match weights".to_string());
    }
    Ok(dot(&weights, &mat_vec(&covariance, &weights)))
}

pub fn portfolio_volatility(weights: Vec<f64>, covariance: Matrix) -> Result<f64, String> {
    Ok(portfolio_variance(weights, covariance)?.max(0.0).sqrt())
}

pub fn risk_contribution(
    weights: Vec<f64>,
    covariance: Matrix,
) -> Result<RiskContributionParts, String> {
    let size = validate_square(&covariance)?;
    if weights.len() != size {
        return Err("covariance dimensions must match weights".to_string());
    }
    let cov_weights = mat_vec(&covariance, &weights);
    let vol = dot(&weights, &cov_weights).max(0.0).sqrt();
    let marginal: Vec<f64> = if vol == 0.0 {
        vec![0.0; size]
    } else {
        cov_weights.into_iter().map(|value| value / vol).collect()
    };
    let component: Vec<f64> = weights
        .iter()
        .zip(&marginal)
        .map(|(weight, marg)| weight * marg)
        .collect();
    let percentage: Vec<f64> = if vol == 0.0 {
        vec![0.0; size]
    } else {
        component.iter().map(|value| value / vol).collect()
    };
    Ok((marginal, component, percentage))
}

pub fn tracking_error(active_weights: Vec<f64>, covariance: Matrix) -> Result<f64, String> {
    portfolio_volatility(active_weights, covariance)
}

pub fn active_share(active_weights: Vec<f64>) -> f64 {
    0.5 * active_weights.iter().map(|value| value.abs()).sum::<f64>()
}

pub fn brinson_attribution(
    portfolio_weights: Vec<f64>,
    benchmark_weights: Vec<f64>,
    portfolio_returns: Vec<f64>,
    benchmark_returns: Vec<f64>,
) -> Result<(f64, f64, f64, f64), String> {
    let size = portfolio_weights.len();
    if benchmark_weights.len() != size
        || portfolio_returns.len() != size
        || benchmark_returns.len() != size
    {
        return Err("all input vectors must have the same length".to_string());
    }
    let benchmark_total = benchmark_weights
        .iter()
        .zip(&benchmark_returns)
        .map(|(weight, ret)| weight * ret)
        .sum::<f64>();
    let mut allocation = 0.0;
    let mut selection = 0.0;
    let mut interaction = 0.0;
    for idx in 0..size {
        allocation += (portfolio_weights[idx] - benchmark_weights[idx])
            * (benchmark_returns[idx] - benchmark_total);
        selection += benchmark_weights[idx] * (portfolio_returns[idx] - benchmark_returns[idx]);
        interaction += (portfolio_weights[idx] - benchmark_weights[idx])
            * (portfolio_returns[idx] - benchmark_returns[idx]);
    }
    Ok((
        allocation,
        selection,
        interaction,
        allocation + selection + interaction,
    ))
}

pub fn factor_return_decomposition(
    exposures: Vec<f64>,
    factor_returns: Vec<f64>,
) -> Result<(Vec<f64>, f64), String> {
    if exposures.len() != factor_returns.len() {
        return Err("exposures and factor_returns must have the same length".to_string());
    }
    let components: Vec<f64> = exposures
        .iter()
        .zip(&factor_returns)
        .map(|(exposure, ret)| exposure * ret)
        .collect();
    let total = components.iter().sum();
    Ok((components, total))
}

/**********************************/
#[cfg(test)]
mod example_tests {
    use super::*;

    #[test]
    fn test_new() {
        let e = Example::new(String::from("test"));
        assert_eq!(e.stuff, String::from("test"));
    }

    #[test]
    fn test_clone_and_eq() {
        let e = Example::new(String::from("test"));
        assert_eq!(e, e.clone());
    }

    #[test]
    fn test_debug() {
        let e = Example::new(String::from("test"));
        assert_eq!(format!("{e:?}"), "Example { stuff: \"test\" }");
    }
}
