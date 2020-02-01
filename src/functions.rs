/// Activation function. Used to compute a neuron's activation value.
pub trait ActivationFn {
    /// Calls the underived version of this function.
    fn call(&self, input: f64) -> f64;
    /// Calls the derivative of this function.
    fn deriv(&self, input: f64) -> f64;
}

/// Loss function. Used to compute a loss value (the error estimate).
pub trait LossFn {
    /// Calls the underived version of this function.
    fn call(&self, found: f64, desired: f64) -> f64;
    /// Calls the derivative of this function.
    fn deriv(&self, found: f64, desired: f64) -> f64;
    /// Computes the combination of the given error values.
    fn join(&self, values: &[f64]) -> f64;
}

/// Logistic function: `1 / (1 + e ^ (-x))`
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct LogisticFn;

impl ActivationFn for LogisticFn {
    fn call(&self, input: f64) -> f64 {
        1.0 / (1.0 + (-input).exp())
    }

    fn deriv(&self, input: f64) -> f64 {
        let raised = input.exp();
        let base = 1.0 + raised;
        raised / (base * base)
    }
}

/// Squared error cost function: `(found - desired) ^ 2`
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct SquaredError;

impl LossFn for SquaredError {
    fn call(&self, found: f64, desired: f64) -> f64 {
        let diff = found - desired;
        diff * diff
    }

    fn deriv(&self, found: f64, desired: f64) -> f64 {
        (found - desired) * 2.0
    }

    fn join(&self, values: &[f64]) -> f64 {
        let mut acc = 0.0f64;

        for &val in values {
            acc = acc.max(val);
        }

        acc
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn logistic_fn() {
        assert!(LogisticFn.call(0.0) - 0.5 < 1e-20);
        assert!(LogisticFn.call(0.425504) - 0.6048 < 1e-3);
        assert!(LogisticFn.call(-0.425504) - 0.3952 < 1e-3);

        let val = LogisticFn.call(10.0);
        assert!(val > 0.0 && val < 1.0);

        let val = LogisticFn.call(-10.0);
        assert!(val > 0.0 && val < 1.0);
    }

    #[test]
    fn logistic_deriv() {
        assert!(LogisticFn.deriv(0.0) - 0.25 < 1e-20);

        assert!(LogisticFn.deriv(0.425504) - 0.23901 < 1e-3);
        assert!(LogisticFn.deriv(-0.425504) - 0.23901 < 1e-3);

        let val = LogisticFn.deriv(10.0);
        assert!(val > 0.0 && val <= 0.25);

        let val = LogisticFn.deriv(-10.0);
        assert!(val > 0.0 && val <= 0.25);
    }

    #[test]
    fn squared_error() {
        assert!(SquaredError.call(2.0, 5.0) - 9.0 < 1e-20);
        assert!(SquaredError.call(5.0, 2.0) - 9.0 < 1e-20);
        assert!(SquaredError.call(1.0, 0.8) - 0.4 < 1e-20);
    }

    #[test]
    fn squared_error_deriv() {
        assert!(SquaredError.deriv(2.0, 5.0) + 6.0 < 1e-20);
        assert!(SquaredError.deriv(5.0, 2.0) - 6.0 < 1e-20);
        assert!(SquaredError.deriv(1.0, 0.8) - 0.4 < 1e-20);
    }
}
