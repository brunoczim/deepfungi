/// Activation function. Used to compute a neuron's activation value.
pub trait ActivationFn {
    /// Calls the underived version of this function.
    fn call(&self, input: f64) -> f64;
    /// Calls the derivative of this function.
    fn deriv(&self, input: f64) -> f64;
}

/// Error function. Used to compute an error value.
pub trait ErrorFn {
    /// Calls the underived version of this function.
    fn call(&self, found: f64, desired: f64) -> f64;
    /// Calls the derivative of this function.
    fn deriv(&self, found: f64, desired: f64) -> f64;
    /// Computes the combination of the given error values.
    fn join(&self, values: &[f64]) -> f64;
}

/// Logistic function: 1 / (1 + e^(-x))
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

/// Square error cost function: (found - desired) ** 2
pub struct SquareError;

impl ErrorFn for SquareError {
    fn call(&self, found: f64, desired: f64) -> f64 {
        let diff = found - desired;
        diff * diff
    }

    fn deriv(&self, found: f64, desired: f64) -> f64 {
        (found - desired) * 2.0
    }

    fn join(&self, values: &[f64]) -> f64 {
        let mut acc = 0.0;

        for val in values {
            acc += val / values.len() as f64;
        }

        acc
    }
}
