use rand::Rng;

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
struct WithDeriv {
    val: f64,
    derivative: f64,
}

impl WithDeriv {
    fn random() -> Self {
        Self { val: rand::thread_rng().gen_range(0.0, 1.0), derivative: 0.0 }
    }
}

impl Default for WithDeriv {
    fn default() -> Self {
        Self::random()
    }
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
struct Activation {
    input: f64,
    val: f64,
    deriv: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Neuron {
    weights: Vec<WithDeriv>,
    bias: WithDeriv,
    activation: Activation,
}

impl Neuron {
    fn new(weights: usize) -> Self {
        let mut this = Self {
            weights: Vec::with_capacity(weights),
            bias: WithDeriv::default(),
            activation: Activation::default(),
        };

        for _ in 0 .. weights {
            this.weights.push(WithDeriv::default())
        }

        this
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DenseLayer {
    neurons: Vec<Neuron>,
}

impl DenseLayer {
    fn new(input_size: usize, output_size: usize) -> Self {
        let mut this = Self { neurons: Vec::with_capacity(output_size) };

        for _ in 0 .. output_size {
            this.neurons.push(Neuron::new(input_size));
        }

        this
    }
}

/// A deep learning, neural network.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NeuralNetwork<A, C>
where
    A: ActivationFn,
    C: CostFn,
{
    layers: Vec<DenseLayer>,
    activation_fn: A,
    cost_fn: C,
}

impl<A, C> NeuralNetwork<A, C>
where
    A: ActivationFn,
    C: CostFn,
{
    /// Creates a new neural network, given the activation and cost functions.
    ///
    /// # Panics
    /// Panics if the `input_size` or any `layer_size` is zero.
    pub fn new(
        activation_fn: A,
        cost_fn: C,
        mut input_size: usize,
        layer_sizes: &[usize],
    ) -> Self {
        assert_ne!(input_size, 0);

        let mut this = Self {
            layers: Vec::with_capacity(layer_sizes.len()),
            activation_fn,
            cost_fn,
        };

        for &layer_size in layer_sizes {
            assert_ne!(layer_size, 0);
            this.layers.push(DenseLayer::new(input_size, layer_size));
            input_size = layer_size;
        }

        this
    }
}

/// Activation function. Used to compute a neuron's activation value.
pub trait ActivationFn {
    /// Calls the underived version of this function.
    fn call(&self, input: f64) -> f64;
    /// Calls the derivative of this function.
    fn deriv(&self, input: f64) -> f64;
}

/// Cost function. Used to compute an error value.
pub trait CostFn {
    /// Calls the underived version of this function.
    fn call(&self, found: f64, desired: f64) -> f64;
    /// Calls the derivative of this function.
    fn deriv(&self, found: f64, desired: f64) -> f64;
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

impl CostFn for SquareError {
    fn call(&self, found: f64, desired: f64) -> f64 {
        let diff = found - desired;
        diff * diff
    }

    fn deriv(&self, found: f64, desired: f64) -> f64 {
        (found - desired) * 2.0
    }
}
