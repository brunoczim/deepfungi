use rand::Rng;

trait Input {
    fn as_float(&self) -> f64;
}

impl Input for f64 {
    fn as_float(&self) -> f64 {
        *self
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
struct Weight {
    val: f64,
    #[serde(skip_serializing)]
    #[serde(skip_deserializing)]
    input: f64,
    #[serde(skip_serializing)]
    #[serde(skip_deserializing)]
    derivative: f64,
}

impl Weight {
    fn random() -> Self {
        Self {
            val: rand::thread_rng().gen_range(0.0, 1.0),
            input: 0.0,
            derivative: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
struct Bias {
    val: f64,
    #[serde(skip_serializing)]
    #[serde(skip_deserializing)]
    derivative: f64,
}

impl Bias {
    fn random() -> Self {
        Self { val: rand::thread_rng().gen_range(0.0, 1.0), derivative: 0.0 }
    }
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
struct Activation {
    #[serde(skip_serializing)]
    #[serde(skip_deserializing)]
    val: f64,
    #[serde(skip_serializing)]
    #[serde(skip_deserializing)]
    deriv: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Neuron {
    weights: Vec<Weight>,
    bias: Bias,
    activation: Activation,
}

impl Neuron {
    fn new(weights: usize) -> Self {
        let mut this = Self {
            weights: Vec::with_capacity(weights),
            bias: Bias::random(),
            activation: Activation::default(),
        };

        for _ in 0 .. weights {
            this.weights.push(Weight::random())
        }

        this
    }

    fn compute_activation<F, I>(&mut self, activation_fn: &F, input: &[I])
    where
        F: ActivationFn,
        I: Input,
    {
        self.activation.val = 0.0;
        for (input, weight) in input.iter().zip(self.weights.iter_mut()) {
            weight.input = input.as_float();
            self.activation.val += weight.val * weight.input;
        }
        self.activation.val += self.bias.val;
    }
}

impl Input for Neuron {
    fn as_float(&self) -> f64 {
        self.activation.val
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
pub struct NeuralNetwork<A, E>
where
    A: ActivationFn,
    E: ErrorFn,
{
    layers: Vec<DenseLayer>,
    activation_fn: A,
    error_fn: E,
}

impl<A, E> NeuralNetwork<A, E>
where
    A: ActivationFn,
    E: ErrorFn,
{
    /// Creates a new neural network, given the activation and cost functions.
    ///
    /// # Panics
    /// Panics if the `input_size` or any `layer_size` is zero.
    pub fn new(
        activation_fn: A,
        error_fn: E,
        mut input_size: usize,
        layer_sizes: &[usize],
    ) -> Self {
        if input_size == 0 {
            panic!("Input size cannot be zero")
        }

        let mut this = Self {
            layers: Vec::with_capacity(layer_sizes.len()),
            activation_fn,
            error_fn,
        };

        if layer_sizes.len() == 0 {
            panic!("There must be at least one layer!")
        }
        for &layer_size in layer_sizes {
            if layer_size == 0 {
                panic!("Layer size cannot be zero")
            }
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

/// Error function. Used to compute an error value.
pub trait ErrorFn {
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

impl ErrorFn for SquareError {
    fn call(&self, found: f64, desired: f64) -> f64 {
        let diff = found - desired;
        diff * diff
    }

    fn deriv(&self, found: f64, desired: f64) -> f64 {
        (found - desired) * 2.0
    }
}
