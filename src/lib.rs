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
    derivative: f64,
}

impl Weight {
    fn random() -> Self {
        Self { val: rand::thread_rng().gen_range(0.0, 1.0), derivative: 0.0 }
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
    input: f64,
    #[serde(skip_serializing)]
    #[serde(skip_deserializing)]
    val: f64,
    #[serde(skip_serializing)]
    #[serde(skip_deserializing)]
    deriv_over_input: f64,
    #[serde(skip_serializing)]
    #[serde(skip_deserializing)]
    deriv_over_val: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Neuron {
    weights: Box<[Weight]>,
    bias: Bias,
    activation: Activation,
}

impl Neuron {
    fn new(weights_len: usize) -> Self {
        let mut weights = Vec::with_capacity(weights_len);

        for _ in 0 .. weights_len {
            weights.push(Weight::random())
        }

        Self {
            weights: weights.into(),
            bias: Bias::random(),
            activation: Activation::default(),
        }
    }

    fn compute_activation<F, I>(&mut self, activation_fn: &F, input: &[I])
    where
        F: ActivationFn,
        I: Input,
    {
        self.activation.input = 0.0;
        for (input, weight) in input.iter().zip(self.weights.iter_mut()) {
            self.activation.input += weight.val * input.as_float();
        }
        self.activation.input += self.bias.val;
        self.activation.val = activation_fn.call(self.activation.input);
    }

    fn compute_deriv_over_act_input<F>(&mut self, activation_fn: &F)
    where
        F: ActivationFn,
    {
        self.activation.deriv_over_input =
            activation_fn.deriv(self.activation.input);
    }

    fn compute_deriv_over_act_val(
        &mut self,
        index: usize,
        next_layer: &DenseLayer,
    ) {
        self.activation.deriv_over_val = 0.0;
        for neuron in &*next_layer.neurons {
            let weight = neuron.weights[index].val;

            let dval = neuron.activation.deriv_over_val;
            let dinp = neuron.activation.deriv_over_input;
            self.activation.deriv_over_val += weight * dval * dinp;
        }
    }

    fn compute_deriv_over_weight<I>(&mut self, input: &[I])
    where
        I: Input,
    {
        let iter = self.weights.iter_mut().zip(input.iter());
        for (weight, input) in iter {
            let dval = self.activation.deriv_over_val;
            let dinp = self.activation.deriv_over_input;
            weight.derivative = dval * dinp * input.as_float();
        }
    }

    fn compute_deriv_over_bias(&mut self) {
        let dval = self.activation.deriv_over_val;
        let dinp = self.activation.deriv_over_input;
        self.bias.derivative = dval * dinp;
    }

    fn compute_all_derivs<F, I>(
        &mut self,
        activation_fn: &F,
        input: &[I],
        index: usize,
        next_layer: &DenseLayer,
    ) where
        F: ActivationFn,
        I: Input,
    {
        self.compute_deriv_over_act_input(activation_fn);
        self.compute_deriv_over_act_val(index, next_layer);
        self.compute_deriv_over_weight(input);
        self.compute_deriv_over_bias();
    }

    fn compute_all_derivs_last<F, I>(
        &mut self,
        activation_fn: &F,
        input: &[I],
        deriv_over_act_val: f64,
    ) where
        F: ActivationFn,
        I: Input,
    {
        self.compute_deriv_over_act_input(activation_fn);
        self.activation.deriv_over_val = deriv_over_act_val;
        self.compute_deriv_over_weight(input);
        self.compute_deriv_over_bias();
    }
}

impl Input for Neuron {
    fn as_float(&self) -> f64 {
        self.activation.val
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DenseLayer {
    neurons: Box<[Neuron]>,
}

impl DenseLayer {
    fn new(input_size: usize, output_size: usize) -> Self {
        let mut neurons = Vec::with_capacity(output_size);

        for _ in 0 .. output_size {
            neurons.push(Neuron::new(input_size));
        }

        Self { neurons: neurons.into() }
    }

    fn compute_activations<F, I>(&mut self, activation_fn: &F, input: &[I])
    where
        F: ActivationFn,
        I: Input,
    {
        for neuron in &mut *self.neurons {
            neuron.compute_activation(activation_fn, input);
        }
    }

    fn compute_derivs<F, I>(
        &mut self,
        activation_fn: &F,
        input: &[I],
        next_layer: &DenseLayer,
    ) where
        F: ActivationFn,
        I: Input,
    {
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            neuron.compute_all_derivs(activation_fn, input, i, next_layer);
        }
    }

    fn compute_derivs_last<F, I>(
        &mut self,
        activation_fn: &F,
        input: &[I],
        deriv_over_act_val: &[f64],
    ) where
        F: ActivationFn,
        I: Input,
    {
        let iter = self.neurons.iter_mut().zip(deriv_over_act_val.iter());
        for (neuron, &deriv_over_act_val) in iter {
            neuron.compute_all_derivs_last(
                activation_fn,
                input,
                deriv_over_act_val,
            );
        }
    }
}

/// A deep learning, neural network.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NeuralNetwork<A, E>
where
    A: ActivationFn,
    E: ErrorFn,
{
    layers: Box<[DenseLayer]>,
    activation_fn: A,
    error_fn: E,
    errors: Box<[f64]>,
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

        let mut layers = Vec::with_capacity(layer_sizes.len());
        if layer_sizes.len() == 0 {
            panic!("There must be at least one layer!")
        }
        for &layer_size in layer_sizes {
            if layer_size == 0 {
                panic!("Layer size cannot be zero")
            }
            layers.push(DenseLayer::new(input_size, layer_size));
            input_size = layer_size;
        }

        Self {
            layers: layers.into(),
            activation_fn,
            error_fn,
            errors: vec![0.0; input_size].into(),
        }
    }

    /// Computes the error of the neural network, i.e. the cost.
    ///
    /// # Panics
    /// Panics if the input doesn't have the size specified at the creation of
    /// the network, or if the expected slice doesn't have the size specified
    /// for the last layer.
    pub fn error(&mut self, input: &[f64], expected: &[f64]) -> f64 {
        self.compute_activations(input);
        self.compute_errors(expected);
        self.error_fn.join(&self.errors)
    }

    /// Predicts an output, given an input, based on previous training.
    ///
    /// # Panics
    /// Panics if the input doesn't have the size specified at the creation of
    /// the network, or if the output slice doesn't have the size specified
    /// for the last layer.
    pub fn predict(&mut self, input: &[f64], output: &mut [f64]) {
        let out_size =
            self.layers.last_mut().expect("One layer is the min").neurons.len();
        if out_size != output.len() {
            panic!(
                "Output size should be {}, but it is {}",
                out_size,
                output.len()
            );
        }

        self.compute_activations(input);

        let last = self.layers.last_mut().expect("One layer is the min");

        for (neuron, out) in last.neurons.iter().zip(output.iter_mut()) {
            *out = neuron.activation.val;
        }
    }

    /// Performs an iteration of training, and returns the cost for the given
    /// input, before training.
    ///
    /// # Panics
    /// Panics if the input doesn't have the size specified at the creation of
    /// the network, or if the expected slice doesn't have the size specified
    /// for the last layer.
    pub fn train(&mut self, input: &[f64], expected: &[f64]) -> f64 {
        self.compute_derivs(input, expected);
        self.error_fn.join(&self.errors)
    }

    fn compute_errors(&mut self, expected: &[f64]) {
        let last = self.layers.last_mut().expect("One layer is the min");

        let iter = self
            .errors
            .iter_mut()
            .zip(last.neurons.iter())
            .zip(expected.iter());
        for ((err, neuron), &expected) in iter {
            *err = self.error_fn.call(neuron.activation.val, expected);
        }
    }

    fn compute_activations(&mut self, input: &[f64]) {
        if self.layers[0].neurons.len() != input.len() {
            panic!(
                "Input size should be {}, but is {}",
                self.layers[0].neurons.len(),
                input.len()
            )
        }
        self.layers[0].compute_activations(&self.activation_fn, input);

        let (init, rest) = self.layers.split_at_mut(1);
        let mut prev_layer = &mut init[0];
        for layer in rest {
            layer.compute_activations(&self.activation_fn, &prev_layer.neurons);
            prev_layer = layer;
        }
    }

    fn compute_derivs(&mut self, input: &[f64], expected: &[f64]) {
        self.compute_activations(input);
        self.compute_errors(expected);

        let (mut next, rest) =
            self.layers.split_last_mut().expect("One layer is the min");

        if let Some((mut curr, rest)) = rest.split_last_mut() {
            next.compute_derivs_last(
                &self.activation_fn,
                &curr.neurons,
                &self.errors,
            );

            for prev in rest.iter_mut().rev() {
                curr.compute_derivs(&self.activation_fn, &prev.neurons, next);
                next = curr;
                curr = prev;
            }

            curr.compute_derivs(&self.activation_fn, input, next);
        } else {
            next.compute_derivs_last(&self.activation_fn, input, &self.errors);
        }
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
