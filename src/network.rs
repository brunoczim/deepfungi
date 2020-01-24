use crate::{
    functions::{ActivationFn, LossFn},
    input::Input,
    layer::DenseLayer,
};

/// A deep learning, neural network.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NeuralNetwork<A, L>
where
    A: ActivationFn,
    L: LossFn,
{
    /// All the layers of this neural network.
    layers: Box<[DenseLayer]>,
    /// Generic activation function.
    activation_fn: A,
    /// Generic error estimative function.
    loss_fn: L,
    /// Computed errors.
    errors: Box<[f64]>,
}

impl<A, L> NeuralNetwork<A, L>
where
    A: ActivationFn,
    L: LossFn,
{
    /// Creates a new neural network, given the activation and cost functions.
    ///
    /// # Panics
    /// Panics if the `input_size` or any `layer_size` is zero.
    pub fn new(
        activation_fn: A,
        loss_fn: L,
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
            loss_fn,
            errors: vec![0.0; input_size].into(),
        }
    }

    /// Computes the error of the neural network, i.e. the cost.
    ///
    /// # Panics
    /// Panics if the input doesn't have the size specified at the creation of
    /// the network, or if the `expected` slice parameter doesn't have the size
    /// specified for the last layer.
    pub fn error(&mut self, input: &[f64], expected: &[f64]) -> f64 {
        self.compute_errors(input, expected);
        self.loss_fn.join(&self.errors)
    }

    /// Predicts an output, given an input, based on previous training.
    ///
    /// # Panics
    /// Panics if the input doesn't have the size specified at the creation of
    /// the network, or if the output slice doesn't have the size specified
    /// for the last layer.
    pub fn predict(&mut self, input: &[f64], output: &mut [f64]) {
        let out_size = self
            .layers
            .last_mut()
            .expect("One layer is the min")
            .neurons()
            .len();
        if out_size != output.len() {
            panic!(
                "Output size should be {}, but it is {}",
                out_size,
                output.len()
            );
        }

        self.compute_activations(input);

        let last = self.layers.last_mut().expect("One layer is the min");

        for (neuron, out) in last.neurons().iter().zip(output.iter_mut()) {
            *out = neuron.as_float();
        }
    }

    /// Performs an iteration of training, and returns the cost for the given
    /// input, before training.
    ///
    /// # Panics
    /// Panics if the input doesn't have the size specified at the creation of
    /// the network, or if the `expected` slice parameter doesn't have the size
    /// specified for the last layer.
    pub fn train(
        &mut self,
        input: &[f64],
        expected: &[f64],
        scale: f64,
    ) -> f64 {
        self.compute_derivs(input, expected);
        self.optimize(scale);
        self.loss_fn.join(&self.errors)
    }

    /// Computes activation values of each neuron, but not the derivatives.
    /// Saves the result.
    ///
    /// # Panics
    /// Panics if the input doesn't have the size specified at the creation of
    /// the network.
    fn compute_activations(&mut self, input: &[f64]) {
        if self.layers[0].neurons().len() != input.len() {
            panic!(
                "Input size should be {}, but is {}",
                self.layers[0].neurons().len(),
                input.len()
            )
        }
        self.layers[0].compute_activations(&self.activation_fn, input);

        let (init, rest) = self.layers.split_at_mut(1);
        let mut prev_layer = &mut init[0];
        for layer in rest {
            layer
                .compute_activations(&self.activation_fn, prev_layer.neurons());
            prev_layer = layer;
        }
    }

    /// Computes the error of each output. Saves the result.
    ///
    /// # Panics
    /// Panics if the input doesn't have the size specified at the creation of
    /// the network, or if the `expected` slice parameter doesn't have the size
    /// specified for the last layer.
    fn compute_errors(&mut self, input: &[f64], expected: &[f64]) {
        self.compute_activations(input);
        let last = self.layers.last_mut().expect("One layer is the min");

        let iter = self
            .errors
            .iter_mut()
            .zip(last.neurons().iter())
            .zip(expected.iter());
        for ((err, neuron), &expected) in iter {
            *err = self.loss_fn.call(neuron.as_float(), expected);
        }
    }

    /// Computes all the derivatives of the neurons' data. Saves the result.
    ///
    /// # Panics
    /// Panics if the input doesn't have the size specified at the creation of
    /// the network, or if the `expected` slice parameter doesn't have the size
    /// specified for the last layer.
    fn compute_derivs(&mut self, input: &[f64], expected: &[f64]) {
        self.compute_errors(input, expected);

        let (mut next, rest) =
            self.layers.split_last_mut().expect("One layer is the min");

        if let Some((mut curr, rest)) = rest.split_last_mut() {
            next.compute_derivs_last(
                &self.activation_fn,
                curr.neurons(),
                &self.errors,
            );

            for prev in rest.iter_mut().rev() {
                curr.compute_derivs(&self.activation_fn, prev.neurons(), next);
                next = curr;
                curr = prev;
            }

            curr.compute_derivs(&self.activation_fn, input, next);
        } else {
            next.compute_derivs_last(&self.activation_fn, input, &self.errors);
        }
    }

    /// Optimizes the weights and the bias of all neurons using the derivatives
    /// scaled by scale.
    fn optimize(&mut self, scale: f64) {
        for layer in &mut *self.layers {
            layer.optimize(scale);
        }
    }
}
