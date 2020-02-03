use crate::{
    functions::{ActivationFn, Input, LossFn},
    layer::Layer,
};
use std::slice;

pub use crate::layer::Size as LayerSize;

/// Iterator over layer dimensions (input and output sizes).
#[derive(Debug)]
pub struct LayerSizes<'net> {
    inner: slice::Iter<'net, Layer>,
}

impl<'net> Iterator for LayerSizes<'net> {
    type Item = LayerSize;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(Layer::size)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    fn nth(&mut self, index: usize) -> Option<Self::Item> {
        self.inner.nth(index).map(Layer::size)
    }
}

/// A deep learning, neural network.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Neural<A, L>
where
    A: ActivationFn,
    L: LossFn,
{
    /// All the layers of this neural network.
    layers: Box<[Layer]>,
    /// Generic activation function.
    activation_fn: A,
    /// Generic error estimative function.
    loss_fn: L,
    /// Computed losses.
    losses: Box<[f64]>,
    /// Computed losses derivatives.
    losses_deriv: Box<[f64]>,
}

impl<A, L> Neural<A, L>
where
    A: ActivationFn,
    L: LossFn,
{
    /// Creates a new neural network, given the activation and loss functions.
    ///
    /// # Panics
    /// Panics if the `input_size` or any `layer_size` is zero.
    pub fn new(
        activation_fn: A,
        loss_fn: L,
        mut input_size: usize,
        hidden_layer_sizes: &[usize],
        output_size: usize,
    ) -> Self {
        if input_size == 0 {
            panic!("Input size cannot be zero")
        }

        let mut layers = Vec::with_capacity(hidden_layer_sizes.len() + 1);
        for &layer_size in hidden_layer_sizes {
            if layer_size == 0 {
                panic!("Layer size cannot be zero")
            }
            layers.push(Layer::new(input_size, layer_size));
            input_size = layer_size;
        }
        layers.push(Layer::new(input_size, output_size));

        Self {
            layers: layers.into(),
            activation_fn,
            loss_fn,
            losses: vec![0.0; output_size].into(),
            losses_deriv: vec![0.0; output_size].into(),
        }
    }

    /// Returns the expected input size of the network.
    pub fn input_size(&self) -> usize {
        self.layers[0].input_size()
    }

    /// Returns the expected output size of the network.
    pub fn output_size(&self) -> usize {
        self.layers.last().expect("One layer is the min").output_size()
    }

    /// Returns an iterator over the layer sizes.
    pub fn layer_sizes(&self) -> LayerSizes {
        LayerSizes { inner: self.layers.iter() }
    }

    /// Computes the loss of the neural network for the given input and given
    /// expected output, i.e. the error estimate.
    ///
    /// # Panics
    /// Panics if the input doesn't have the size specified at the creation of
    /// the network, or if the `expected` slice parameter doesn't have the size
    /// specified for the last layer.
    pub fn loss(&mut self, input: &[f64], expected: &[f64]) -> f64 {
        self.compute_losses(input, expected);
        self.loss_fn.join(&self.losses)
    }

    /// Predicts an output, given an input, based on previous training.
    ///
    /// # Panics
    /// Panics if the input doesn't have the size specified at the creation of
    /// the network, or if the output slice doesn't have the size specified
    /// for the last layer.
    pub fn predict(&mut self, input: &[f64], output: &mut [f64]) {
        let out_size =
            self.layers.last_mut().expect("One layer is the min").output_size();
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
            *out = neuron.activation();
        }
    }

    /// Performs an iteration of training, and returns the loss for the given
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
        self.loss_fn.join(&self.losses)
    }

    /// Computes activation values of each neuron, but not the derivatives.
    /// Saves the result.
    ///
    /// # Panics
    /// Panics if the input doesn't have the size specified at the creation of
    /// the network.
    fn compute_activations(&mut self, input: &[f64]) {
        if self.layers[0].input_size() != input.len() {
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

    /// Computes the loss of each output. Saves the result. Also computes the
    /// activation values.
    ///
    /// # Panics
    /// Panics if the input doesn't have the size specified at the creation of
    /// the network, or if the `expected` slice parameter doesn't have the size
    /// specified for the last layer.
    fn compute_losses(&mut self, input: &[f64], expected: &[f64]) {
        self.compute_activations(input);
        let last = self.layers.last_mut().expect("One layer is the min");

        let iter = self
            .losses
            .iter_mut()
            .zip(last.neurons().iter())
            .zip(expected.iter());
        for ((err, neuron), &expected) in iter {
            *err = self.loss_fn.call(neuron.activation(), expected);
        }
    }

    /// Computes the loss derivatives of each output. Saves the result. Also
    /// computes the activation values.
    ///
    /// # Panics
    /// Panics if the input doesn't have the size specified at the creation of
    /// the network, or if the `expected` slice parameter doesn't have the size
    /// specified for the last layer.
    fn compute_losses_deriv(&mut self, input: &[f64], expected: &[f64]) {
        self.compute_losses(input, expected);
        let last = self.layers.last_mut().expect("One layer is the min");

        let iter = self
            .losses_deriv
            .iter_mut()
            .zip(last.neurons().iter())
            .zip(expected.iter());
        for ((err, neuron), &expected) in iter {
            *err = self.loss_fn.deriv(neuron.activation(), expected);
        }
    }

    /// Computes all the derivatives of the neurons' data. Saves the result.
    /// Also computes the activation values and the losses.
    ///
    /// # Panics
    /// Panics if the input doesn't have the size specified at the creation of
    /// the network, or if the `expected` slice parameter doesn't have the size
    /// specified for the last layer.
    fn compute_derivs(&mut self, input: &[f64], expected: &[f64]) {
        self.compute_losses_deriv(input, expected);

        let (mut next, rest) =
            self.layers.split_last_mut().expect("One layer is the min");

        if let Some((mut curr, rest)) = rest.split_last_mut() {
            next.compute_derivs_last(
                &self.activation_fn,
                curr.neurons(),
                &self.losses_deriv,
            );

            for prev in rest.iter_mut().rev() {
                curr.compute_derivs(&self.activation_fn, prev.neurons(), next);
                next = curr;
                curr = prev;
            }

            curr.compute_derivs(&self.activation_fn, input, next);
        } else {
            next.compute_derivs_last(
                &self.activation_fn,
                input,
                &self.losses_deriv,
            );
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        functions::{LogisticFn, SquaredError},
        layer,
    };

    pub fn network() -> Neural<LogisticFn, SquaredError> {
        let mut network = Neural::new(LogisticFn, SquaredError, 2, &[3], 2);
        network.layers[0] = layer::test::layer1();
        network.layers[1] = layer::test::layer2();
        network
    }
}
