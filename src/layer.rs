use crate::{
    functions::{ActivationFn, Input},
    neuron::Neuron,
};

/// The dimensions of a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Size {
    /// Input size of the layer.
    pub input: usize,
    /// Output size of the layer.
    pub output: usize,
}

/// A dense layer of deep learning, i.e. all layer input connected to all
/// previous layer output; all layer output connected to all next layer input.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Layer {
    neurons: Box<[Neuron]>,
}

impl Layer {
    /// Creates a new dense layer. Caller should certificate that neither
    /// `input_size` nor `output_size` are zero.
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut neurons = Vec::with_capacity(output_size);

        for _ in 0 .. output_size {
            neurons.push(Neuron::new(input_size));
        }

        Self { neurons: neurons.into() }
    }

    /// Returns the slice of neurons this layer owns. Guaranteed to have at
    /// least one element.
    pub fn neurons(&self) -> &[Neuron] {
        &self.neurons
    }

    /// Input size of the layer (weights per neuron).
    pub fn input_size(&self) -> usize {
        self.neurons[0].input_size()
    }

    /// Output size of the layer (neuron count).
    pub fn output_size(&self) -> usize {
        self.neurons.len()
    }

    /// Size of input and output of the layer.
    pub fn size(&self) -> Size {
        Size { input: self.input_size(), output: self.output_size() }
    }

    /// Computes the activation values (and not the derivatives) of the neurons
    /// and saves the result. Callers should certify that `input` has the same
    /// size as this layer was constructed with.
    pub fn compute_activations<F, I>(&mut self, activation_fn: &F, input: &[I])
    where
        F: ActivationFn,
        I: Input,
    {
        for neuron in &mut *self.neurons {
            neuron.compute_activation(activation_fn, input);
        }
    }

    /// Computes the all the derivatives of the neurons (activation derivative,
    /// weights derivatives and bias derivatives), for a layer that is not the
    /// last one. `compute_activations` should be called first. Callers should
    /// certify that `input` has the same size as this layer was constructed
    /// with.
    pub fn compute_derivs<F, I>(
        &mut self,
        activation_fn: &F,
        input: &[I],
        next_layer: &Self,
    ) where
        F: ActivationFn,
        I: Input,
    {
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            neuron.compute_all_derivs(activation_fn, input, i, next_layer);
        }
    }

    /// Computes the all the derivatives of the neurons (activation derivative,
    /// weights derivatives and bias derivatives), for a layer that is the last
    /// one. `compute_activations` should be called first. Callers should
    /// certify that `input` has the same size as this layer was constructed
    /// with.
    pub fn compute_derivs_last<F, I>(
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

    /// Optimizes the weights and the bias of all neurons using the derivatives
    /// scaled by scale.
    pub fn optimize(&mut self, scale: f64) {
        for neuron in &mut *self.neurons {
            neuron.optimize(scale);
        }
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::{
        functions::{LogisticFn, LossFn, SquaredError},
        neuron,
    };

    pub fn layer1() -> Layer {
        let mut layer = Layer::new(2, 3);
        layer.neurons[0] = neuron::test::neuron1();
        layer.neurons[1] = neuron::test::neuron2();
        layer.neurons[2] = neuron::test::neuron3();
        layer
    }

    pub fn layer2() -> Layer {
        let mut layer = Layer::new(3, 2);
        layer.neurons[0] = neuron::test::neuron4();
        layer.neurons[1] = neuron::test::neuron5();
        layer
    }

    #[test]
    fn activation_val() {
        let mut prev_layer = layer1();
        let mut layer = layer2();

        prev_layer.compute_activations(&LogisticFn, &[1.0, 0.5]);
        layer.compute_activations(&LogisticFn, &prev_layer.neurons);

        assert_float_eq!(prev_layer.neurons[0].activation(), 0.6048, 1e-3);
        assert_float_eq!(prev_layer.neurons[1].activation(), 0.5, 1e-3);
        assert_float_eq!(prev_layer.neurons[2].activation(), 0.3952, 1e-3);
        assert_float_eq!(layer.neurons[0].activation(), 0.62435, 1e-3);
        assert_float_eq!(layer.neurons[1].activation(), 0.57444, 1e-3);
    }

    #[test]
    fn derivatives() {
        let mut prev_layer = layer1();
        let mut layer = layer2();

        prev_layer.compute_activations(&LogisticFn, &[1.0, 0.5]);
        layer.compute_activations(&LogisticFn, &prev_layer.neurons);
        layer.compute_derivs_last(
            &LogisticFn,
            &prev_layer.neurons,
            &[
                SquaredError.deriv(layer.neurons[0].activation(), 0.0),
                SquaredError.deriv(layer.neurons[1].activation(), 1.0),
            ],
        );
        prev_layer.compute_derivs(&LogisticFn, &[1.0, 0.5], &layer);

        assert_float_eq!(prev_layer.neurons[0].activation(), 0.6048, 1e-3);
        assert_float_eq!(prev_layer.neurons[1].activation(), 0.5, 1e-3);
        assert_float_eq!(prev_layer.neurons[2].activation(), 0.3952, 1e-3);
        assert_float_eq!(layer.neurons[0].activation(), 0.62435, 1e-3);
        assert_float_eq!(layer.neurons[1].activation(), 0.57444, 1e-3);

        assert_float_eq!(
            prev_layer.neurons[0].deriv_over_act_input(),
            0.23901,
            1e-3
        );
        assert_float_eq!(
            prev_layer.neurons[1].deriv_over_act_input(),
            0.25,
            1e-3
        );
        assert_float_eq!(
            prev_layer.neurons[2].deriv_over_act_input(),
            0.23901,
            1e-3
        );
        assert_float_eq!(layer.neurons[0].deriv_over_act_input(), 0.2345, 1e-3);
        assert_float_eq!(
            layer.neurons[1].deriv_over_act_input(),
            0.24446,
            1e-3
        );

        assert_float_eq!(layer.neurons[0].deriv_over_act_val(), 1.2487, 1e-3);
        assert_float_eq!(layer.neurons[1].deriv_over_act_val(), -0.85112, 1e-3);
        assert_float_eq!(
            prev_layer.neurons[0].deriv_over_act_val(),
            0.033902,
            1e-3
        );
        assert_float_eq!(
            prev_layer.neurons[1].deriv_over_act_val(),
            0.06703,
            1e-3
        );
        assert_float_eq!(
            prev_layer.neurons[2].deriv_over_act_val(),
            0.15103,
            1e-3
        );
    }
}
