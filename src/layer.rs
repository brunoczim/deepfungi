use crate::{functions::ActivationFn, input::Input, neuron::Neuron};

/// A dense layer of deep learning, i.e. all layer input connected to all
/// previous layer output; all layer output connected to all next layer input.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DenseLayer {
    neurons: Box<[Neuron]>,
}

impl DenseLayer {
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

    /// Computes the activation values (and not the derivatives) of the neurons
    /// and saves the result.
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
    /// last one. `compute_activations` should be called first.
    pub fn compute_derivs<F, I>(
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

    /// Computes the all the derivatives of the neurons (activation derivative,
    /// weights derivatives and bias derivatives), for a layer that is the last
    /// one. `compute_activations` should be called first.
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
}
