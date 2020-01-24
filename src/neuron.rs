use crate::{functions::ActivationFn, input::Input, layer::DenseLayer};
use rand::Rng;

/// A weight on an input of a neuron.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
struct Weight {
    /// Value of the weight.
    val: f64,
    /// Derivative of the weight.
    #[serde(skip)]
    derivative: f64,
}

impl Weight {
    /// A weight with random value set.
    fn random() -> Self {
        Self { val: rand::thread_rng().gen_range(0.0, 1.0), derivative: 0.0 }
    }
}

/// A bias of a neuron.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
struct Bias {
    /// Value of the bias.
    val: f64,
    /// Derivative of the bias.
    #[serde(skip)]
    derivative: f64,
}

impl Bias {
    /// A bias with random value set.
    fn random() -> Self {
        Self { val: rand::thread_rng().gen_range(0.0, 1.0), derivative: 0.0 }
    }
}

/// A neuron's activation data.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
struct Activation {
    /// Computed input, i.e. `bias + sum of w[i] * a[i]` for weights w, inputs
    /// a.
    #[serde(skip)]
    input: f64,
    /// Value of the activation, i.e. input passed to activation function.
    #[serde(skip)]
    val: f64,
    /// Derivative of the activation over the input.
    #[serde(skip)]
    deriv_over_input: f64,
    /// Derivative of the activation over the activation value.
    #[serde(skip)]
    deriv_over_val: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Neuron {
    /// Weights of this neuron, each one for an input.
    weights: Box<[Weight]>,
    /// A single bias of this neuron.
    bias: Bias,
    /// Activation data of this neuron.
    activation: Activation,
}

impl Neuron {
    /// A new neuron with support for `weights_len` input size. Also the number
    /// of weights, as the name implies. Callers should certify that this number
    /// is not zero.
    pub fn new(weights_len: usize) -> Self {
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

    /// Computes activation values only, not the derivatives. Saves the result.
    pub fn compute_activation<F, I>(&mut self, activation_fn: &F, input: &[I])
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

    /// Computes the derivatives over the activation input `b + sum(w[i] *
    /// a[i])`, but not over the activation value. Saves the result.
    /// `compute_activation` should be called first.
    pub fn compute_deriv_over_act_input<F>(&mut self, activation_fn: &F)
    where
        F: ActivationFn,
    {
        self.activation.deriv_over_input =
            activation_fn.deriv(self.activation.input);
    }

    /// Computes the derivatives over the activation value, but not over the
    /// activation input. Saves the result. `compute_activation` should be
    /// called first.
    pub fn compute_deriv_over_act_val(
        &mut self,
        index: usize,
        next_layer: &DenseLayer,
    ) {
        self.activation.deriv_over_val = 0.0;
        for neuron in next_layer.neurons() {
            let weight = neuron.weights[index].val;

            let dval = neuron.activation.deriv_over_val;
            let dinp = neuron.activation.deriv_over_input;
            self.activation.deriv_over_val += weight * dval * dinp;
        }
    }

    /// Computes the derivative over the weights, for each weight. Saves the
    /// result. `compute_deriv_over_act_val` and  `compute_deriv_over_act_input`
    /// should be called first.
    pub fn compute_deriv_over_weight<I>(&mut self, input: &[I])
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

    /// Computes the derivative over the bias. Saves the
    /// result. `compute_deriv_over_act_val` and  `compute_deriv_over_act_input`
    /// should be called first.
    pub fn compute_deriv_over_bias(&mut self) {
        let dval = self.activation.deriv_over_val;
        let dinp = self.activation.deriv_over_input;
        self.bias.derivative = dval * dinp;
    }

    /// Computes all derivatives related to this neuron, for neurons not in the
    /// last layer. `compute_activation` should be called first.
    pub fn compute_all_derivs<F, I>(
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

    /// Computes all derivatives related to this neuron, for neurons in the last
    /// layer. `compute_activation` should be called first.
    pub fn compute_all_derivs_last<F, I>(
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

    /// Optimizes the weights and the bias using the derivatives scaled by
    /// scale.
    pub fn optimize(&mut self, scale: f64) {
        for weight in &mut *self.weights {
            weight.val -= weight.derivative * scale;
        }
        self.bias.val -= self.bias.derivative * scale;
    }
}

impl Input for Neuron {
    fn as_float(&self) -> f64 {
        self.activation.val
    }
}
