use crate::{
    functions::{ActivationFn, Input},
    layer::Layer,
};
use rand::Rng;

/// A weight on an input of a neuron.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Weight {
    /// Value of the weight.
    pub val: f64,
    /// Derivative of the weight.
    #[serde(skip)]
    pub derivative: f64,
}

impl Weight {
    /// A weight with random value set.
    fn random() -> Self {
        Self { val: rand::thread_rng().gen_range(0.0, 1.0), derivative: 0.0 }
    }
}

/// A bias of a neuron.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Bias {
    /// Value of the bias.
    pub val: f64,
    /// Derivative of the bias.
    #[serde(skip)]
    pub derivative: f64,
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

/// A neuron, with weights, bias, and activation values.
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
    pub fn new(input_size: usize) -> Self {
        let mut weights = Vec::with_capacity(input_size);

        for _ in 0 .. input_size {
            weights.push(Weight::random())
        }

        Self {
            weights: weights.into(),
            bias: Bias::random(),
            activation: Activation::default(),
        }
    }

    /// Returns the expected size of an input, i.e. the number of weights.
    pub fn input_size(&self) -> usize {
        self.weights.len()
    }

    /// Computes activation values only, not the derivatives. Saves the result.
    /// Callers should certify that `input` has the same size as this neuron
    /// was constructed.
    pub fn compute_activation<F, I>(&mut self, activation_fn: &F, input: &[I])
    where
        F: ActivationFn,
        I: Input,
    {
        self.activation.input = 0.0;
        for (input, weight) in input.iter().zip(self.weights.iter_mut()) {
            self.activation.input += weight.val * input.activation();
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
        next_layer: &Layer,
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
    /// should be called first. Callers should certify that `input` has the same
    /// size as this neuron was constructed.
    pub fn compute_deriv_over_weight<I>(&mut self, input: &[I])
    where
        I: Input,
    {
        let iter = self.weights.iter_mut().zip(input.iter());
        for (weight, input) in iter {
            let dval = self.activation.deriv_over_val;
            let dinp = self.activation.deriv_over_input;
            weight.derivative = dval * dinp * input.activation();
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
    /// last layer. `compute_activation` should be called first. Callers should
    /// certify that `input` has the same size as this neuron was constructed
    /// with.
    pub fn compute_all_derivs<F, I>(
        &mut self,
        activation_fn: &F,
        input: &[I],
        index: usize,
        next_layer: &Layer,
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
    /// layer. `compute_activation` should be called first. Callers should
    /// certify that `input` has the same size as this neuron was constructed
    /// with.
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

    /// Returns the previously computed derivative over activation val.
    pub fn deriv_over_act_input(&self) -> f64 {
        self.activation.deriv_over_input
    }

    /// Returns the previously computed derivative over activation val.
    pub fn deriv_over_act_val(&self) -> f64 {
        self.activation.deriv_over_val
    }

    /// Returns reference to the weights of this neuron.
    pub fn weights(&self) -> &[Weight] {
        &self.weights
    }

    /// Returns reference to the bias of this neuron.
    pub fn bias(&self) -> &Bias {
        &self.bias
    }
}

impl Input for Neuron {
    fn activation(&self) -> f64 {
        self.activation.val
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::functions::LogisticFn;

    // 0.425504 for input [1.0, 0.5]
    pub fn neuron1() -> Neuron {
        let mut neuron = Neuron::new(2);

        neuron.weights[0].val = 0.225504;
        neuron.weights[1].val = 0.6;
        neuron.bias.val = -0.1;

        neuron
    }

    // 0 for input [1.0, 0.5]
    pub fn neuron2() -> Neuron {
        let mut neuron = Neuron::new(2);

        neuron.weights[0].val = 0.1;
        neuron.weights[1].val = 0.6;
        neuron.bias.val = -0.4;

        neuron
    }

    /// -0.425504 for input [1.0, 0.5]
    pub fn neuron3() -> Neuron {
        let mut neuron = Neuron::new(2);

        neuron.weights[0].val = -0.22504;
        neuron.weights[1].val = -0.6;
        neuron.bias.val = 0.1;

        neuron
    }

    /// 0.5 for input [0.5, 1.0, 0.25]
    pub fn neuron4() -> Neuron {
        let mut neuron = Neuron::new(3);

        neuron.weights[0].val = 0.4;
        neuron.weights[1].val = 0.3;
        neuron.weights[2].val = 0.8;
        neuron.bias.val = -0.2;

        neuron
    }

    /// 0.25 for input [0.5, 1.0, 0.25]
    pub fn neuron5() -> Neuron {
        let mut neuron = Neuron::new(3);

        neuron.weights[0].val = 0.4;
        neuron.weights[1].val = 0.1;
        neuron.weights[2].val = 0.4;
        neuron.bias.val = -0.15;

        neuron
    }

    #[test]
    fn activation_val() {
        let mut neuron = neuron1();
        neuron.compute_activation(&LogisticFn, &[1.0, 0.5]);
        assert_float_eq!(neuron.activation.val, 0.6048, 1e-3);

        let mut neuron = neuron2();
        neuron.compute_activation(&LogisticFn, &[1.0, 0.5]);
        assert_float_eq!(neuron.activation.val, 0.5, 1e-3);

        let mut neuron = neuron3();
        neuron.compute_activation(&LogisticFn, &[1.0, 0.5]);
        assert_float_eq!(neuron.activation.val, 0.3952, 1e-3);

        let mut neuron = neuron4();
        neuron.compute_activation(&LogisticFn, &[0.5, 1.0, 0.25]);
        assert_float_eq!(neuron.activation.val, 0.62246, 1e-3);

        let mut neuron = neuron5();
        neuron.compute_activation(&LogisticFn, &[0.5, 1.0, 0.25]);
        assert_float_eq!(neuron.activation.val, 0.56127, 1e-3);
    }

    #[test]
    fn deriv_over_act_input() {
        let mut neuron = neuron1();
        neuron.compute_activation(&LogisticFn, &[1.0, 0.5]);
        neuron.compute_deriv_over_act_input(&LogisticFn);
        assert_float_eq!(neuron.activation.val, 0.6048, 1e-3);
        assert_float_eq!(neuron.activation.deriv_over_input, 0.23901, 1e-3);

        let mut neuron = neuron2();
        neuron.compute_activation(&LogisticFn, &[1.0, 0.5]);
        neuron.compute_deriv_over_act_input(&LogisticFn);
        assert_float_eq!(neuron.activation.val, 0.5, 1e-3);
        assert_float_eq!(neuron.activation.deriv_over_input, 0.25, 1e-3);

        let mut neuron = neuron3();
        neuron.compute_activation(&LogisticFn, &[1.0, 0.5]);
        neuron.compute_deriv_over_act_input(&LogisticFn);
        assert_float_eq!(neuron.activation.val, 0.3952, 1e-3);
        assert_float_eq!(neuron.activation.deriv_over_input, 0.23901, 1e-3);

        let mut neuron = neuron4();
        neuron.compute_activation(&LogisticFn, &[0.5, 1.0, 0.25]);
        neuron.compute_deriv_over_act_input(&LogisticFn);
        assert_float_eq!(neuron.activation.val, 0.62246, 1e-3);
        assert_float_eq!(neuron.activation.deriv_over_input, 0.235, 1e-3);

        let mut neuron = neuron5();
        neuron.compute_activation(&LogisticFn, &[0.5, 1.0, 0.25]);
        neuron.compute_deriv_over_act_input(&LogisticFn);
        assert_float_eq!(neuron.activation.val, 0.56127, 1e-3);
        assert_float_eq!(neuron.activation.deriv_over_input, 0.24613, 1e-3);
    }
}
