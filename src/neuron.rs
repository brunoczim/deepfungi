use crate::{functions::ActivationFn, input::Input, layer::DenseLayer};
use rand::Rng;

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Weight {
    pub val: f64,
    #[serde(skip)]
    pub derivative: f64,
}

impl Weight {
    pub fn random() -> Self {
        Self { val: rand::thread_rng().gen_range(0.0, 1.0), derivative: 0.0 }
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Bias {
    pub val: f64,
    #[serde(skip)]
    pub derivative: f64,
}

impl Bias {
    pub fn random() -> Self {
        Self { val: rand::thread_rng().gen_range(0.0, 1.0), derivative: 0.0 }
    }
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct Activation {
    #[serde(skip)]
    pub input: f64,
    #[serde(skip)]
    pub val: f64,
    #[serde(skip)]
    pub deriv_over_input: f64,
    #[serde(skip)]
    pub deriv_over_val: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Neuron {
    pub weights: Box<[Weight]>,
    pub bias: Bias,
    pub activation: Activation,
}

impl Neuron {
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

    pub fn compute_deriv_over_act_input<F>(&mut self, activation_fn: &F)
    where
        F: ActivationFn,
    {
        self.activation.deriv_over_input =
            activation_fn.deriv(self.activation.input);
    }

    pub fn compute_deriv_over_act_val(
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

    pub fn compute_deriv_over_bias(&mut self) {
        let dval = self.activation.deriv_over_val;
        let dinp = self.activation.deriv_over_input;
        self.bias.derivative = dval * dinp;
    }

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
}

impl Input for Neuron {
    fn as_float(&self) -> f64 {
        self.activation.val
    }
}
