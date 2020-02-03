use super::*;
use crate::{
    functions::{LogisticFn, SquaredError},
    layer,
};

pub fn network1() -> Neural<LogisticFn, SquaredError> {
    let mut network = Neural::new(LogisticFn, SquaredError, 2, &[3], 2);
    network.layers[0] = layer::test::layer1();
    network.layers[1] = layer::test::layer2();
    network
}

#[test]
fn activation() {
    let mut network = network1();
    network.compute_activations(&[1.0, 0.5]);
    assert_float_eq!(network.layers[0].neurons()[0].activation(), 0.6048, 1e-3);
    assert_float_eq!(network.layers[0].neurons()[1].activation(), 0.5, 1e-3);
    assert_float_eq!(network.layers[0].neurons()[2].activation(), 0.3952, 1e-3);
    assert_float_eq!(
        network.layers[1].neurons()[0].activation(),
        0.62435,
        1e-3
    );
    assert_float_eq!(
        network.layers[1].neurons()[1].activation(),
        0.57444,
        1e-3
    );
}

#[test]
fn losses() {
    let mut network = network1();
    network.compute_losses(&[1.0, 0.5], &[0.0, 1.0]);
    assert_float_eq!(network.layers[0].neurons()[0].activation(), 0.6048, 1e-3);
    assert_float_eq!(network.layers[0].neurons()[1].activation(), 0.5, 1e-3);
    assert_float_eq!(network.layers[0].neurons()[2].activation(), 0.3952, 1e-3);
    assert_float_eq!(
        network.layers[1].neurons()[0].activation(),
        0.62435,
        1e-3
    );
    assert_float_eq!(
        network.layers[1].neurons()[1].activation(),
        0.57444,
        1e-3
    );

    assert_float_eq!(&network.losses[0], 0.38981, 1e-3);
    assert_float_eq!(&network.losses[1], 0.181101, 1e-3);
}

#[test]
fn losses_deriv() {
    let mut network = network1();
    network.compute_losses_deriv(&[1.0, 0.5], &[0.0, 1.0]);
    assert_float_eq!(network.layers[0].neurons()[0].activation(), 0.6048, 1e-3);
    assert_float_eq!(network.layers[0].neurons()[1].activation(), 0.5, 1e-3);
    assert_float_eq!(network.layers[0].neurons()[2].activation(), 0.3952, 1e-3);
    assert_float_eq!(
        network.layers[1].neurons()[0].activation(),
        0.62435,
        1e-3
    );
    assert_float_eq!(
        network.layers[1].neurons()[1].activation(),
        0.57444,
        1e-3
    );

    assert_float_eq!(&network.losses[0], 0.38981, 1e-3);
    assert_float_eq!(&network.losses[1], 0.181101, 1e-3);

    assert_float_eq!(&network.losses_deriv[0], 1.2487, 1e-3);
    assert_float_eq!(&network.losses_deriv[1], -0.85112, 1e-3);
}

#[test]
fn derivs() {
    let mut network = network1();
    network.compute_derivs(&[1.0, 0.5], &[0.0, 1.0]);
    assert_float_eq!(network.layers[0].neurons()[0].activation(), 0.6048, 1e-3);
    assert_float_eq!(network.layers[0].neurons()[1].activation(), 0.5, 1e-3);
    assert_float_eq!(network.layers[0].neurons()[2].activation(), 0.3952, 1e-3);
    assert_float_eq!(
        network.layers[1].neurons()[0].activation(),
        0.62435,
        1e-3
    );
    assert_float_eq!(
        network.layers[1].neurons()[1].activation(),
        0.57444,
        1e-3
    );

    assert_float_eq!(&network.losses[0], 0.38981, 1e-3);
    assert_float_eq!(&network.losses[1], 0.181101, 1e-3);

    assert_float_eq!(&network.losses_deriv[0], 1.2487, 1e-3);
    assert_float_eq!(&network.losses_deriv[1], -0.85112, 1e-3);

    assert_float_eq!(
        network.layers[0].neurons()[0].deriv_over_act_input(),
        0.23901,
        1e-3
    );
    assert_float_eq!(
        network.layers[0].neurons()[1].deriv_over_act_input(),
        0.25,
        1e-3
    );
    assert_float_eq!(
        network.layers[0].neurons()[2].deriv_over_act_input(),
        0.23901,
        1e-3
    );
    assert_float_eq!(
        network.layers[1].neurons()[0].deriv_over_act_input(),
        0.2345,
        1e-3
    );
    assert_float_eq!(
        network.layers[1].neurons()[1].deriv_over_act_input(),
        0.24446,
        1e-3
    );

    assert_float_eq!(
        network.layers[0].neurons()[0].deriv_over_act_val(),
        0.033902,
        1e-3
    );
    assert_float_eq!(
        network.layers[0].neurons()[1].deriv_over_act_val(),
        0.06703,
        1e-3
    );
    assert_float_eq!(
        network.layers[0].neurons()[2].deriv_over_act_val(),
        0.15103,
        1e-3
    );
    assert_float_eq!(
        network.layers[1].neurons()[0].deriv_over_act_val(),
        1.2487,
        1e-3
    );
    assert_float_eq!(
        network.layers[1].neurons()[1].deriv_over_act_val(),
        -0.85112,
        1e-3
    );

    assert_float_eq!(
        network.layers[0].neurons()[0].weights()[0].derivative,
        0.008107,
        1e-3
    );
    assert_float_eq!(
        network.layers[0].neurons()[0].weights()[1].derivative,
        0.004053,
        1e-3
    );
    assert_float_eq!(
        network.layers[0].neurons()[0].bias().derivative,
        0.008107,
        1e-3
    );

    assert_float_eq!(
        network.layers[0].neurons()[1].weights()[0].derivative,
        0.01675,
        1e-3
    );
    assert_float_eq!(
        network.layers[0].neurons()[1].weights()[1].derivative,
        0.008379,
        1e-3
    );
    assert_float_eq!(
        network.layers[0].neurons()[1].bias().derivative,
        0.01675,
        1e-3
    );

    assert_float_eq!(
        network.layers[0].neurons()[2].weights()[0].derivative,
        0.036098,
        1e-3
    );
    assert_float_eq!(
        network.layers[0].neurons()[2].weights()[1].derivative,
        0.018049,
        1e-3
    );
    assert_float_eq!(
        network.layers[0].neurons()[2].bias().derivative,
        0.036098,
        1e-3
    );

    assert_float_eq!(
        network.layers[1].neurons()[0].weights()[0].derivative,
        0.177098,
        1e-3
    );
    assert_float_eq!(
        network.layers[1].neurons()[0].weights()[1].derivative,
        0.14641,
        1e-3
    );
    assert_float_eq!(
        network.layers[1].neurons()[0].weights()[2].derivative,
        0.115722,
        1e-3
    );
    assert_float_eq!(
        network.layers[1].neurons()[0].bias().derivative,
        0.29282,
        1e-3
    );

    assert_float_eq!(
        network.layers[1].neurons()[1].weights()[0].derivative,
        -0.12583,
        1e-3
    );
    assert_float_eq!(
        network.layers[1].neurons()[1].weights()[1].derivative,
        -0.10403,
        1e-3
    );
    assert_float_eq!(
        network.layers[1].neurons()[1].weights()[2].derivative,
        -0.08222,
        1e-3
    );
    assert_float_eq!(
        network.layers[1].neurons()[1].bias().derivative,
        -0.20806,
        1e-3
    );
}
