#![allow(dead_code, unused_macros)]

#[cfg(test)]
#[macro_use]
/// Test utilities.
mod test;

/// Artificial neurons and related items.
mod neuron;

/// Layers of neural networks.
mod layer;

/// Module providing an artificial neural network.
pub mod network;

/// Module providing functions to be used with the neural network.
pub mod functions;
