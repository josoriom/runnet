#![doc = include_str!("../README.md")]
#![deny(unreachable_pub)]
#![deny(unsafe_code)]

mod error;
mod graph;
mod layers;
mod matrix;
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod matrix_avx;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod matrix_neon;
mod model;
mod model_file;
mod proto_reader;
mod tensor;
#[cfg(test)]
mod tests;

pub use error::Error;
pub use model::Model;
pub use tensor::Tensor;
