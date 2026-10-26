pub(crate) mod arithmetic;
pub(crate) mod compare;
pub(crate) mod constant;
pub(crate) mod conv;
pub(crate) mod linear;
pub(crate) mod math;
pub(crate) mod normalize;
pub(crate) mod pool;
pub(crate) mod reduce;
pub(crate) mod select;
pub(crate) mod shape;
pub(crate) mod window;

use crate::{error::Error, model_file::NodeFile, tensor::Tensor};

pub(crate) trait Layer: Send + Sync {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error>;
}

type LayerFinder = fn(&NodeFile) -> Option<Box<dyn Layer>>;

const LAYER_FINDERS: &[LayerFinder] = &[
    math::find_layer,
    arithmetic::find_layer,
    compare::find_layer,
    linear::find_layer,
    conv::find_layer,
    pool::find_layer,
    reduce::find_layer,
    normalize::find_layer,
    shape::find_layer,
    select::find_layer,
];

pub(crate) fn get_layer(node: &NodeFile) -> Result<Box<dyn Layer>, Error> {
    LAYER_FINDERS
        .iter()
        .find_map(|find_layer| find_layer(node))
        .ok_or_else(|| Error::layer_not_supported(&node.kind))
}

pub(crate) fn get_input<'a>(
    inputs: &[Option<&'a Tensor>],
    index: usize,
) -> Result<&'a Tensor, Error> {
    find_input(inputs, index).ok_or_else(|| Error::wrong_input("an input is missing"))
}

pub(crate) fn find_input<'a>(inputs: &[Option<&'a Tensor>], index: usize) -> Option<&'a Tensor> {
    inputs.get(index).copied().flatten()
}
