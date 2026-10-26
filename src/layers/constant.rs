use crate::{
    error::Error,
    model_file::{AttributeValue, NodeFile},
    tensor::Tensor,
};

pub(crate) fn is_constant(node: &NodeFile) -> bool {
    node.kind == "Constant"
}

pub(crate) fn get_constant(node: &NodeFile) -> Result<Tensor, Error> {
    let attribute = node
        .attributes
        .first()
        .ok_or_else(|| Error::wrong_file("a constant has no value"))?;
    let tensor = match &attribute.value {
        AttributeValue::Tensor(tensor) => tensor.clone(),
        AttributeValue::Float(value) => Tensor::from_floats(vec![], vec![*value]),
        AttributeValue::Int(value) => Tensor::from_ints(vec![], vec![*value]),
        AttributeValue::Floats(values) => Tensor::from_floats(vec![values.len()], values.clone()),
        AttributeValue::Ints(values) => Tensor::from_ints(vec![values.len()], values.clone()),
        _ => return Err(Error::wrong_file("a constant has an unsupported value")),
    };
    Ok(tensor)
}
