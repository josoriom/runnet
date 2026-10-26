use crate::{
    error::Error,
    layers::{Layer, find_input, get_input},
    model_file::NodeFile,
    tensor::{Tensor, Values},
};

#[derive(Clone, Copy)]
enum MathKind {
    Relu,
    LeakyRelu(f32),
    Elu(f32),
    Sigmoid,
    HardSigmoid(f32, f32),
    Tanh,
    Softplus,
    Gelu,
    GeluTanh,
    Erf,
    Exp,
    Log,
    Sqrt,
    Reciprocal,
    Abs,
    Neg,
    Floor,
    Ceil,
    Round,
    Sin,
    Cos,
}

struct MathLayer {
    kind: MathKind,
}

struct ClipLayer;

pub(crate) fn find_layer(node: &NodeFile) -> Option<Box<dyn Layer>> {
    let kind = match node.kind.as_str() {
        "Relu" => MathKind::Relu,
        "LeakyRelu" => MathKind::LeakyRelu(node.get_float("alpha", 0.01)),
        "Elu" => MathKind::Elu(node.get_float("alpha", 1.0)),
        "Sigmoid" => MathKind::Sigmoid,
        "HardSigmoid" => {
            MathKind::HardSigmoid(node.get_float("alpha", 0.2), node.get_float("beta", 0.5))
        }
        "Tanh" => MathKind::Tanh,
        "Softplus" => MathKind::Softplus,
        "Gelu" if node.get_text("approximate") == Some("tanh") => MathKind::GeluTanh,
        "Gelu" => MathKind::Gelu,
        "Erf" => MathKind::Erf,
        "Exp" => MathKind::Exp,
        "Log" => MathKind::Log,
        "Sqrt" => MathKind::Sqrt,
        "Reciprocal" => MathKind::Reciprocal,
        "Abs" => MathKind::Abs,
        "Neg" => MathKind::Neg,
        "Floor" => MathKind::Floor,
        "Ceil" => MathKind::Ceil,
        "Round" => MathKind::Round,
        "Sin" => MathKind::Sin,
        "Cos" => MathKind::Cos,
        "Clip" => return Some(Box::new(ClipLayer)),
        _ => return None,
    };
    Some(Box::new(MathLayer { kind }))
}

impl Layer for MathLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        let values = match (&input.values, self.kind) {
            (Values::Int(values), MathKind::Abs) => {
                Values::Int(values.iter().map(|v| v.abs()).collect())
            }
            (Values::Int(values), MathKind::Neg) => {
                Values::Int(values.iter().map(|v| -v).collect())
            }
            _ => Values::Float(self.get_outputs(input.floats()?)),
        };
        Ok(vec![Tensor {
            shape: input.shape.clone(),
            values,
        }])
    }
}

impl MathLayer {
    fn get_outputs(&self, values: &[f32]) -> Vec<f32> {
        match self.kind {
            MathKind::Relu => map(values, |x| x.max(0.0)),
            MathKind::LeakyRelu(alpha) => map(values, |x| if x < 0.0 { alpha * x } else { x }),
            MathKind::Elu(alpha) => map(
                values,
                |x| if x < 0.0 { alpha * (x.exp() - 1.0) } else { x },
            ),
            MathKind::Sigmoid => map(values, get_sigmoid),
            MathKind::HardSigmoid(alpha, beta) => {
                map(values, |x| (alpha * x + beta).clamp(0.0, 1.0))
            }
            MathKind::Tanh => map(values, f32::tanh),
            MathKind::Softplus => map(values, get_softplus),
            MathKind::Gelu => map(values, |x| {
                0.5 * x * (1.0 + get_erf(x / std::f32::consts::SQRT_2))
            }),
            MathKind::GeluTanh => map(values, get_gelu_tanh),
            MathKind::Erf => map(values, get_erf),
            MathKind::Exp => map(values, f32::exp),
            MathKind::Log => map(values, f32::ln),
            MathKind::Sqrt => map(values, f32::sqrt),
            MathKind::Reciprocal => map(values, |x| 1.0 / x),
            MathKind::Abs => map(values, f32::abs),
            MathKind::Neg => map(values, |x| -x),
            MathKind::Floor => map(values, f32::floor),
            MathKind::Ceil => map(values, f32::ceil),
            MathKind::Round => map(values, f32::round_ties_even),
            MathKind::Sin => map(values, f32::sin),
            MathKind::Cos => map(values, f32::cos),
        }
    }
}

fn map(values: &[f32], get_output: impl Fn(f32) -> f32) -> Vec<f32> {
    values.iter().map(|value| get_output(*value)).collect()
}

impl Layer for ClipLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        let min = get_limit(find_input(inputs, 1), f32::MIN)?;
        let max = get_limit(find_input(inputs, 2), f32::MAX)?;
        let values: Vec<_> = input
            .floats()?
            .iter()
            .map(|value| value.max(min).min(max))
            .collect();
        Ok(vec![Tensor::from_floats(input.shape.clone(), values)])
    }
}

fn get_limit(input: Option<&Tensor>, default: f32) -> Result<f32, Error> {
    match input {
        Some(tensor) => tensor
            .to_floats()
            .first()
            .copied()
            .ok_or_else(|| Error::wrong_input("a clip limit is empty")),
        None => Ok(default),
    }
}

fn get_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp = x.exp();
        exp / (1.0 + exp)
    }
}

fn get_softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

fn get_gelu_tanh(x: f32) -> f32 {
    let inner = (2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

fn get_erf(x: f32) -> f32 {
    let z = x.abs();
    let polynomial = 1.0
        + z * (0.070_523_08
            + z * (0.042_282_01
                + z * (0.009_270_527
                    + z * (0.000_152_014_3 + z * (0.000_276_567_2 + z * 0.000_043_063_8)))));
    let squared = polynomial * polynomial;
    let fourth = squared * squared;
    let eighth = fourth * fourth;
    (1.0 - 1.0 / (eighth * eighth)).copysign(x)
}
