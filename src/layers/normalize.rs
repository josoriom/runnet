use crate::{
    error::Error,
    layers::{Layer, find_input, get_input},
    model_file::NodeFile,
    tensor::{Tensor, get_axis, get_lines},
};

struct SoftmaxLayer {
    axis: i64,
    use_log: bool,
}

struct LayerNormLayer {
    axis: i64,
    epsilon: f32,
}

struct BatchNormLayer {
    epsilon: f32,
}

pub(crate) fn find_layer(node: &NodeFile) -> Option<Box<dyn Layer>> {
    match node.kind.as_str() {
        "Softmax" => Some(Box::new(SoftmaxLayer {
            axis: node.get_int("axis", -1),
            use_log: false,
        })),
        "LogSoftmax" => Some(Box::new(SoftmaxLayer {
            axis: node.get_int("axis", -1),
            use_log: true,
        })),
        "LayerNormalization" => Some(Box::new(LayerNormLayer {
            axis: node.get_int("axis", -1),
            epsilon: node.get_float("epsilon", 1e-5),
        })),
        "BatchNormalization" => Some(Box::new(BatchNormLayer {
            epsilon: node.get_float("epsilon", 1e-5),
        })),
        _ => None,
    }
}

impl Layer for SoftmaxLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        let lines = get_lines(&input.shape, get_axis(self.axis, input.rank())?);
        let mut output = input.floats()?.to_vec();
        for outer in 0..lines.outer {
            for inner in 0..lines.inner {
                let first = outer * lines.length * lines.inner + inner;
                let places = (0..lines.length).map(|step| first + step * lines.inner);
                self.normalize_line(&mut output, places);
            }
        }
        Ok(vec![Tensor::from_floats(input.shape.clone(), output)])
    }
}

impl SoftmaxLayer {
    fn normalize_line(&self, values: &mut [f32], places: impl Iterator<Item = usize> + Clone) {
        let max = places
            .clone()
            .map(|at| values[at])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0;
        for at in places.clone() {
            values[at] = (values[at] - max).exp();
            total += values[at];
        }
        for at in places {
            values[at] = if self.use_log {
                (values[at] / total).ln()
            } else {
                values[at] / total
            };
        }
    }
}

impl Layer for LayerNormLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        let scale = get_input(inputs, 1)?.floats()?;
        let bias = find_input(inputs, 2)
            .map(|bias| bias.floats())
            .transpose()?;
        let axis = get_axis(self.axis, input.rank())?;
        let length: usize = input.shape[axis..].iter().product();
        if scale.len() != length || bias.is_some_and(|bias| bias.len() != length) {
            return Err(Error::wrong_input("layer norm weights have the wrong size"));
        }

        let mut output = input.floats()?.to_vec();
        for line in output.chunks_exact_mut(length) {
            let mean = line.iter().sum::<f32>() / length as f32;
            let variance =
                line.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / length as f32;
            let spread = (variance + self.epsilon).sqrt();
            for (at, value) in line.iter_mut().enumerate() {
                *value = (*value - mean) / spread * scale[at] + bias.map_or(0.0, |bias| bias[at]);
            }
        }
        Ok(vec![Tensor::from_floats(input.shape.clone(), output)])
    }
}

impl Layer for BatchNormLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        let scale = get_input(inputs, 1)?.floats()?;
        let bias = get_input(inputs, 2)?.floats()?;
        let mean = get_input(inputs, 3)?.floats()?;
        let variance = get_input(inputs, 4)?.floats()?;
        if input.rank() < 2 {
            return Err(Error::wrong_input("batch norm needs channels"));
        }
        let channels = input.shape[1];
        let plane_size: usize = input.shape[2..].iter().product();

        let mut output = input.floats()?.to_vec();
        for (at, plane) in output.chunks_exact_mut(plane_size).enumerate() {
            let channel = at % channels;
            let factor = scale[channel] / (variance[channel] + self.epsilon).sqrt();
            let shift = bias[channel] - mean[channel] * factor;
            for value in plane.iter_mut() {
                *value = *value * factor + shift;
            }
        }
        Ok(vec![Tensor::from_floats(input.shape.clone(), output)])
    }
}
