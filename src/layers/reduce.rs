use crate::{
    error::Error,
    layers::{Layer, find_input, get_input},
    model_file::NodeFile,
    tensor::{Lines, Tensor, Values, get_axis, get_lines},
};

#[derive(Clone, Copy)]
enum ReduceKind {
    Mean,
    Max,
    Min,
    Sum,
}

struct ReduceLayer {
    kind: ReduceKind,
    axes: Option<Vec<i64>>,
    keep_axes: bool,
    skip_without_axes: bool,
}

struct CumSumLayer {
    exclusive: bool,
    reverse: bool,
}

pub(crate) fn find_layer(node: &NodeFile) -> Option<Box<dyn Layer>> {
    let kind = match node.kind.as_str() {
        "ReduceMean" => ReduceKind::Mean,
        "ReduceMax" => ReduceKind::Max,
        "ReduceMin" => ReduceKind::Min,
        "ReduceSum" => ReduceKind::Sum,
        "CumSum" => {
            return Some(Box::new(CumSumLayer {
                exclusive: node.get_int("exclusive", 0) == 1,
                reverse: node.get_int("reverse", 0) == 1,
            }));
        }
        _ => return None,
    };
    Some(Box::new(ReduceLayer {
        kind,
        axes: node.get_ints("axes"),
        keep_axes: node.get_int("keepdims", 1) == 1,
        skip_without_axes: node.get_int("noop_with_empty_axes", 0) == 1,
    }))
}

impl Layer for ReduceLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        let axes = match find_input(inputs, 1) {
            Some(axes) => axes.to_ints(),
            None => self.axes.clone().unwrap_or_default(),
        };
        if axes.is_empty() && self.skip_without_axes {
            return Ok(vec![input.clone()]);
        }
        let mut axes: Vec<usize> = if axes.is_empty() {
            (0..input.rank()).collect()
        } else {
            axes.iter()
                .map(|axis| get_axis(*axis, input.rank()))
                .collect::<Result<_, _>>()?
        };
        axes.sort_unstable();
        axes.dedup();

        let mut shape = input.shape.clone();
        let mut values = input.to_floats();
        for axis in &axes {
            values = self.reduce(&values, &get_lines(&shape, *axis));
            shape[*axis] = 1;
        }
        if !self.keep_axes {
            shape = shape
                .iter()
                .enumerate()
                .filter(|(axis, _)| !axes.contains(axis))
                .map(|(_, size)| *size)
                .collect();
        }
        let output = Tensor::from_floats(shape, values);
        match input.values {
            Values::Int(_) => Ok(vec![Tensor::from_ints(
                output.shape.clone(),
                output.to_ints(),
            )]),
            Values::Float(_) => Ok(vec![output]),
        }
    }
}

impl ReduceLayer {
    fn reduce(&self, values: &[f32], lines: &Lines) -> Vec<f32> {
        let start_value = match self.kind {
            ReduceKind::Max => f32::NEG_INFINITY,
            ReduceKind::Min => f32::INFINITY,
            ReduceKind::Mean | ReduceKind::Sum => 0.0,
        };
        let mut output = vec![start_value; lines.outer * lines.inner];
        if lines.inner == 0 || lines.length == 0 {
            return output;
        }
        if lines.inner == 1 {
            for (line, result) in values.chunks_exact(lines.length).zip(output.iter_mut()) {
                *result = match self.kind {
                    ReduceKind::Max => line.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                    ReduceKind::Min => line.iter().copied().fold(f32::INFINITY, f32::min),
                    ReduceKind::Mean => get_sum(line) / lines.length as f32,
                    ReduceKind::Sum => get_sum(line),
                };
            }
            return output;
        }
        let blocks = values.chunks_exact(lines.length * lines.inner);
        for (block, results) in blocks.zip(output.chunks_exact_mut(lines.inner)) {
            for row in block.chunks_exact(lines.inner) {
                let pairs = results.iter_mut().zip(row);
                match self.kind {
                    ReduceKind::Max => {
                        pairs.for_each(|(result, value)| *result = result.max(*value))
                    }
                    ReduceKind::Min => {
                        pairs.for_each(|(result, value)| *result = result.min(*value))
                    }
                    ReduceKind::Mean | ReduceKind::Sum => {
                        pairs.for_each(|(result, value)| *result += value)
                    }
                }
            }
        }
        if matches!(self.kind, ReduceKind::Mean) {
            output
                .iter_mut()
                .for_each(|result| *result /= lines.length as f32);
        }
        output
    }
}

impl Layer for CumSumLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        let axis = get_input(inputs, 1)?
            .to_ints()
            .first()
            .copied()
            .ok_or_else(|| Error::wrong_input("cumsum needs an axis"))?;
        let lines = get_lines(&input.shape, get_axis(axis, input.rank())?);
        let values = input.floats()?;
        let mut output = vec![0.0_f32; values.len()];

        for outer in 0..lines.outer {
            for inner in 0..lines.inner {
                let mut total = 0.0;
                for step in 0..lines.length {
                    let place = if self.reverse {
                        lines.length - 1 - step
                    } else {
                        step
                    };
                    let at = (outer * lines.length + place) * lines.inner + inner;
                    if self.exclusive {
                        output[at] = total;
                        total += values[at];
                    } else {
                        total += values[at];
                        output[at] = total;
                    }
                }
            }
        }
        Ok(vec![Tensor::from_floats(input.shape.clone(), output)])
    }
}

fn get_sum(values: &[f32]) -> f32 {
    let (blocks, rest) = values.as_chunks::<8>();
    let mut sums = [0.0_f32; 8];
    for block in blocks {
        for (sum, value) in sums.iter_mut().zip(block) {
            *sum += value;
        }
    }
    sums.iter().sum::<f32>() + rest.iter().sum::<f32>()
}
