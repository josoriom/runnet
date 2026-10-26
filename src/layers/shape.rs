use crate::{
    error::Error,
    layers::{Layer, find_input, get_input},
    model_file::{AttributeValue, NodeFile, get_number_kind},
    tensor::{
        NumberKind, Tensor, Values, get_axis, get_broadcast_positions, get_broadcast_shape,
        get_positions, get_strides,
    },
};

struct IdentityLayer;

struct ReshapeLayer {
    allow_zero: bool,
}

struct FlattenLayer {
    axis: i64,
}

struct SqueezeLayer;

struct UnsqueezeLayer;

struct TransposeLayer {
    order: Option<Vec<i64>>,
}

struct ExpandLayer;

struct ShapeLayer {
    start: i64,
    end: Option<i64>,
}

struct CastLayer {
    to_float: bool,
}

struct FilledLayer {
    value: Tensor,
}

struct RangeLayer;

pub(crate) fn find_layer(node: &NodeFile) -> Option<Box<dyn Layer>> {
    let layer: Box<dyn Layer> = match node.kind.as_str() {
        "Identity" | "Dropout" => Box::new(IdentityLayer),
        "Reshape" => Box::new(ReshapeLayer {
            allow_zero: node.get_int("allowzero", 0) == 1,
        }),
        "Flatten" => Box::new(FlattenLayer {
            axis: node.get_int("axis", 1),
        }),
        "Squeeze" => Box::new(SqueezeLayer),
        "Unsqueeze" => Box::new(UnsqueezeLayer),
        "Transpose" => Box::new(TransposeLayer {
            order: node.get_ints("perm"),
        }),
        "Expand" => Box::new(ExpandLayer),
        "Shape" => Box::new(ShapeLayer {
            start: node.get_int("start", 0),
            end: node.find("end").map(|_| node.get_int("end", 0)),
        }),
        "Cast" => Box::new(CastLayer {
            to_float: get_number_kind(node.get_int("to", 1)) == Some(NumberKind::Float),
        }),
        "ConstantOfShape" => Box::new(FilledLayer {
            value: match node.find("value") {
                Some(AttributeValue::Tensor(tensor)) => tensor.clone(),
                _ => Tensor::from_floats(vec![1], vec![0.0]),
            },
        }),
        "Range" => Box::new(RangeLayer),
        _ => return None,
    };
    Some(layer)
}

impl Layer for IdentityLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        Ok(vec![get_input(inputs, 0)?.clone()])
    }
}

impl Layer for ReshapeLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        let wanted = get_input(inputs, 1)?.to_ints();
        let mut shape = Vec::with_capacity(wanted.len());
        let mut unknown_axis = None;
        for (axis, size) in wanted.iter().enumerate() {
            match *size {
                -1 => {
                    unknown_axis = Some(axis);
                    shape.push(1);
                }
                0 if !self.allow_zero => shape.push(*input.shape.get(axis).unwrap_or(&1)),
                size => shape.push(size as usize),
            }
        }
        if let Some(axis) = unknown_axis {
            let known: usize = shape.iter().product();
            shape[axis] = input.len().checked_div(known).unwrap_or(0);
        }
        Ok(vec![input.with_shape(shape)?])
    }
}

impl Layer for FlattenLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        let axis = get_axis(self.axis, input.rank() + 1)?;
        let rows = input.shape[..axis].iter().product();
        let columns = input.shape[axis..].iter().product();
        Ok(vec![input.with_shape(vec![rows, columns])?])
    }
}

impl Layer for SqueezeLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        let axes = get_axes(&read_axes(inputs), input.rank())?;
        let shape = input
            .shape
            .iter()
            .enumerate()
            .filter(|(axis, size)| {
                let remove = if axes.is_empty() {
                    **size == 1
                } else {
                    axes.contains(axis)
                };
                !remove
            })
            .map(|(_, size)| *size)
            .collect();
        Ok(vec![input.with_shape(shape)?])
    }
}

impl Layer for UnsqueezeLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        let axes = read_axes(inputs);
        let output_rank = input.rank() + axes.len();
        let axes = get_axes(&axes, output_rank)?;
        let mut sizes = input.shape.iter();
        let shape = (0..output_rank)
            .map(|axis| {
                if axes.contains(&axis) {
                    1
                } else {
                    *sizes.next().unwrap_or(&1)
                }
            })
            .collect();
        Ok(vec![input.with_shape(shape)?])
    }
}

fn read_axes(inputs: &[Option<&Tensor>]) -> Vec<i64> {
    find_input(inputs, 1)
        .map(Tensor::to_ints)
        .unwrap_or_default()
}

fn get_axes(axes: &[i64], rank: usize) -> Result<Vec<usize>, Error> {
    axes.iter().map(|axis| get_axis(*axis, rank)).collect()
}

impl Layer for TransposeLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        let order: Vec<usize> = match &self.order {
            Some(order) => order.iter().map(|axis| *axis as usize).collect(),
            None => (0..input.rank()).rev().collect(),
        };
        if order.len() != input.rank() || order.iter().any(|axis| *axis >= input.rank()) {
            return Err(Error::wrong_input(
                "transpose order does not match the input",
            ));
        }

        let shape: Vec<usize> = order.iter().map(|axis| input.shape[*axis]).collect();
        let input_strides = get_strides(&input.shape);
        let moved_strides: Vec<usize> = order.iter().map(|axis| input_strides[*axis]).collect();
        let positions = get_positions(&shape, &moved_strides);
        Ok(vec![input.pick(positions.into_iter(), shape)])
    }
}

impl Layer for ExpandLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        let wanted: Vec<usize> = get_input(inputs, 1)?
            .to_ints()
            .iter()
            .map(|size| *size as usize)
            .collect();
        let shape = get_broadcast_shape(&input.shape, &wanted)?;
        let positions = get_broadcast_positions(&input.shape, &shape);
        Ok(vec![input.pick(positions.into_iter(), shape)])
    }
}

impl Layer for ShapeLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        let rank = input.rank() as i64;
        let start = get_clamped(self.start, rank);
        let end = get_clamped(self.end.unwrap_or(rank), rank).max(start);
        let sizes: Vec<i64> = input.shape[start..end]
            .iter()
            .map(|size| *size as i64)
            .collect();
        Ok(vec![Tensor::from_ints(vec![sizes.len()], sizes)])
    }
}

fn get_clamped(axis: i64, rank: i64) -> usize {
    let axis = if axis < 0 { axis + rank } else { axis };
    axis.clamp(0, rank) as usize
}

impl Layer for CastLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        let values = if self.to_float {
            Values::Float(input.to_floats())
        } else {
            Values::Int(input.to_ints())
        };
        Ok(vec![Tensor {
            shape: input.shape.clone(),
            values,
        }])
    }
}

impl Layer for FilledLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let shape: Vec<usize> = get_input(inputs, 0)?
            .to_ints()
            .iter()
            .map(|size| *size as usize)
            .collect();
        let size = shape.iter().product();
        Ok(vec![self.value.pick(std::iter::repeat_n(0, size), shape)])
    }
}

impl Layer for RangeLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let start = get_input(inputs, 0)?;
        let limit = get_input(inputs, 1)?;
        let step = get_input(inputs, 2)?;
        match &start.values {
            Values::Int(_) => {
                let [start, limit, step] = [start, limit, step].map(|tensor| tensor.to_ints()[0]);
                if step == 0 {
                    return Err(Error::wrong_input("range step is zero"));
                }
                let count = ((limit - start) as f64 / step as f64).ceil().max(0.0) as usize;
                let values: Vec<_> = (0..count).map(|at| start + at as i64 * step).collect();
                Ok(vec![Tensor::from_ints(vec![count], values)])
            }
            Values::Float(_) => {
                let [start, limit, step] = [start, limit, step].map(|tensor| tensor.to_floats()[0]);
                let count = ((limit - start) / step).ceil().max(0.0) as usize;
                let values: Vec<_> = (0..count).map(|at| start + at as f32 * step).collect();
                Ok(vec![Tensor::from_floats(vec![count], values)])
            }
        }
    }
}
