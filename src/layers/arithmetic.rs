use crate::{
    error::Error,
    layers::{Layer, get_input},
    model_file::NodeFile,
    tensor::{Tensor, Values, get_broadcast_positions, get_broadcast_shape},
};

#[derive(Clone, Copy)]
enum ArithmeticKind {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Max,
    Min,
}

struct ArithmeticLayer {
    kind: ArithmeticKind,
}

pub(crate) fn find_layer(node: &NodeFile) -> Option<Box<dyn Layer>> {
    let kind = match node.kind.as_str() {
        "Add" | "Sum" => ArithmeticKind::Add,
        "Sub" => ArithmeticKind::Sub,
        "Mul" => ArithmeticKind::Mul,
        "Div" => ArithmeticKind::Div,
        "Pow" => ArithmeticKind::Pow,
        "Max" => ArithmeticKind::Max,
        "Min" => ArithmeticKind::Min,
        _ => return None,
    };
    Some(Box::new(ArithmeticLayer { kind }))
}

impl Layer for ArithmeticLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let first = get_input(inputs, 0)?;
        if inputs.len() == 1 {
            return Ok(vec![first.clone()]);
        }
        let mut output = self.get_output(first, get_input(inputs, 1)?)?;
        for index in 2..inputs.len() {
            output = self.get_output(&output, get_input(inputs, index)?)?;
        }
        Ok(vec![output])
    }
}

impl ArithmeticLayer {
    fn get_output(&self, left: &Tensor, right: &Tensor) -> Result<Tensor, Error> {
        match (&left.values, &right.values) {
            (Values::Int(_), Values::Int(_)) => self.get_int_output(left, right),
            _ => self.get_float_output(left, right),
        }
    }

    fn get_float_output(&self, left: &Tensor, right: &Tensor) -> Result<Tensor, Error> {
        let right = match right.values {
            Values::Int(_) => &Tensor::from_floats(right.shape.clone(), right.to_floats()),
            Values::Float(_) => right,
        };
        let is_square = right.len() == 1 && right.floats()?[0] == 2.0;
        let (shape, values) = match self.kind {
            ArithmeticKind::Pow if is_square => combine(left, right, |a: f32, _| a * a)?,
            ArithmeticKind::Pow => combine(left, right, |a: f32, b| a.powf(b))?,
            ArithmeticKind::Add => combine(left, right, |a: f32, b| a + b)?,
            ArithmeticKind::Sub => combine(left, right, |a: f32, b| a - b)?,
            ArithmeticKind::Mul => combine(left, right, |a: f32, b| a * b)?,
            ArithmeticKind::Div => combine(left, right, |a: f32, b| a / b)?,
            ArithmeticKind::Max => combine(left, right, |a: f32, b| a.max(b))?,
            ArithmeticKind::Min => combine(left, right, |a: f32, b| a.min(b))?,
        };
        Ok(Tensor::from_floats(shape, values))
    }

    fn get_int_output(&self, left: &Tensor, right: &Tensor) -> Result<Tensor, Error> {
        let (shape, values) = match self.kind {
            ArithmeticKind::Add => combine(left, right, |a: i64, b| a.wrapping_add(b))?,
            ArithmeticKind::Sub => combine(left, right, |a: i64, b| a.wrapping_sub(b))?,
            ArithmeticKind::Mul => combine(left, right, |a: i64, b| a.wrapping_mul(b))?,
            ArithmeticKind::Div => combine(left, right, |a: i64, b| a.checked_div(b).unwrap_or(0))?,
            ArithmeticKind::Max => combine(left, right, |a: i64, b| a.max(b))?,
            ArithmeticKind::Min => combine(left, right, |a: i64, b| a.min(b))?,
            ArithmeticKind::Pow => {
                combine(left, right, |a: i64, b| a.wrapping_pow(b.max(0) as u32))?
            }
        };
        Ok(Tensor::from_ints(shape, values))
    }
}

pub(crate) fn combine<T: Copy, O>(
    left: &Tensor,
    right: &Tensor,
    get_value: impl Fn(T, T) -> O,
) -> Result<(Vec<usize>, Vec<O>), Error>
where
    Tensor: ValuesOf<T>,
{
    let left_values = left.values_of()?;
    let right_values = right.values_of()?;
    let shape = get_broadcast_shape(&left.shape, &right.shape)?;

    if left.shape == right.shape {
        let values = left_values
            .iter()
            .zip(right_values)
            .map(|(a, b)| get_value(*a, *b))
            .collect();
        return Ok((shape, values));
    }
    if left.shape == shape
        && let Some(inner) = find_repeat(&right.shape, &shape)
    {
        let values = combine_repeated(left_values, right_values, inner, &get_value);
        return Ok((shape, values));
    }
    if right.shape == shape
        && let Some(inner) = find_repeat(&left.shape, &shape)
    {
        let values = combine_repeated(right_values, left_values, inner, |b, a| get_value(a, b));
        return Ok((shape, values));
    }

    let left_positions = get_broadcast_positions(&left.shape, &shape);
    let right_positions = get_broadcast_positions(&right.shape, &shape);
    let values = left_positions
        .iter()
        .zip(&right_positions)
        .map(|(left_at, right_at)| get_value(left_values[*left_at], right_values[*right_at]))
        .collect();
    Ok((shape, values))
}

fn find_repeat(small_shape: &[usize], shape: &[usize]) -> Option<usize> {
    let missing_axes = shape.len() - small_shape.len();
    let sizes: Vec<usize> = std::iter::repeat_n(1, missing_axes)
        .chain(small_shape.iter().copied())
        .collect();
    let used_axes: Vec<usize> = (0..shape.len()).filter(|axis| sizes[*axis] != 1).collect();
    let (Some(first), Some(last)) = (used_axes.first(), used_axes.last()) else {
        return Some(shape.iter().product());
    };
    let is_one_block = (*first..=*last).all(|axis| sizes[axis] == shape[axis]);
    is_one_block.then(|| shape[last + 1..].iter().product())
}

fn combine_repeated<T: Copy, O>(
    full: &[T],
    repeated: &[T],
    inner: usize,
    get_value: impl Fn(T, T) -> O,
) -> Vec<O> {
    let mut output = Vec::with_capacity(full.len());
    if full.is_empty() {
        return output;
    }
    if inner == 1 {
        for block in full.chunks_exact(repeated.len()) {
            output.extend(block.iter().zip(repeated).map(|(a, b)| get_value(*a, *b)));
        }
        return output;
    }
    for block in full.chunks_exact(repeated.len() * inner) {
        for (part, value) in block.chunks_exact(inner).zip(repeated) {
            output.extend(part.iter().map(|a| get_value(*a, *value)));
        }
    }
    output
}

pub(crate) trait ValuesOf<T> {
    fn values_of(&self) -> Result<&[T], Error>;
}

impl ValuesOf<f32> for Tensor {
    fn values_of(&self) -> Result<&[f32], Error> {
        self.floats()
    }
}

impl ValuesOf<i64> for Tensor {
    fn values_of(&self) -> Result<&[i64], Error> {
        self.ints()
    }
}
