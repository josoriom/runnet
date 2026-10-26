use crate::error::Error;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Values {
    Float(Vec<f32>),
    Int(Vec<i64>),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum NumberKind {
    Float,
    Int,
}

impl NumberKind {
    pub(crate) fn get_name(self) -> &'static str {
        match self {
            NumberKind::Float => "floats",
            NumberKind::Int => "ints",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub(crate) shape: Vec<usize>,
    pub(crate) values: Values,
}

impl Tensor {
    pub fn from_floats(shape: impl Into<Vec<usize>>, values: impl Into<Vec<f32>>) -> Self {
        Self {
            shape: shape.into(),
            values: Values::Float(values.into()),
        }
    }

    pub fn from_ints(shape: impl Into<Vec<usize>>, values: impl Into<Vec<i64>>) -> Self {
        Self {
            shape: shape.into(),
            values: Values::Int(values.into()),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn into_floats(self) -> Result<Vec<f32>, Error> {
        match self.values {
            Values::Float(values) => Ok(values),
            Values::Int(_) => Err(Error::wrong_type("floats", "ints")),
        }
    }

    pub fn into_ints(self) -> Result<Vec<i64>, Error> {
        match self.values {
            Values::Int(values) => Ok(values),
            Values::Float(_) => Err(Error::wrong_type("ints", "floats")),
        }
    }

    pub(crate) fn get_kind(&self) -> NumberKind {
        match self.values {
            Values::Float(_) => NumberKind::Float,
            Values::Int(_) => NumberKind::Int,
        }
    }

    pub(crate) fn len(&self) -> usize {
        match &self.values {
            Values::Float(values) => values.len(),
            Values::Int(values) => values.len(),
        }
    }

    pub(crate) fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn floats(&self) -> Result<&[f32], Error> {
        match &self.values {
            Values::Float(values) => Ok(values),
            Values::Int(_) => Err(Error::wrong_type("floats", "ints")),
        }
    }

    pub fn ints(&self) -> Result<&[i64], Error> {
        match &self.values {
            Values::Int(values) => Ok(values),
            Values::Float(_) => Err(Error::wrong_type("ints", "floats")),
        }
    }

    pub(crate) fn to_floats(&self) -> Vec<f32> {
        match &self.values {
            Values::Float(values) => values.clone(),
            Values::Int(values) => values.iter().map(|value| *value as f32).collect(),
        }
    }

    pub(crate) fn to_ints(&self) -> Vec<i64> {
        match &self.values {
            Values::Float(values) => values.iter().map(|value| *value as i64).collect(),
            Values::Int(values) => values.clone(),
        }
    }

    pub(crate) fn with_shape(&self, shape: Vec<usize>) -> Result<Tensor, Error> {
        if shape.iter().product::<usize>() != self.len() {
            return Err(Error::wrong_input("the new shape has a different size"));
        }
        Ok(Tensor {
            shape,
            values: self.values.clone(),
        })
    }

    pub(crate) fn pick(&self, positions: impl Iterator<Item = usize>, shape: Vec<usize>) -> Tensor {
        let values = match &self.values {
            Values::Float(values) => Values::Float(positions.map(|at| values[at]).collect()),
            Values::Int(values) => Values::Int(positions.map(|at| values[at]).collect()),
        };
        Tensor { shape, values }
    }
}

pub(crate) struct Lines {
    pub(crate) outer: usize,
    pub(crate) length: usize,
    pub(crate) inner: usize,
}

pub(crate) fn get_lines(shape: &[usize], axis: usize) -> Lines {
    Lines {
        outer: shape[..axis].iter().product(),
        length: shape[axis],
        inner: shape[axis + 1..].iter().product(),
    }
}

pub(crate) fn get_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

pub(crate) fn get_axis(axis: i64, rank: usize) -> Result<usize, Error> {
    let axis = if axis < 0 { axis + rank as i64 } else { axis };
    if axis < 0 || axis as usize >= rank.max(1) {
        return Err(Error::wrong_input("the axis is out of range"));
    }
    Ok(axis as usize)
}

pub(crate) fn get_broadcast_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>, Error> {
    let rank = left.len().max(right.len());
    let mut shape = vec![1; rank];
    for (axis, size) in shape.iter_mut().enumerate() {
        let left_size = get_size_from_end(left, rank - axis);
        let right_size = get_size_from_end(right, rank - axis);
        *size = match (left_size, right_size) {
            (a, b) if a == b => a,
            (1, b) => b,
            (a, 1) => a,
            _ => return Err(Error::wrong_input("the shapes can not be broadcast")),
        };
    }
    Ok(shape)
}

fn get_size_from_end(shape: &[usize], place_from_end: usize) -> usize {
    if place_from_end > shape.len() {
        1
    } else {
        shape[shape.len() - place_from_end]
    }
}

pub(crate) fn get_broadcast_positions(shape: &[usize], output_shape: &[usize]) -> Vec<usize> {
    if shape == output_shape {
        return (0..output_shape.iter().product()).collect();
    }
    let missing_axes = output_shape.len() - shape.len();
    let strides = get_strides(shape);
    let output_strides: Vec<usize> = (0..output_shape.len())
        .map(|axis| {
            if axis < missing_axes || shape[axis - missing_axes] == 1 {
                0
            } else {
                strides[axis - missing_axes]
            }
        })
        .collect();

    get_positions(output_shape, &output_strides)
}

pub(crate) fn get_positions(shape: &[usize], strides: &[usize]) -> Vec<usize> {
    let size: usize = shape.iter().product();
    let mut positions = Vec::with_capacity(size);
    let mut counter = vec![0; shape.len()];
    let mut position = 0;
    for _ in 0..size {
        positions.push(position);
        for axis in (0..shape.len()).rev() {
            counter[axis] += 1;
            position += strides[axis];
            if counter[axis] < shape[axis] {
                break;
            }
            position -= strides[axis] * counter[axis];
            counter[axis] = 0;
        }
    }
    positions
}
