use std::borrow::Cow;

use crate::{
    error::Error,
    layers::{Layer, find_input, get_input},
    matrix::{Matrix, flip, multiply_add},
    model_file::NodeFile,
    tensor::{Tensor, get_broadcast_positions, get_broadcast_shape},
};

struct MatMulLayer;

struct GemmLayer {
    alpha: f32,
    beta: f32,
    flip_left: bool,
    flip_right: bool,
}

pub(crate) fn find_layer(node: &NodeFile) -> Option<Box<dyn Layer>> {
    match node.kind.as_str() {
        "MatMul" => Some(Box::new(MatMulLayer)),
        "Gemm" => Some(Box::new(GemmLayer {
            alpha: node.get_float("alpha", 1.0),
            beta: node.get_float("beta", 1.0),
            flip_left: node.get_int("transA", 0) == 1,
            flip_right: node.get_int("transB", 0) == 1,
        })),
        _ => None,
    }
}

impl Layer for MatMulLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let left = get_input(inputs, 0)?;
        let right = get_input(inputs, 1)?;
        let left_shape = if left.rank() == 1 {
            vec![1, left.shape[0]]
        } else {
            left.shape.clone()
        };
        let right_shape = if right.rank() == 1 {
            vec![right.shape[0], 1]
        } else {
            right.shape.clone()
        };

        let (left_batch, left_matrix) = left_shape.split_at(left_shape.len() - 2);
        let (right_batch, right_matrix) = right_shape.split_at(right_shape.len() - 2);
        let [rows, inner] = [left_matrix[0], left_matrix[1]];
        let [right_inner, columns] = [right_matrix[0], right_matrix[1]];
        if inner != right_inner {
            return Err(Error::wrong_input("matmul sizes do not match"));
        }

        let batch = get_broadcast_shape(left_batch, right_batch)?;
        let left_positions = get_broadcast_positions(left_batch, &batch);
        let right_positions = get_broadcast_positions(right_batch, &batch);
        let left_values = left.floats()?;
        let right_values = right.floats()?;
        let mut output = vec![0.0_f32; left_positions.len() * rows * columns];

        for ((output_block, left_at), right_at) in output
            .chunks_exact_mut(rows * columns)
            .zip(&left_positions)
            .zip(&right_positions)
        {
            let left_start = left_at * rows * inner;
            let right_start = right_at * inner * columns;
            multiply_add(
                &Matrix {
                    values: &left_values[left_start..left_start + rows * inner],
                    rows,
                    columns: inner,
                },
                &Matrix {
                    values: &right_values[right_start..right_start + inner * columns],
                    rows: inner,
                    columns,
                },
                output_block,
            );
        }

        let mut shape = batch;
        if left.rank() != 1 {
            shape.push(rows);
        }
        if right.rank() != 1 {
            shape.push(columns);
        }
        Ok(vec![Tensor::from_floats(shape, output)])
    }
}

impl Layer for GemmLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let left = get_input(inputs, 0)?;
        let right = get_input(inputs, 1)?;
        if left.rank() != 2 || right.rank() != 2 {
            return Err(Error::wrong_input("gemm needs two matrices"));
        }

        let left_values = if self.flip_left {
            Cow::Owned(flip(left.floats()?, left.shape[0], left.shape[1]))
        } else {
            Cow::Borrowed(left.floats()?)
        };
        let (rows, inner) = if self.flip_left {
            (left.shape[1], left.shape[0])
        } else {
            (left.shape[0], left.shape[1])
        };
        let (right_inner, columns) = if self.flip_right {
            (right.shape[1], right.shape[0])
        } else {
            (right.shape[0], right.shape[1])
        };
        if inner != right_inner {
            return Err(Error::wrong_input("gemm sizes do not match"));
        }

        let right_values = if self.flip_right {
            Cow::Owned(flip(right.floats()?, right.shape[0], right.shape[1]))
        } else {
            Cow::Borrowed(right.floats()?)
        };
        let mut output = vec![0.0_f32; rows * columns];
        multiply_add(
            &Matrix {
                values: &left_values,
                rows,
                columns: inner,
            },
            &Matrix {
                values: &right_values,
                rows: inner,
                columns,
            },
            &mut output,
        );
        self.add_bias(&mut output, find_input(inputs, 2), rows, columns)?;
        Ok(vec![Tensor::from_floats(vec![rows, columns], output)])
    }
}

impl GemmLayer {
    fn add_bias(
        &self,
        output: &mut [f32],
        bias: Option<&Tensor>,
        rows: usize,
        columns: usize,
    ) -> Result<(), Error> {
        let output_shape = [rows, columns];
        let bias_positions = bias
            .map(|bias| -> Result<Vec<usize>, Error> {
                get_broadcast_shape(&bias.shape, &output_shape)?;
                Ok(get_broadcast_positions(&bias.shape, &output_shape))
            })
            .transpose()?;
        let bias_values = bias.map(|bias| bias.floats()).transpose()?;
        for (at, value) in output.iter_mut().enumerate() {
            *value *= self.alpha;
            if let (Some(positions), Some(bias)) = (&bias_positions, bias_values) {
                *value += self.beta * bias[positions[at]];
            }
        }
        Ok(())
    }
}
