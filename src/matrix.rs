#[cfg(target_arch = "x86_64")]
use crate::matrix_avx;
#[cfg(target_arch = "aarch64")]
use crate::matrix_neon;

const ROWS_AT_ONCE: usize = 4;
const COLUMNS_AT_ONCE: usize = 8;

pub(crate) struct Matrix<'a> {
    pub values: &'a [f32],
    pub rows: usize,
    pub columns: usize,
}

impl Matrix<'_> {
    pub(crate) fn get(&self, row: usize, column: usize) -> f32 {
        self.values[row * self.columns + column]
    }
}

pub(crate) type AddBlocks = fn(&Matrix, &Matrix, &mut [f32]) -> (usize, usize);

pub(crate) fn multiply_add(left: &Matrix, right: &Matrix, output: &mut [f32]) {
    multiply_add_with(get_fastest_add_blocks(), left, right, output);
}

pub(crate) fn multiply_add_with(
    add_blocks: AddBlocks,
    left: &Matrix,
    right: &Matrix,
    output: &mut [f32],
) {
    assert!(left.values.len() >= left.rows * left.columns);
    assert!(right.values.len() >= right.rows * right.columns);
    assert!(output.len() >= left.rows * right.columns);
    assert_eq!(left.columns, right.rows);
    if left.columns == 0 {
        return;
    }

    let (full_rows, full_columns) = add_blocks(left, right, output);
    add_simply(
        left,
        right,
        0..full_rows,
        full_columns..right.columns,
        output,
    );
    add_simply(left, right, full_rows..left.rows, 0..right.columns, output);
}

#[cfg(target_arch = "aarch64")]
fn get_fastest_add_blocks() -> AddBlocks {
    matrix_neon::add_blocks
}

#[cfg(target_arch = "x86_64")]
fn get_fastest_add_blocks() -> AddBlocks {
    if matrix_avx::is_available() {
        matrix_avx::add_blocks
    } else {
        add_blocks
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn get_fastest_add_blocks() -> AddBlocks {
    add_blocks
}

#[cfg_attr(all(target_arch = "aarch64", not(test)), allow(dead_code))]
pub(crate) fn add_blocks(left: &Matrix, right: &Matrix, output: &mut [f32]) -> (usize, usize) {
    let full_rows = left.rows - left.rows % ROWS_AT_ONCE;
    let full_columns = right.columns - right.columns % COLUMNS_AT_ONCE;
    for first_row in (0..full_rows).step_by(ROWS_AT_ONCE) {
        for first_column in (0..full_columns).step_by(COLUMNS_AT_ONCE) {
            add_block(left, right, first_row, first_column, output);
        }
    }
    (full_rows, full_columns)
}

#[cfg_attr(all(target_arch = "aarch64", not(test)), allow(dead_code))]
fn add_block(
    left: &Matrix,
    right: &Matrix,
    first_row: usize,
    first_column: usize,
    output: &mut [f32],
) {
    let mut sums = [[0.0_f32; COLUMNS_AT_ONCE]; ROWS_AT_ONCE];
    for inner in 0..left.columns {
        let start = inner * right.columns + first_column;
        let values: &[f32; COLUMNS_AT_ONCE] = right.values[start..start + COLUMNS_AT_ONCE]
            .try_into()
            .expect("a block of columns");
        for (row, row_sums) in sums.iter_mut().enumerate() {
            let weight = left.get(first_row + row, inner);
            for (sum, value) in row_sums.iter_mut().zip(values) {
                *sum += weight * value;
            }
        }
    }
    for (row, row_sums) in sums.iter().enumerate() {
        let start = (first_row + row) * right.columns + first_column;
        for (out, sum) in output[start..start + COLUMNS_AT_ONCE]
            .iter_mut()
            .zip(row_sums)
        {
            *out += sum;
        }
    }
}

fn add_simply(
    left: &Matrix,
    right: &Matrix,
    rows: std::ops::Range<usize>,
    columns: std::ops::Range<usize>,
    output: &mut [f32],
) {
    for row in rows {
        for column in columns.clone() {
            let sum: f32 = (0..left.columns)
                .map(|inner| left.get(row, inner) * right.get(inner, column))
                .sum();
            output[row * right.columns + column] += sum;
        }
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(crate) fn pack_weights(
    left: &Matrix,
    first_row: usize,
    rows_at_once: usize,
    packed_weights: &mut [f32],
) {
    for (inner, weights) in packed_weights.chunks_exact_mut(rows_at_once).enumerate() {
        for (row, weight) in weights.iter_mut().enumerate() {
            *weight = left.get(first_row + row, inner);
        }
    }
}

pub(crate) fn flip(values: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    let mut flipped = vec![0.0; values.len()];
    for row in 0..rows {
        for column in 0..columns {
            flipped[column * rows + row] = values[row * columns + column];
        }
    }
    flipped
}
