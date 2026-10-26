use std::arch::x86_64::{
    _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_storeu_ps,
};

use crate::matrix::{Matrix, pack_weights};

const ROWS_AT_ONCE: usize = 8;
const COLUMNS_AT_ONCE: usize = 8;

pub(crate) fn is_available() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

pub(crate) fn add_blocks(left: &Matrix, right: &Matrix, output: &mut [f32]) -> (usize, usize) {
    assert!(is_available());
    let full_rows = left.rows - left.rows % ROWS_AT_ONCE;
    let full_columns = right.columns - right.columns % COLUMNS_AT_ONCE;
    let mut packed_weights = vec![0.0_f32; left.columns * ROWS_AT_ONCE];
    for first_row in (0..full_rows).step_by(ROWS_AT_ONCE) {
        pack_weights(left, first_row, ROWS_AT_ONCE, &mut packed_weights);
        for first_column in (0..full_columns).step_by(COLUMNS_AT_ONCE) {
            unsafe { add_block(&packed_weights, right, first_row, first_column, output) };
        }
    }
    (full_rows, full_columns)
}

#[target_feature(enable = "avx2,fma")]
fn add_block(
    packed_weights: &[f32],
    right: &Matrix,
    first_row: usize,
    first_column: usize,
    output: &mut [f32],
) {
    let inner_count = packed_weights.len() / ROWS_AT_ONCE;
    let last_value = (inner_count.max(1) - 1) * right.columns + first_column + COLUMNS_AT_ONCE;
    let last_output =
        (first_row + ROWS_AT_ONCE - 1) * right.columns + first_column + COLUMNS_AT_ONCE;
    assert!(last_value <= right.values.len() && last_output <= output.len());

    let mut sums = [_mm256_setzero_ps(); ROWS_AT_ONCE];
    for (inner, weights) in packed_weights
        .as_chunks::<ROWS_AT_ONCE>()
        .0
        .iter()
        .enumerate()
    {
        let values_at = unsafe {
            right
                .values
                .as_ptr()
                .add(inner * right.columns + first_column)
        };
        let values = unsafe { _mm256_loadu_ps(values_at) };
        for (sum, weight) in sums.iter_mut().zip(weights) {
            *sum = _mm256_fmadd_ps(_mm256_set1_ps(*weight), values, *sum);
        }
    }

    for (row, sum) in sums.iter().enumerate() {
        let start = (first_row + row) * right.columns + first_column;
        unsafe {
            let output_at = output.as_mut_ptr().add(start);
            _mm256_storeu_ps(output_at, _mm256_add_ps(_mm256_loadu_ps(output_at), *sum));
        }
    }
}
