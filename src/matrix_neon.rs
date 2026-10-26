use std::arch::aarch64::{
    float32x4_t, vaddq_f32, vdupq_n_f32, vfmaq_laneq_f32, vld1q_f32, vst1q_f32,
};

use crate::matrix::{Matrix, pack_weights};

const ROWS_AT_ONCE: usize = 8;
const COLUMNS_AT_ONCE: usize = 8;

type Pair = [float32x4_t; 2];

pub(crate) fn add_blocks(left: &Matrix, right: &Matrix, output: &mut [f32]) -> (usize, usize) {
    let full_rows = left.rows - left.rows % ROWS_AT_ONCE;
    let full_columns = right.columns - right.columns % COLUMNS_AT_ONCE;
    let mut packed_weights = vec![0.0_f32; left.columns * ROWS_AT_ONCE];
    for first_row in (0..full_rows).step_by(ROWS_AT_ONCE) {
        pack_weights(left, first_row, ROWS_AT_ONCE, &mut packed_weights);
        for first_column in (0..full_columns).step_by(COLUMNS_AT_ONCE) {
            add_block(&packed_weights, right, first_row, first_column, output);
        }
    }
    (full_rows, full_columns)
}

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

    let zero = unsafe { vdupq_n_f32(0.0) };
    let mut sums = [[zero; 2]; ROWS_AT_ONCE];
    for inner in 0..inner_count {
        let (values, low_weights, high_weights) = unsafe {
            let values_at = right
                .values
                .as_ptr()
                .add(inner * right.columns + first_column);
            let weights_at = packed_weights.as_ptr().add(inner * ROWS_AT_ONCE);
            (
                [vld1q_f32(values_at), vld1q_f32(values_at.add(4))],
                vld1q_f32(weights_at),
                vld1q_f32(weights_at.add(4)),
            )
        };
        add_row::<0>(&mut sums[0], values, low_weights);
        add_row::<1>(&mut sums[1], values, low_weights);
        add_row::<2>(&mut sums[2], values, low_weights);
        add_row::<3>(&mut sums[3], values, low_weights);
        add_row::<0>(&mut sums[4], values, high_weights);
        add_row::<1>(&mut sums[5], values, high_weights);
        add_row::<2>(&mut sums[6], values, high_weights);
        add_row::<3>(&mut sums[7], values, high_weights);
    }

    for (row, row_sums) in sums.iter().enumerate() {
        let start = (first_row + row) * right.columns + first_column;
        unsafe {
            let output_at = output.as_mut_ptr().add(start);
            vst1q_f32(output_at, vaddq_f32(vld1q_f32(output_at), row_sums[0]));
            vst1q_f32(
                output_at.add(4),
                vaddq_f32(vld1q_f32(output_at.add(4)), row_sums[1]),
            );
        }
    }
}

#[inline(always)]
fn add_row<const LANE: i32>(sums: &mut Pair, values: Pair, weights: float32x4_t) {
    unsafe {
        sums[0] = vfmaq_laneq_f32::<LANE>(sums[0], values[0], weights);
        sums[1] = vfmaq_laneq_f32::<LANE>(sums[1], values[1], weights);
    }
}
