use crate::matrix::{AddBlocks, Matrix, add_blocks, multiply_add_with};

const SHAPES: [(usize, usize, usize); 6] = [
    (1, 1, 1),
    (3, 5, 7),
    (8, 8, 8),
    (8, 0, 8),
    (13, 37, 21),
    (16, 64, 64),
];
const TOLERANCE: f32 = 1e-4;

fn get_values(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1 << 24) as f32 - 0.5
        })
        .collect()
}

fn get_plain_product(left: &Matrix, right: &Matrix, start: &[f32]) -> Vec<f32> {
    let mut output = start.to_vec();
    for row in 0..left.rows {
        for column in 0..right.columns {
            for inner in 0..left.columns {
                output[row * right.columns + column] +=
                    left.get(row, inner) * right.get(inner, column);
            }
        }
    }
    output
}

fn assert_kernel_matches_plain_loop(add_blocks: AddBlocks) {
    for (rows, inner, columns) in SHAPES {
        let left_values = get_values(rows * inner, 1);
        let right_values = get_values(inner * columns, 2);
        let start = get_values(rows * columns, 3);
        let left = Matrix {
            values: &left_values,
            rows,
            columns: inner,
        };
        let right = Matrix {
            values: &right_values,
            rows: inner,
            columns,
        };

        let mut output = start.clone();
        multiply_add_with(add_blocks, &left, &right, &mut output);

        let expected = get_plain_product(&left, &right, &start);
        for (at, (found, expected)) in output.iter().zip(&expected).enumerate() {
            assert!(
                (found - expected).abs() <= TOLERANCE,
                "{}x{}x{} value {} is {} but should be {}",
                rows,
                inner,
                columns,
                at,
                found,
                expected
            );
        }
    }
}

#[test]
fn portable_kernel_matches_plain_loop() {
    assert_kernel_matches_plain_loop(add_blocks);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_kernel_matches_plain_loop() {
    assert_kernel_matches_plain_loop(crate::matrix_neon::add_blocks);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx_kernel_matches_plain_loop() {
    if crate::matrix_avx::is_available() {
        assert_kernel_matches_plain_loop(crate::matrix_avx::add_blocks);
    }
}

const GUARD: f32 = 12345.0;
const GUARD_SIZE: usize = 16;
const OFFSETS: [usize; 3] = [0, 1, 3];
const LARGE_SHAPES: [(usize, usize, usize); 3] = [(33, 65, 47), (64, 128, 96), (5, 300, 9)];

fn get_guarded(values: &[f32], offset: usize) -> Vec<f32> {
    let mut guarded = vec![GUARD; offset];
    guarded.extend_from_slice(values);
    guarded.extend(std::iter::repeat_n(GUARD, GUARD_SIZE));
    guarded
}

fn assert_guards_untouched(buffer: &[f32], offset: usize, length: usize, name: &str) {
    let guards = buffer[..offset].iter().chain(&buffer[offset + length..]);
    for guard in guards {
        assert_eq!(
            guard.to_bits(),
            GUARD.to_bits(),
            "a guard around {} was overwritten",
            name
        );
    }
}

fn assert_case_is_right_and_in_bounds(
    add_blocks: AddBlocks,
    (rows, inner, columns): (usize, usize, usize),
    offset: usize,
) {
    let left_values = get_guarded(&get_values(rows * inner, 1), offset);
    let right_values = get_guarded(&get_values(inner * columns, 2), offset);
    let start = get_values(rows * columns, 3);
    let mut output = get_guarded(&start, offset);
    let left = Matrix {
        values: &left_values[offset..offset + rows * inner],
        rows,
        columns: inner,
    };
    let right = Matrix {
        values: &right_values[offset..offset + inner * columns],
        rows: inner,
        columns,
    };

    multiply_add_with(add_blocks, &left, &right, &mut output[offset..]);

    assert_guards_untouched(&output, offset, rows * columns, "the output");
    assert_guards_untouched(&left_values, offset, rows * inner, "the left matrix");
    assert_guards_untouched(&right_values, offset, inner * columns, "the right matrix");
    let expected = get_plain_product(&left, &right, &start);
    for (at, (found, expected)) in output[offset..].iter().zip(&expected).enumerate() {
        assert!(
            (found - expected).abs() <= TOLERANCE,
            "{}x{}x{} at offset {} value {} is {} but should be {}",
            rows,
            inner,
            columns,
            offset,
            at,
            found,
            expected
        );
    }
}

fn assert_kernel_is_right_and_in_bounds(add_blocks: AddBlocks) {
    for rows in 0..=9 {
        for inner in 0..=9 {
            for columns in 0..=17 {
                for offset in OFFSETS {
                    assert_case_is_right_and_in_bounds(add_blocks, (rows, inner, columns), offset);
                }
            }
        }
    }
    for shape in LARGE_SHAPES {
        for offset in OFFSETS {
            assert_case_is_right_and_in_bounds(add_blocks, shape, offset);
        }
    }
}

#[test]
fn portable_kernel_is_right_and_in_bounds() {
    assert_kernel_is_right_and_in_bounds(add_blocks);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_kernel_is_right_and_in_bounds() {
    assert_kernel_is_right_and_in_bounds(crate::matrix_neon::add_blocks);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx_kernel_is_right_and_in_bounds() {
    if crate::matrix_avx::is_available() {
        assert_kernel_is_right_and_in_bounds(crate::matrix_avx::add_blocks);
    }
}
