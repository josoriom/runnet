use crate::{
    error::Error,
    layers::{
        Layer, find_input, get_input,
        window::{Window, WindowSettings},
    },
    matrix::{Matrix, multiply_add},
    model_file::NodeFile,
    tensor::Tensor,
};

struct ConvLayer {
    settings: WindowSettings,
    groups: usize,
}

pub(crate) fn find_layer(node: &NodeFile) -> Option<Box<dyn Layer>> {
    if node.kind != "Conv" {
        return None;
    }
    Some(Box::new(ConvLayer {
        settings: WindowSettings::from_node(node),
        groups: node.get_int("group", 1).max(1) as usize,
    }))
}

impl Layer for ConvLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        let weights = get_input(inputs, 1)?;
        let bias = find_input(inputs, 2)
            .map(|bias| bias.floats())
            .transpose()?;
        if input.rank() < 3 || weights.rank() != input.rank() {
            return Err(Error::wrong_input(
                "conv needs inputs with a batch and channels",
            ));
        }

        let batch_count = input.shape[0];
        let input_channels = input.shape[1];
        let output_channels = weights.shape[0];
        let channels_per_group = weights.shape[1];
        if channels_per_group * self.groups != input_channels || output_channels % self.groups != 0
        {
            return Err(Error::wrong_input("conv channels do not match the weights"));
        }

        let window = self
            .settings
            .get_window(&input.shape[2..], Some(&weights.shape[2..]))?;
        let input_size = window.input[0] * window.input[1];
        let output_size = window.output[0] * window.output[1];
        let kernel_size = window.kernel[0] * window.kernel[1];
        let outputs_per_group = output_channels / self.groups;

        let input_values = input.floats()?;
        let weight_values = weights.floats()?;
        let group_weight_size = outputs_per_group * channels_per_group * kernel_size;
        let mut output = vec![0.0_f32; batch_count * output_channels * output_size];
        let mut columns = vec![0.0_f32; channels_per_group * kernel_size * output_size];

        for (sample, sample_output) in output
            .chunks_exact_mut(output_channels * output_size)
            .enumerate()
        {
            for (channel, plane) in sample_output.chunks_exact_mut(output_size).enumerate() {
                plane.fill(bias.map_or(0.0, |bias| bias[channel]));
            }
            for (group, group_output) in sample_output
                .chunks_exact_mut(outputs_per_group * output_size)
                .enumerate()
            {
                let input_start =
                    (sample * input_channels + group * channels_per_group) * input_size;
                let group_input =
                    &input_values[input_start..input_start + channels_per_group * input_size];
                fill_columns(&mut columns, group_input, &window);
                let group_weights =
                    &weight_values[group * group_weight_size..(group + 1) * group_weight_size];
                multiply_add(
                    &Matrix {
                        values: group_weights,
                        rows: outputs_per_group,
                        columns: channels_per_group * kernel_size,
                    },
                    &Matrix {
                        values: &columns,
                        rows: channels_per_group * kernel_size,
                        columns: output_size,
                    },
                    group_output,
                );
            }
        }

        let mut shape = vec![batch_count, output_channels];
        shape.extend_from_slice(&window.output[2 - (input.rank() - 2)..]);
        Ok(vec![Tensor::from_floats(shape, output)])
    }
}

fn fill_columns(columns: &mut [f32], input: &[f32], window: &Window) {
    let input_size = window.input[0] * window.input[1];
    let output_size = window.output[0] * window.output[1];
    let output_width = window.output[1];
    let input_width = window.input[1];
    columns.fill(0.0);

    let mut column_rows = columns.chunks_exact_mut(output_size);
    for plane in input.chunks_exact(input_size) {
        for kernel_row in 0..window.kernel[0] {
            let rows = window.get_valid_outputs(0, kernel_row);
            for kernel_column in 0..window.kernel[1] {
                let column_row = column_rows.next().expect("a row for each kernel place");
                let places = window.get_valid_outputs(1, kernel_column);
                if places.is_empty() {
                    continue;
                }
                for output_row in rows.clone() {
                    let input_row = window.get_input_place(0, output_row, kernel_row) as usize;
                    let first_input =
                        window.get_input_place(1, places.start, kernel_column) as usize;
                    let input_line = &plane[input_row * input_width + first_input..];
                    let output_line = &mut column_row[output_row * output_width..][places.clone()];
                    let step = window.strides[1];
                    for (out, value) in output_line.iter_mut().zip(input_line.iter().step_by(step))
                    {
                        *out = *value;
                    }
                }
            }
        }
    }
}
