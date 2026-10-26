use crate::{
    error::Error,
    layers::{
        Layer, get_input,
        window::{Window, WindowSettings},
    },
    model_file::NodeFile,
    tensor::Tensor,
};

#[derive(Clone, Copy)]
enum PoolKind {
    Max,
    Average { count_padding: bool },
}

struct PoolLayer {
    kind: PoolKind,
    settings: WindowSettings,
}

struct GlobalPoolLayer {
    kind: PoolKind,
}

pub(crate) fn find_layer(node: &NodeFile) -> Option<Box<dyn Layer>> {
    let average = PoolKind::Average {
        count_padding: node.get_int("count_include_pad", 0) == 1,
    };
    let settings = WindowSettings::from_node(node);
    match node.kind.as_str() {
        "MaxPool" => Some(Box::new(PoolLayer {
            kind: PoolKind::Max,
            settings,
        })),
        "AveragePool" => Some(Box::new(PoolLayer {
            kind: average,
            settings,
        })),
        "GlobalMaxPool" => Some(Box::new(GlobalPoolLayer {
            kind: PoolKind::Max,
        })),
        "GlobalAveragePool" => Some(Box::new(GlobalPoolLayer { kind: average })),
        _ => None,
    }
}

impl Layer for PoolLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        if input.rank() < 3 {
            return Err(Error::wrong_input(
                "pool needs inputs with a batch and channels",
            ));
        }
        let window = self.settings.get_window(&input.shape[2..], None)?;
        let input_size = window.input[0] * window.input[1];
        let output_size = window.output[0] * window.output[1];

        let mut output = Vec::with_capacity(input.shape[0] * input.shape[1] * output_size);
        for plane in input.floats()?.chunks_exact(input_size) {
            for row in 0..window.output[0] {
                for column in 0..window.output[1] {
                    output.push(self.get_value(plane, &window, row, column));
                }
            }
        }

        let mut shape = input.shape[..2].to_vec();
        shape.extend_from_slice(&window.output[2 - (input.rank() - 2)..]);
        Ok(vec![Tensor::from_floats(shape, output)])
    }
}

impl PoolLayer {
    fn get_value(&self, plane: &[f32], window: &Window, row: usize, column: usize) -> f32 {
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0;
        let mut count = 0;
        for kernel_row in 0..window.kernel[0] {
            let input_row = window.get_input_place(0, row, kernel_row);
            for kernel_column in 0..window.kernel[1] {
                let input_column = window.get_input_place(1, column, kernel_column);
                let is_inside = input_row >= 0
                    && input_column >= 0
                    && (input_row as usize) < window.input[0]
                    && (input_column as usize) < window.input[1];
                if is_inside {
                    let value = plane[input_row as usize * window.input[1] + input_column as usize];
                    max = max.max(value);
                    sum += value;
                    count += 1;
                } else if matches!(
                    self.kind,
                    PoolKind::Average {
                        count_padding: true
                    }
                ) {
                    count += 1;
                }
            }
        }
        match self.kind {
            PoolKind::Max => max,
            PoolKind::Average { .. } => sum / count.max(1) as f32,
        }
    }
}

impl Layer for GlobalPoolLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        if input.rank() < 3 {
            return Err(Error::wrong_input(
                "pool needs inputs with a batch and channels",
            ));
        }
        let plane_size: usize = input.shape[2..].iter().product();
        let output: Vec<_> = input
            .floats()?
            .chunks_exact(plane_size)
            .map(|plane| match self.kind {
                PoolKind::Max => plane.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                PoolKind::Average { .. } => plane.iter().sum::<f32>() / plane_size as f32,
            })
            .collect();

        let mut shape = input.shape[..2].to_vec();
        shape.resize(input.rank(), 1);
        Ok(vec![Tensor::from_floats(shape, output)])
    }
}
