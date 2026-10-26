use std::ops::Range;

use crate::{error::Error, model_file::NodeFile};

pub(crate) struct WindowSettings {
    kernel: Option<Vec<i64>>,
    strides: Option<Vec<i64>>,
    dilations: Option<Vec<i64>>,
    pads: Option<Vec<i64>>,
    auto_pad: String,
    round_up: bool,
}

pub(crate) struct Window {
    pub input: [usize; 2],
    pub output: [usize; 2],
    pub kernel: [usize; 2],
    pub strides: [usize; 2],
    pub dilations: [usize; 2],
    pub pads_before: [usize; 2],
}

impl WindowSettings {
    pub(crate) fn from_node(node: &NodeFile) -> Self {
        Self {
            kernel: node.get_ints("kernel_shape"),
            strides: node.get_ints("strides"),
            dilations: node.get_ints("dilations"),
            pads: node.get_ints("pads"),
            auto_pad: node.get_text("auto_pad").unwrap_or("NOTSET").to_string(),
            round_up: node.get_int("ceil_mode", 0) == 1,
        }
    }

    pub(crate) fn get_window(
        &self,
        input: &[usize],
        kernel: Option<&[usize]>,
    ) -> Result<Window, Error> {
        let spatial_count = input.len();
        if spatial_count == 0 || spatial_count > 2 {
            return Err(Error::wrong_input("only 1D and 2D windows are supported"));
        }
        let kernel: Vec<usize> = match (kernel, &self.kernel) {
            (Some(kernel), _) => kernel.to_vec(),
            (None, Some(kernel)) => kernel.iter().map(|size| *size as usize).collect(),
            (None, None) => return Err(Error::wrong_input("the window has no kernel size")),
        };
        let strides = get_list(&self.strides, spatial_count, 1);
        let dilations = get_list(&self.dilations, spatial_count, 1);
        let pads = get_list(&self.pads, spatial_count * 2, 0);

        let mut window = Window {
            input: [1, 1],
            output: [1, 1],
            kernel: [1, 1],
            strides: [1, 1],
            dilations: [1, 1],
            pads_before: [0, 0],
        };
        let first_axis = 2 - spatial_count;
        for axis in 0..spatial_count {
            let place = first_axis + axis;
            window.input[place] = input[axis];
            window.kernel[place] = kernel[axis];
            window.strides[place] = strides[axis];
            window.dilations[place] = dilations[axis];
            let span = (kernel[axis] - 1) * dilations[axis] + 1;
            let (pad_before, output) = self.get_pad_and_output(
                input[axis],
                span,
                strides[axis],
                pads[axis],
                pads[axis + spatial_count],
            );
            window.pads_before[place] = pad_before;
            window.output[place] = output;
        }
        Ok(window)
    }

    fn get_pad_and_output(
        &self,
        input: usize,
        span: usize,
        stride: usize,
        pad_before: usize,
        pad_after: usize,
    ) -> (usize, usize) {
        match self.auto_pad.as_str() {
            "SAME_UPPER" | "SAME_LOWER" => {
                let output = input.div_ceil(stride);
                let total_pad = ((output - 1) * stride + span).saturating_sub(input);
                let pad_before = if self.auto_pad == "SAME_UPPER" {
                    total_pad / 2
                } else {
                    total_pad.div_ceil(2)
                };
                (pad_before, output)
            }
            "VALID" => (0, (input.saturating_sub(span)) / stride + 1),
            _ => {
                let padded = (input + pad_before + pad_after).saturating_sub(span);
                let output = if self.round_up {
                    padded.div_ceil(stride) + 1
                } else {
                    padded / stride + 1
                };
                (pad_before, output)
            }
        }
    }
}

impl Window {
    pub(crate) fn get_input_place(
        &self,
        axis: usize,
        output_place: usize,
        kernel_place: usize,
    ) -> isize {
        (output_place * self.strides[axis] + kernel_place * self.dilations[axis]) as isize
            - self.pads_before[axis] as isize
    }

    pub(crate) fn get_valid_outputs(&self, axis: usize, kernel_place: usize) -> Range<usize> {
        let shift =
            self.pads_before[axis] as isize - (kernel_place * self.dilations[axis]) as isize;
        let stride = self.strides[axis] as isize;
        let first = if shift > 0 {
            (shift + stride - 1) / stride
        } else {
            0
        };
        let last = (self.input[axis] as isize - 1 + shift).div_euclid(stride) + 1;
        let first = first as usize;
        let last = last.clamp(0, self.output[axis] as isize) as usize;
        first.min(last)..last
    }
}

fn get_list(values: &Option<Vec<i64>>, count: usize, default: usize) -> Vec<usize> {
    match values {
        Some(values) if values.len() == count => {
            values.iter().map(|value| *value as usize).collect()
        }
        _ => vec![default; count],
    }
}
