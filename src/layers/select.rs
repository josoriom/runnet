use crate::{
    error::Error,
    layers::{Layer, find_input, get_input},
    model_file::NodeFile,
    tensor::{Tensor, Values, get_axis, get_positions, get_strides},
};

struct ConcatLayer {
    axis: i64,
}

struct SliceLayer;

struct GatherLayer {
    axis: i64,
}

struct GatherElementsLayer {
    axis: i64,
}

struct SplitLayer {
    axis: i64,
    output_count: usize,
}

struct PadLayer;

struct TileLayer;

pub(crate) fn find_layer(node: &NodeFile) -> Option<Box<dyn Layer>> {
    let layer: Box<dyn Layer> = match node.kind.as_str() {
        "Concat" => Box::new(ConcatLayer {
            axis: node.get_int("axis", 0),
        }),
        "Slice" => Box::new(SliceLayer),
        "Gather" => Box::new(GatherLayer {
            axis: node.get_int("axis", 0),
        }),
        "GatherElements" => Box::new(GatherElementsLayer {
            axis: node.get_int("axis", 0),
        }),
        "Split" => Box::new(SplitLayer {
            axis: node.get_int("axis", 0),
            output_count: node.outputs.len(),
        }),
        "Pad" if node.get_text("mode").unwrap_or("constant") == "constant" => Box::new(PadLayer),
        "Tile" => Box::new(TileLayer),
        _ => return None,
    };
    Some(layer)
}

impl Layer for ConcatLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let parts: Vec<&Tensor> = inputs.iter().flatten().copied().collect();
        let first = parts
            .first()
            .ok_or_else(|| Error::wrong_input("concat has no inputs"))?;
        let axis = get_axis(self.axis, first.rank())?;
        let outer: usize = first.shape[..axis].iter().product();

        let mut shape = first.shape.clone();
        shape[axis] = parts.iter().map(|part| part.shape[axis]).sum();
        let blocks: Vec<usize> = parts.iter().map(|part| part.len() / outer.max(1)).collect();

        let all_ints = parts
            .iter()
            .all(|part| matches!(part.values, Values::Int(_)));
        let values = if all_ints {
            let lists: Vec<Vec<i64>> = parts.iter().map(|part| part.to_ints()).collect();
            Values::Int(join_blocks(&lists, &blocks, outer))
        } else {
            let lists: Vec<Vec<f32>> = parts.iter().map(|part| part.to_floats()).collect();
            Values::Float(join_blocks(&lists, &blocks, outer))
        };
        Ok(vec![Tensor { shape, values }])
    }
}

fn join_blocks<T: Copy>(lists: &[Vec<T>], blocks: &[usize], outer: usize) -> Vec<T> {
    let mut output = Vec::with_capacity(lists.iter().map(Vec::len).sum());
    for row in 0..outer {
        for (list, block) in lists.iter().zip(blocks) {
            output.extend_from_slice(&list[row * block..(row + 1) * block]);
        }
    }
    output
}

impl Layer for SliceLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        let starts = get_input(inputs, 1)?.to_ints();
        let ends = get_input(inputs, 2)?.to_ints();
        let axes = match find_input(inputs, 3) {
            Some(axes) => axes.to_ints(),
            None => (0..starts.len() as i64).collect(),
        };
        let steps = match find_input(inputs, 4) {
            Some(steps) => steps.to_ints(),
            None => vec![1; starts.len()],
        };

        let input_strides = get_strides(&input.shape);
        let mut shape = input.shape.clone();
        let mut strides = input_strides.clone();
        let mut first_position = 0;
        for at in 0..starts.len() {
            let axis = get_axis(axes[at], input.rank())?;
            let step = steps[at];
            if step <= 0 {
                return Err(Error::wrong_input(
                    "only positive slice steps are supported",
                ));
            }
            let size = input.shape[axis] as i64;
            let start = get_slice_place(starts[at], size);
            let end = get_slice_place(ends[at], size).max(start);
            shape[axis] = ((end - start) as usize).div_ceil(step as usize);
            strides[axis] = input_strides[axis] * step as usize;
            first_position += start as usize * input_strides[axis];
        }

        let positions = get_positions(&shape, &strides);
        Ok(vec![input.pick(
            positions.into_iter().map(|at| at + first_position),
            shape,
        )])
    }
}

fn get_slice_place(place: i64, size: i64) -> i64 {
    let place = if place < 0 { place + size } else { place };
    place.clamp(0, size)
}

impl Layer for GatherLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        let indices = get_input(inputs, 1)?;
        let axis = get_axis(self.axis, input.rank())?;
        let size = input.shape[axis] as i64;
        let outer: usize = input.shape[..axis].iter().product();
        let inner: usize = input.shape[axis + 1..].iter().product();

        let mut places = Vec::with_capacity(indices.len());
        for index in indices.to_ints() {
            let place = if index < 0 { index + size } else { index };
            if place < 0 || place >= size {
                return Err(Error::wrong_input("a gather index is out of range"));
            }
            places.push(place as usize);
        }

        let mut shape = input.shape[..axis].to_vec();
        shape.extend_from_slice(&indices.shape);
        shape.extend_from_slice(&input.shape[axis + 1..]);
        let positions = (0..outer).flat_map(|row| {
            places.iter().flat_map(move |place| {
                let start = (row * size as usize + place) * inner;
                start..start + inner
            })
        });
        Ok(vec![input.pick(positions, shape)])
    }
}

impl Layer for GatherElementsLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        let indices = get_input(inputs, 1)?;
        if indices.rank() != input.rank() {
            return Err(Error::wrong_input(
                "gather elements needs indices of the same rank",
            ));
        }
        let axis = get_axis(self.axis, input.rank())?;
        let size = input.shape[axis] as i64;
        let input_strides = get_strides(&input.shape);
        let index_positions = get_positions(&indices.shape, &input_strides);
        let index_strides = get_strides(&indices.shape);

        let mut positions = Vec::with_capacity(indices.len());
        for (at, index) in indices.to_ints().into_iter().enumerate() {
            let place = if index < 0 { index + size } else { index };
            if place < 0 || place >= size {
                return Err(Error::wrong_input("a gather index is out of range"));
            }
            let place_on_axis = at / index_strides[axis] % indices.shape[axis];
            let start = index_positions[at] - place_on_axis * input_strides[axis];
            positions.push(start + place as usize * input_strides[axis]);
        }
        Ok(vec![
            input.pick(positions.into_iter(), indices.shape.clone()),
        ])
    }
}

impl Layer for SplitLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        let axis = get_axis(self.axis, input.rank())?;
        let length = input.shape[axis];
        let sizes: Vec<usize> = match find_input(inputs, 1).map(Tensor::to_ints) {
            Some(sizes) => sizes.iter().map(|size| *size as usize).collect(),
            None => {
                let part = length.div_ceil(self.output_count.max(1));
                (0..self.output_count)
                    .map(|at| part.min(length.saturating_sub(at * part)))
                    .collect()
            }
        };

        let outer: usize = input.shape[..axis].iter().product();
        let inner: usize = input.shape[axis + 1..].iter().product();
        let mut outputs = Vec::with_capacity(sizes.len());
        let mut offset = 0;
        for size in sizes {
            let mut shape = input.shape.clone();
            shape[axis] = size;
            let positions = (0..outer).flat_map(|row| {
                let start = (row * length + offset) * inner;
                start..start + size * inner
            });
            outputs.push(input.pick(positions, shape));
            offset += size;
        }
        Ok(outputs)
    }
}

impl Layer for PadLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        let pads = get_input(inputs, 1)?.to_ints();
        let fill = find_input(inputs, 2)
            .and_then(|fill| fill.to_floats().first().copied())
            .unwrap_or(0.0);
        let rank = input.rank();
        if pads.len() != rank * 2 || pads.iter().any(|pad| *pad < 0) {
            return Err(Error::wrong_input(
                "only positive pads on every axis are supported",
            ));
        }

        let shape: Vec<usize> = (0..rank)
            .map(|axis| input.shape[axis] + (pads[axis] + pads[axis + rank]) as usize)
            .collect();
        let mut output = vec![fill; shape.iter().product()];
        let output_strides = get_strides(&shape);
        let first_position: usize = (0..rank)
            .map(|axis| pads[axis] as usize * output_strides[axis])
            .sum();
        for (value, at) in input
            .to_floats()
            .iter()
            .zip(get_positions(&input.shape, &output_strides))
        {
            output[at + first_position] = *value;
        }
        Ok(vec![Tensor::from_floats(shape, output)])
    }
}

impl Layer for TileLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        let repeats = get_input(inputs, 1)?.to_ints();
        if repeats.len() != input.rank() {
            return Err(Error::wrong_input("tile repeats do not match the input"));
        }
        let shape: Vec<usize> = input
            .shape
            .iter()
            .zip(&repeats)
            .map(|(size, repeat)| size * *repeat as usize)
            .collect();
        let input_strides = get_strides(&input.shape);
        let size: usize = shape.iter().product();
        let output_shape = shape.clone();
        let positions = (0..size).map(|at| {
            let mut rest = at;
            let mut position = 0;
            for axis in (0..shape.len()).rev() {
                let place = rest % shape[axis];
                rest /= shape[axis];
                position += (place % input.shape[axis]) * input_strides[axis];
            }
            position
        });
        Ok(vec![input.pick(positions, output_shape)])
    }
}
