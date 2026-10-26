use std::collections::HashMap;

use crate::{
    error::Error,
    layers::{
        Layer,
        constant::{get_constant, is_constant},
        get_layer,
    },
    model_file::GraphFile,
    tensor::{NumberKind, Tensor},
};

struct Step {
    kind: String,
    layer: Box<dyn Layer>,
    inputs: Vec<Option<usize>>,
    outputs: Vec<usize>,
    free_after: Vec<usize>,
}

pub(crate) struct Graph {
    steps: Vec<Step>,
    weights: Vec<Tensor>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    input_names: Vec<String>,
    input_kinds: Vec<Option<NumberKind>>,
    output_names: Vec<String>,
    value_count: usize,
}

struct Memory<'a> {
    weights: &'a [Tensor],
    values: Vec<Option<Tensor>>,
}

#[derive(Default)]
struct Names {
    slots: HashMap<String, usize>,
}

impl Names {
    fn add(&mut self, name: &str) -> usize {
        let next = self.slots.len();
        *self.slots.entry(name.to_string()).or_insert(next)
    }

    fn find(&self, name: &str) -> Result<usize, Error> {
        self.slots.get(name).copied().ok_or_else(|| {
            Error::wrong_file(format!("the value {} is used before it is made", name))
        })
    }
}

impl Graph {
    pub(crate) fn from_file(file: GraphFile) -> Result<Self, Error> {
        let mut names = Names::default();
        let mut weights = Vec::new();
        for (name, tensor) in file.weights {
            names.add(&name);
            weights.push(tensor);
        }
        for node in file.nodes.iter().filter(|node| is_constant(node)) {
            names.add(&node.outputs[0]);
            weights.push(get_constant(node)?);
        }

        let mut inputs = Vec::new();
        let mut input_names = Vec::new();
        let mut input_kinds = Vec::new();
        for input in file.inputs {
            if !names.slots.contains_key(&input.name) {
                inputs.push(names.add(&input.name));
                input_kinds.push(input.kind);
                input_names.push(input.name);
            }
        }

        let mut steps = Vec::new();
        for node in file.nodes.iter().filter(|node| !is_constant(node)) {
            let step_inputs = node
                .inputs
                .iter()
                .map(|name| {
                    if name.is_empty() {
                        Ok(None)
                    } else {
                        names.find(name).map(Some)
                    }
                })
                .collect::<Result<_, _>>()?;
            steps.push(Step {
                kind: node.kind.clone(),
                layer: get_layer(node)?,
                inputs: step_inputs,
                outputs: node.outputs.iter().map(|name| names.add(name)).collect(),
                free_after: Vec::new(),
            });
        }

        let output_names: Vec<String> =
            file.outputs.into_iter().map(|output| output.name).collect();
        let outputs = output_names
            .iter()
            .map(|name| names.find(name))
            .collect::<Result<Vec<_>, _>>()?;
        set_free_after(&mut steps, &outputs, weights.len());

        Ok(Self {
            steps,
            value_count: names.slots.len(),
            weights,
            inputs,
            outputs,
            input_names,
            input_kinds,
            output_names,
        })
    }

    pub(crate) fn get_input_names(&self) -> &[String] {
        &self.input_names
    }

    pub(crate) fn get_output_names(&self) -> &[String] {
        &self.output_names
    }

    pub(crate) fn get_step_count(&self) -> usize {
        self.steps.len()
    }

    pub(crate) fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, Error> {
        self.check_inputs(&inputs)?;
        let mut memory = Memory {
            weights: &self.weights,
            values: vec![None; self.value_count],
        };
        for (slot, tensor) in self.inputs.iter().zip(inputs) {
            memory.values[*slot] = Some(tensor);
        }

        for step in &self.steps {
            let step_inputs = step
                .inputs
                .iter()
                .map(|slot| slot.map(|slot| memory.get(slot)).transpose())
                .collect::<Result<Vec<_>, _>>()?;
            let step_outputs = step
                .layer
                .run(&step_inputs)
                .map_err(|error| error.in_layer(&step.kind))?;
            if step_outputs.len() < step.outputs.len() {
                return Err(Error::wrong_input(
                    "a layer gave fewer outputs than expected",
                ));
            }
            for (slot, tensor) in step.outputs.iter().zip(step_outputs) {
                memory.values[*slot] = Some(tensor);
            }
            for slot in &step.free_after {
                memory.values[*slot] = None;
            }
        }

        self.outputs.iter().map(|slot| memory.take(*slot)).collect()
    }
}

impl Graph {
    fn check_inputs(&self, inputs: &[Tensor]) -> Result<(), Error> {
        if inputs.len() != self.inputs.len() {
            let input_word = if self.inputs.len() == 1 {
                "input"
            } else {
                "inputs"
            };
            return Err(Error::wrong_input(format!(
                "the model expects {} {} ({}), got {}",
                self.inputs.len(),
                input_word,
                self.input_names.join(", "),
                inputs.len()
            )));
        }
        for (index, tensor) in inputs.iter().enumerate() {
            let name = &self.input_names[index];
            let needed: usize = tensor.shape.iter().product();
            if tensor.len() != needed {
                return Err(Error::wrong_input(format!(
                    "input {} ({}) has {} values but its shape {:?} needs {}",
                    index,
                    name,
                    tensor.len(),
                    tensor.shape,
                    needed
                )));
            }
            if let Some(kind) = self.input_kinds[index]
                && kind != tensor.get_kind()
            {
                return Err(Error::wrong_input(format!(
                    "input {} ({}) holds {} but the model expects {}",
                    index,
                    name,
                    tensor.get_kind().get_name(),
                    kind.get_name()
                )));
            }
        }
        Ok(())
    }
}

impl Memory<'_> {
    fn get(&self, slot: usize) -> Result<&Tensor, Error> {
        match self.weights.get(slot) {
            Some(weight) => Ok(weight),
            None => self.values[slot]
                .as_ref()
                .ok_or_else(|| Error::wrong_input("a value is missing")),
        }
    }

    fn take(&mut self, slot: usize) -> Result<Tensor, Error> {
        match self.weights.get(slot) {
            Some(weight) => Ok(weight.clone()),
            None => self.values[slot]
                .take()
                .ok_or_else(|| Error::wrong_input("an output is missing")),
        }
    }
}

fn set_free_after(steps: &mut [Step], outputs: &[usize], weight_count: usize) {
    let mut last_use = HashMap::new();
    for (step_index, step) in steps.iter().enumerate() {
        for slot in step.inputs.iter().flatten() {
            last_use.insert(*slot, step_index);
        }
    }
    for (slot, step_index) in last_use {
        if slot >= weight_count && !outputs.contains(&slot) {
            steps[step_index].free_after.push(slot);
        }
    }
}
