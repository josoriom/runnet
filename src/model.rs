use std::{fmt, path::Path};

use crate::{error::Error, graph::Graph, model_file::ModelFile, tensor::Tensor};

pub struct Model {
    graph: Graph,
    keys: Vec<String>,
    values: Vec<String>,
}

impl Model {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|error| Error::FileNotRead {
            path: path.display().to_string(),
            reason: error.to_string(),
        })?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let file = ModelFile::read(bytes)?;
        let (keys, values) = file.settings.into_iter().unzip();
        Ok(Self {
            graph: Graph::from_file(file.graph)?,
            keys,
            values,
        })
    }

    pub fn run(&self, inputs: impl IntoIterator<Item = Tensor>) -> Result<Vec<Tensor>, Error> {
        self.graph.run(inputs.into_iter().collect())
    }

    pub fn input_names(&self) -> &[String] {
        self.graph.get_input_names()
    }

    pub fn output_names(&self) -> &[String] {
        self.graph.get_output_names()
    }

    pub fn get_keys(&self) -> &[String] {
        &self.keys
    }

    pub fn get_values(&self) -> &[String] {
        &self.values
    }

    pub fn get_value(&self, key: &str) -> Option<&str> {
        self.keys
            .iter()
            .position(|name| name == key)
            .map(|index| self.values[index].as_str())
    }
}

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Model")
            .field("inputs", &self.input_names())
            .field("outputs", &self.output_names())
            .field("layers", &self.graph.get_step_count())
            .finish()
    }
}
