use std::{fmt, path::Path};

use crate::{error::Error, graph::Graph, model_file::ModelFile, tensor::Tensor};

pub struct Model {
    graph: Graph,
    settings: Vec<(String, String)>,
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
        Ok(Self {
            graph: Graph::from_file(file.graph)?,
            settings: file.settings,
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

    pub fn setting(&self, name: &str) -> Option<&str> {
        self.settings
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
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
