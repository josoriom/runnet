use std::fmt;

pub(crate) const OLDEST_OPSET: i64 = 13;

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Error {
    #[non_exhaustive]
    FileNotRead { path: String, reason: String },
    #[non_exhaustive]
    WrongFile { reason: String },
    #[non_exhaustive]
    OpsetTooOld { opset: i64, oldest: i64 },
    #[non_exhaustive]
    LayerNotSupported { layer: String },
    #[non_exhaustive]
    WrongInput {
        layer: Option<String>,
        reason: String,
    },
    #[non_exhaustive]
    WrongType {
        expected: &'static str,
        found: &'static str,
    },
}

impl Error {
    pub(crate) fn wrong_file(reason: impl Into<String>) -> Self {
        Self::WrongFile {
            reason: reason.into(),
        }
    }

    pub(crate) fn wrong_input(reason: impl Into<String>) -> Self {
        Self::WrongInput {
            layer: None,
            reason: reason.into(),
        }
    }

    pub(crate) fn wrong_type(expected: &'static str, found: &'static str) -> Self {
        Self::WrongType { expected, found }
    }

    pub(crate) fn opset_too_old(opset: i64) -> Self {
        Self::OpsetTooOld {
            opset,
            oldest: OLDEST_OPSET,
        }
    }

    pub(crate) fn layer_not_supported(layer: &str) -> Self {
        Self::LayerNotSupported {
            layer: layer.to_string(),
        }
    }

    pub(crate) fn in_layer(self, kind: &str) -> Self {
        let reason = match self {
            Self::WrongInput {
                layer: None,
                reason,
            } => reason,
            Self::WrongType { expected, found } => {
                format!("expected {}, found {}", expected, found)
            }
            other => return other,
        };
        Self::WrongInput {
            layer: Some(kind.to_string()),
            reason,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::FileNotRead { path, reason } => write!(f, "cannot read {}: {}", path, reason),
            Error::WrongFile { reason } => {
                write!(f, "the file is not a valid onnx model: {}", reason)
            }
            Error::OpsetTooOld { opset, oldest } => write!(
                f,
                "the model uses opset {}, runnet needs opset {} or newer",
                opset, oldest
            ),
            Error::LayerNotSupported { layer } => write!(f, "the layer {} is not supported", layer),
            Error::WrongInput {
                layer: Some(layer),
                reason,
            } => write!(f, "{}: {}", layer, reason),
            Error::WrongInput {
                layer: None,
                reason,
            } => write!(f, "the input is not valid: {}", reason),
            Error::WrongType { expected, found } => {
                write!(f, "the tensor holds {}, not {}", found, expected)
            }
        }
    }
}

impl std::error::Error for Error {}
