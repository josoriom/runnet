use crate::{
    error::{Error, OLDEST_OPSET},
    proto_reader::{Field, ProtoReader},
    tensor::{NumberKind, Tensor},
};

const FLOAT: i64 = 1;
const UINT8: i64 = 2;
const INT8: i64 = 3;
const UINT16: i64 = 4;
const INT16: i64 = 5;
const INT32: i64 = 6;
const INT64: i64 = 7;
const BOOL: i64 = 9;
const FLOAT16: i64 = 10;
const DOUBLE: i64 = 11;
const UINT32: i64 = 12;
const UINT64: i64 = 13;
const BFLOAT16: i64 = 16;

pub(crate) struct ModelFile {
    pub graph: GraphFile,
    pub settings: Vec<(String, String)>,
}

#[derive(Default)]
pub(crate) struct GraphFile {
    pub nodes: Vec<NodeFile>,
    pub weights: Vec<(String, Tensor)>,
    pub inputs: Vec<GraphValue>,
    pub outputs: Vec<GraphValue>,
}

pub(crate) struct GraphValue {
    pub name: String,
    pub kind: Option<NumberKind>,
}

#[derive(Default)]
pub(crate) struct NodeFile {
    pub kind: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: Vec<Attribute>,
}

pub(crate) struct Attribute {
    pub name: String,
    pub value: AttributeValue,
}

pub(crate) enum AttributeValue {
    Float(f32),
    Int(i64),
    Text(String),
    Tensor(Tensor),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Other,
}

impl ModelFile {
    pub(crate) fn read(bytes: &[u8]) -> Result<Self, Error> {
        let mut graph = None;
        let mut settings = Vec::new();
        let mut opset = None;
        let mut reader = ProtoReader::new(bytes);
        while !reader.is_done() {
            let field = reader.read_field()?;
            match field.number {
                7 => graph = Some(read_graph(field.as_bytes()?)?),
                8 => opset = read_opset(field.as_bytes()?)?.or(opset),
                14 => settings.push(read_setting(field.as_bytes()?)?),
                _ => {}
            }
        }
        let opset = opset.ok_or_else(|| Error::wrong_file("the model has no opset"))?;
        if opset < OLDEST_OPSET {
            return Err(Error::opset_too_old(opset));
        }
        let graph = graph.ok_or_else(|| Error::wrong_file("the model has no graph"))?;
        Ok(Self { graph, settings })
    }
}

impl NodeFile {
    pub(crate) fn find(&self, name: &str) -> Option<&AttributeValue> {
        self.attributes
            .iter()
            .find(|attribute| attribute.name == name)
            .map(|attribute| &attribute.value)
    }

    pub(crate) fn get_int(&self, name: &str, default: i64) -> i64 {
        match self.find(name) {
            Some(AttributeValue::Int(value)) => *value,
            _ => default,
        }
    }

    pub(crate) fn get_float(&self, name: &str, default: f32) -> f32 {
        match self.find(name) {
            Some(AttributeValue::Float(value)) => *value,
            _ => default,
        }
    }

    pub(crate) fn get_ints(&self, name: &str) -> Option<Vec<i64>> {
        match self.find(name) {
            Some(AttributeValue::Ints(values)) => Some(values.clone()),
            _ => None,
        }
    }

    pub(crate) fn get_text(&self, name: &str) -> Option<&str> {
        match self.find(name) {
            Some(AttributeValue::Text(value)) => Some(value),
            _ => None,
        }
    }
}

fn read_opset(bytes: &[u8]) -> Result<Option<i64>, Error> {
    let mut domain = String::new();
    let mut version = 0;
    let mut reader = ProtoReader::new(bytes);
    while !reader.is_done() {
        let field = reader.read_field()?;
        match field.number {
            1 => domain = field.as_text()?,
            2 => version = field.as_int()?,
            _ => {}
        }
    }
    let is_main_domain = domain.is_empty() || domain == "ai.onnx";
    Ok(is_main_domain.then_some(version))
}

fn read_setting(bytes: &[u8]) -> Result<(String, String), Error> {
    let mut key = String::new();
    let mut value = String::new();
    let mut reader = ProtoReader::new(bytes);
    while !reader.is_done() {
        let field = reader.read_field()?;
        match field.number {
            1 => key = field.as_text()?,
            2 => value = field.as_text()?,
            _ => {}
        }
    }
    Ok((key, value))
}

fn read_graph(bytes: &[u8]) -> Result<GraphFile, Error> {
    let mut graph = GraphFile::default();
    let mut reader = ProtoReader::new(bytes);
    while !reader.is_done() {
        let field = reader.read_field()?;
        match field.number {
            1 => graph.nodes.push(read_node(field.as_bytes()?)?),
            5 => graph.weights.push(read_tensor(field.as_bytes()?)?),
            11 => graph.inputs.push(read_graph_value(field.as_bytes()?)?),
            12 => graph.outputs.push(read_graph_value(field.as_bytes()?)?),
            _ => {}
        }
    }
    Ok(graph)
}

fn read_graph_value(bytes: &[u8]) -> Result<GraphValue, Error> {
    let mut name = None;
    let mut kind = None;
    let mut reader = ProtoReader::new(bytes);
    while !reader.is_done() {
        let field = reader.read_field()?;
        match field.number {
            1 => name = Some(field.as_text()?),
            2 => kind = read_value_kind(field.as_bytes()?)?,
            _ => {}
        }
    }
    let name = name.ok_or_else(|| Error::wrong_file("a graph input or output has no name"))?;
    Ok(GraphValue { name, kind })
}

fn read_value_kind(bytes: &[u8]) -> Result<Option<NumberKind>, Error> {
    let mut reader = ProtoReader::new(bytes);
    while !reader.is_done() {
        let field = reader.read_field()?;
        if field.number == 1 {
            return read_element_kind(field.as_bytes()?);
        }
    }
    Ok(None)
}

fn read_element_kind(bytes: &[u8]) -> Result<Option<NumberKind>, Error> {
    let mut reader = ProtoReader::new(bytes);
    while !reader.is_done() {
        let field = reader.read_field()?;
        if field.number == 1 {
            return Ok(get_number_kind(field.as_int()?));
        }
    }
    Ok(None)
}

pub(crate) fn get_number_kind(element_type: i64) -> Option<NumberKind> {
    match element_type {
        FLOAT | FLOAT16 | DOUBLE | BFLOAT16 => Some(NumberKind::Float),
        UINT8 | INT8 | UINT16 | INT16 | INT32 | INT64 | BOOL | UINT32 | UINT64 => {
            Some(NumberKind::Int)
        }
        _ => None,
    }
}

fn read_node(bytes: &[u8]) -> Result<NodeFile, Error> {
    let mut node = NodeFile::default();
    let mut reader = ProtoReader::new(bytes);
    while !reader.is_done() {
        let field = reader.read_field()?;
        match field.number {
            1 => node.inputs.push(field.as_text()?),
            2 => node.outputs.push(field.as_text()?),
            4 => node.kind = field.as_text()?,
            5 => node.attributes.push(read_attribute(field.as_bytes()?)?),
            _ => {}
        }
    }
    Ok(node)
}

fn read_attribute(bytes: &[u8]) -> Result<Attribute, Error> {
    let mut name = String::new();
    let mut value = AttributeValue::Other;
    let mut floats = Vec::new();
    let mut ints = Vec::new();
    let mut reader = ProtoReader::new(bytes);
    while !reader.is_done() {
        let field = reader.read_field()?;
        match field.number {
            1 => name = field.as_text()?,
            2 => value = AttributeValue::Float(field.as_float()?),
            3 => value = AttributeValue::Int(field.as_int()?),
            4 => value = AttributeValue::Text(field.as_text()?),
            5 => value = AttributeValue::Tensor(read_tensor(field.as_bytes()?)?.1),
            7 => field.add_floats_to(&mut floats)?,
            8 => field.add_ints_to(&mut ints)?,
            _ => {}
        }
    }
    if !floats.is_empty() {
        value = AttributeValue::Floats(floats);
    }
    if !ints.is_empty() {
        value = AttributeValue::Ints(ints);
    }
    Ok(Attribute { name, value })
}

#[derive(Default)]
struct TensorParts<'a> {
    name: String,
    shape: Vec<i64>,
    kind: i64,
    raw_bytes: Option<&'a [u8]>,
    floats: Vec<f32>,
    doubles: Vec<f64>,
    ints: Vec<i64>,
    is_external: bool,
}

fn read_tensor(bytes: &[u8]) -> Result<(String, Tensor), Error> {
    let mut parts = TensorParts::default();
    let mut reader = ProtoReader::new(bytes);
    while !reader.is_done() {
        let field = reader.read_field()?;
        read_tensor_field(&field, &mut parts)?;
    }
    if parts.is_external {
        return Err(Error::wrong_file(
            "weights in separate files are not supported",
        ));
    }

    let shape: Vec<usize> = parts.shape.iter().map(|size| *size as usize).collect();
    let tensor = match parts.raw_bytes {
        Some(raw_bytes) => get_tensor_from_bytes(parts.kind, shape, raw_bytes)?,
        None => get_tensor_from_lists(parts.kind, shape, parts.floats, parts.doubles, parts.ints)?,
    };
    if tensor.len() != tensor.shape.iter().product::<usize>() {
        return Err(Error::wrong_file("a weight has the wrong size"));
    }
    Ok((parts.name, tensor))
}

fn read_tensor_field<'a>(field: &Field<'a>, parts: &mut TensorParts<'a>) -> Result<(), Error> {
    match field.number {
        1 => field.add_ints_to(&mut parts.shape)?,
        2 => parts.kind = field.as_int()?,
        4 => field.add_floats_to(&mut parts.floats)?,
        5 | 7 | 11 => field.add_ints_to(&mut parts.ints)?,
        8 => parts.name = field.as_text()?,
        9 => parts.raw_bytes = Some(field.as_bytes()?),
        10 => field.add_doubles_to(&mut parts.doubles)?,
        13 => parts.is_external = true,
        _ => {}
    }
    Ok(())
}

fn get_tensor_from_bytes(kind: i64, shape: Vec<usize>, bytes: &[u8]) -> Result<Tensor, Error> {
    let tensor = match kind {
        FLOAT => Tensor::from_floats(shape, read_each(bytes, f32::from_le_bytes)),
        DOUBLE => Tensor::from_floats(shape, read_each(bytes, |b| f64::from_le_bytes(b) as f32)),
        FLOAT16 => {
            Tensor::from_floats(shape, read_each(bytes, |b| get_half(u16::from_le_bytes(b))))
        }
        INT64 => Tensor::from_ints(shape, read_each(bytes, i64::from_le_bytes)),
        UINT64 => Tensor::from_ints(shape, read_each(bytes, |b| u64::from_le_bytes(b) as i64)),
        INT32 => Tensor::from_ints(shape, read_each(bytes, |b| i32::from_le_bytes(b) as i64)),
        UINT32 => Tensor::from_ints(shape, read_each(bytes, |b| u32::from_le_bytes(b) as i64)),
        INT16 => Tensor::from_ints(shape, read_each(bytes, |b| i16::from_le_bytes(b) as i64)),
        UINT16 => Tensor::from_ints(shape, read_each(bytes, |b| u16::from_le_bytes(b) as i64)),
        INT8 => Tensor::from_ints(shape, read_each(bytes, |b: [u8; 1]| b[0] as i8 as i64)),
        UINT8 | BOOL => Tensor::from_ints(shape, read_each(bytes, |b: [u8; 1]| b[0] as i64)),
        _ => return Err(Error::wrong_file("a weight has an unsupported type")),
    };
    Ok(tensor)
}

fn get_tensor_from_lists(
    kind: i64,
    shape: Vec<usize>,
    floats: Vec<f32>,
    doubles: Vec<f64>,
    ints: Vec<i64>,
) -> Result<Tensor, Error> {
    let tensor = match kind {
        FLOAT => Tensor::from_floats(shape, floats),
        DOUBLE => Tensor::from_floats(
            shape,
            doubles
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>(),
        ),
        FLOAT16 => Tensor::from_floats(
            shape,
            ints.iter()
                .map(|bits| get_half(*bits as u16))
                .collect::<Vec<_>>(),
        ),
        INT64 | UINT64 | INT32 | UINT32 | INT16 | UINT16 | INT8 | UINT8 | BOOL => {
            Tensor::from_ints(shape, ints)
        }
        _ => return Err(Error::wrong_file("a weight has an unsupported type")),
    };
    Ok(tensor)
}

fn read_each<const SIZE: usize, T>(bytes: &[u8], read: impl Fn([u8; SIZE]) -> T) -> Vec<T> {
    let (chunks, _) = bytes.as_chunks::<SIZE>();
    chunks.iter().map(|chunk| read(*chunk)).collect()
}

pub(crate) fn get_half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let fraction = (bits & 0x3ff) as f32;
    match exponent {
        0 => sign * fraction * 2_f32.powi(-24),
        31 if fraction == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + fraction / 1024.0) * 2_f32.powi(exponent - 15),
    }
}
