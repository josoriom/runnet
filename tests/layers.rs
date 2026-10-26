use runnet::{Model, Tensor};
use serde_json::Value;

const FOLDER: &str = "tests/fixtures";
const TOLERANCE: f64 = 1e-4;

struct Check {
    inputs: Vec<Tensor>,
    outputs: Vec<(String, Tensor)>,
}

fn read_check(name: &str) -> Check {
    let text = std::fs::read_to_string(format!("{}/{}.json", FOLDER, name)).expect("check file");
    let check: Value = serde_json::from_str(&text).expect("check json");
    let read_list = |key: &str| -> Vec<(String, Tensor)> {
        check[key]
            .as_array()
            .expect("tensor list")
            .iter()
            .map(|value| {
                (
                    value["name"].as_str().expect("name").to_string(),
                    read_tensor(value),
                )
            })
            .collect()
    };
    Check {
        inputs: read_list("inputs")
            .into_iter()
            .map(|(_, tensor)| tensor)
            .collect(),
        outputs: read_list("outputs"),
    }
}

fn read_tensor(value: &Value) -> Tensor {
    let shape: Vec<usize> = value["shape"]
        .as_array()
        .expect("shape")
        .iter()
        .map(|size| size.as_u64().expect("size") as usize)
        .collect();
    let numbers = value["values"].as_array().expect("values");
    match value["kind"].as_str() {
        Some("int") => Tensor::from_ints(
            shape,
            numbers
                .iter()
                .map(|number| number.as_i64().expect("int"))
                .collect::<Vec<_>>(),
        ),
        _ => Tensor::from_floats(
            shape,
            numbers
                .iter()
                .map(|number| number.as_f64().expect("float") as f32)
                .collect::<Vec<_>>(),
        ),
    }
}

fn get_values(tensor: &Tensor) -> Vec<f64> {
    match (tensor.floats(), tensor.ints()) {
        (Ok(values), _) => values.iter().map(|value| *value as f64).collect(),
        (_, Ok(values)) => values.iter().map(|value| *value as f64).collect(),
        _ => unreachable!("a tensor holds floats or integers"),
    }
}

fn assert_same(name: &str, found: &Tensor, expected: &Tensor) {
    assert_eq!(found.shape(), expected.shape(), "shape of {}", name);
    assert_eq!(
        found.ints().is_ok(),
        expected.ints().is_ok(),
        "number kind of {}",
        name
    );
    for (at, (found, expected)) in get_values(found)
        .iter()
        .zip(get_values(expected))
        .enumerate()
    {
        let allowed = TOLERANCE * expected.abs().max(1.0);
        assert!(
            (found - expected).abs() <= allowed,
            "{} value {} is {} but should be {}",
            name,
            at,
            found,
            expected
        );
    }
}

fn assert_case(name: &str) {
    let check = read_check(name);
    let model = Model::load(format!("{}/{}.onnx", FOLDER, name)).expect("model");
    let outputs = model.run(check.inputs).expect("run");
    assert_eq!(outputs.len(), check.outputs.len());
    for (found, (output_name, expected)) in outputs.iter().zip(&check.outputs) {
        assert_same(output_name, found, expected);
    }
}

#[test]
fn math_layers_match_onnx() {
    assert_case("math");
}

#[test]
fn arithmetic_layers_match_onnx() {
    assert_case("arithmetic");
}

#[test]
fn compare_layers_match_onnx() {
    assert_case("compare");
}

#[test]
fn conv_layers_match_onnx() {
    assert_case("conv");
}

#[test]
fn linear_layers_match_onnx() {
    assert_case("linear");
}

#[test]
fn pool_layers_match_onnx() {
    assert_case("pool");
}

#[test]
fn reduce_layers_match_onnx() {
    assert_case("reduce");
}

#[test]
fn normalize_layers_match_onnx() {
    assert_case("normalize");
}

#[test]
fn shape_layers_match_onnx() {
    assert_case("shape");
}

#[test]
fn select_layers_match_onnx() {
    assert_case("select");
}

#[test]
fn constant_layers_match_onnx() {
    assert_case("constant");
}

#[test]
fn every_weight_format_matches_onnx() {
    assert_case("weights");
}

#[test]
fn shared_values_and_listed_weights_match_onnx() {
    assert_case("graph");
}

#[test]
fn small_model_matches_onnx() {
    assert_case("small_model");
}
