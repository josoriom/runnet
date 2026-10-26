use runnet::{Error, Model, Tensor};

const FOLDER: &str = "tests/fixtures";

fn load_model(name: &str) -> Result<Model, Error> {
    Model::load(format!("{}/{}.onnx", FOLDER, name))
}

fn get_one_input() -> [Tensor; 1] {
    [Tensor::from_floats([1], [-2.0])]
}

#[test]
fn a_setting_is_given_as_written_in_the_file() {
    let model = load_model("settings").expect("model");

    assert_eq!(model.setting("points"), Some("64"));
    assert_eq!(model.setting("time_from"), Some(" 0.5 "));
    assert_eq!(model.setting("missing"), None);
}

#[test]
fn inputs_and_outputs_are_named_in_order() {
    let model = load_model("small_model").expect("model");

    assert_eq!(
        model.input_names(),
        ["target_eics", "targets", "context_eics"]
    );
    assert_eq!(model.output_names(), ["predicted_rt", "uncertainty"]);
}

#[test]
fn weights_listed_as_inputs_are_not_inputs() {
    let model = load_model("graph").expect("model");

    assert_eq!(model.input_names(), ["x"]);
}

#[test]
fn a_model_shows_its_names_when_debugged() {
    let model = load_model("settings").expect("model");

    assert_eq!(
        format!("{:?}", model),
        r#"Model { inputs: ["x"], outputs: ["y"], layers: 1 }"#
    );
}

#[test]
fn a_model_runs_from_bytes() {
    let bytes = std::fs::read(format!("{}/settings.onnx", FOLDER)).expect("model file");
    let model = Model::from_bytes(&bytes).expect("model");

    let outputs = model.run(get_one_input()).expect("run");

    assert_eq!(outputs[0].floats(), Ok(&[0.0][..]));
}

#[test]
fn outputs_give_their_values_without_a_copy() {
    let model = load_model("settings").expect("model");

    let output = model.run(get_one_input()).expect("run").swap_remove(0);

    assert_eq!(output.into_floats(), Ok(vec![0.0]));
}

#[test]
fn asking_for_the_wrong_number_type_is_refused() {
    let floats = Tensor::from_floats([1], [1.5]);
    let ints = Tensor::from_ints([1], [2]);

    assert_eq!(ints.ints(), Ok(&[2][..]));
    assert!(matches!(
        floats.ints(),
        Err(Error::WrongType {
            expected: "ints",
            found: "floats",
            ..
        })
    ));
    assert!(matches!(
        ints.floats(),
        Err(Error::WrongType {
            expected: "floats",
            found: "ints",
            ..
        })
    ));
    assert!(matches!(floats.into_ints(), Err(Error::WrongType { .. })));
    assert!(matches!(ints.into_floats(), Err(Error::WrongType { .. })));
}

#[test]
fn an_old_opset_is_refused() {
    let result = load_model("old_opset");

    assert!(matches!(
        result,
        Err(Error::OpsetTooOld {
            opset: 12,
            oldest: 13,
            ..
        })
    ));
}

#[test]
fn errors_explain_themselves() {
    let message = load_model("old_opset").expect_err("error").to_string();

    assert_eq!(
        message,
        "the model uses opset 12, runnet needs opset 13 or newer"
    );
}

#[test]
fn an_unknown_layer_is_named() {
    let result = load_model("unknown_layer");

    assert!(matches!(result, Err(Error::LayerNotSupported { layer, .. }) if layer == "Unknown"));
}

#[test]
fn a_value_used_before_it_is_made_is_refused() {
    let result = load_model("wrong_order");

    assert!(matches!(result, Err(Error::WrongFile { .. })));
}

#[test]
fn weights_in_a_separate_file_are_refused() {
    let result = load_model("outside_weights");

    assert!(matches!(result, Err(Error::WrongFile { .. })));
}

#[test]
fn a_cut_file_is_refused() {
    let bytes = std::fs::read(format!("{}/small_model.onnx", FOLDER)).expect("model file");

    let result = Model::from_bytes(&bytes[..bytes.len() / 2]);

    assert!(matches!(result, Err(Error::WrongFile { .. })));
}

#[test]
fn a_missing_file_is_named() {
    let result = load_model("missing");

    assert!(
        matches!(result, Err(Error::FileNotRead { path, .. }) if path.ends_with("missing.onnx"))
    );
}

#[test]
fn the_wrong_number_of_inputs_is_refused() {
    let model = load_model("settings").expect("model");

    let result = model.run([]);

    assert!(matches!(
        result,
        Err(Error::WrongInput { layer: None, reason, .. }) if reason == "the model expects 1 input (x), got 0"
    ));
    let three_inputs = load_model("small_model").expect("model").run([]);
    assert!(matches!(
        three_inputs,
        Err(Error::WrongInput { layer: None, reason, .. })
            if reason == "the model expects 3 inputs (target_eics, targets, context_eics), got 0"
    ));
}

#[test]
fn an_input_whose_values_do_not_fit_its_shape_is_refused() {
    let model = load_model("settings").expect("model");

    let result = model.run([Tensor::from_floats([2], [1.0])]);

    assert!(matches!(
        result,
        Err(Error::WrongInput { layer: None, reason, .. }) if reason == "input 0 (x) has 1 values but its shape [2] needs 2"
    ));
}

#[test]
fn an_input_of_the_wrong_number_type_is_refused() {
    let model = load_model("settings").expect("model");

    let result = model.run([Tensor::from_ints([1], [3])]);

    assert!(matches!(
        result,
        Err(Error::WrongInput { layer: None, reason, .. }) if reason == "input 0 (x) holds ints but the model expects floats"
    ));
}

#[test]
fn a_failing_layer_is_named_in_the_error() {
    let model = load_model("bad_reshape").expect("model");

    let result = model.run(get_one_input());

    assert!(
        matches!(result, Err(Error::WrongInput { layer: Some(layer), .. }) if layer == "Reshape")
    );
}

#[test]
fn public_types_can_be_shared_between_threads() {
    fn assert_thread_safe<T: Send + Sync>() {}

    assert_thread_safe::<Model>();
    assert_thread_safe::<Tensor>();
    assert_thread_safe::<Error>();
}
