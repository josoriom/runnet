import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

OUTPUT_FOLDER = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
RANDOM = np.random.default_rng(7)
OPSET = 17
GELU_OPSET = 20


def get_weight(name, values):
    return numpy_helper.from_array(np.asarray(values), name)


def get_random(name, *shape, scale=0.5):
    return get_weight(name, (RANDOM.standard_normal(shape) * scale).astype(np.float32))


def get_ints(name, values):
    return get_weight(name, np.asarray(values, dtype=np.int64))


def get_float(name, value):
    return get_weight(name, np.asarray(value, dtype=np.float32))


def get_kind(values):
    return TensorProto.INT64 if np.issubdtype(values.dtype, np.integer) else TensorProto.FLOAT


def build_model(name, nodes, inputs, outputs, weights, opset=OPSET, settings=None, listed_weights=()):
    graph_inputs = [helper.make_tensor_value_info(key, get_kind(value), None) for key, value in inputs.items()]
    graph_inputs += [helper.make_tensor_value_info(key, TensorProto.FLOAT, None) for key in listed_weights]
    graph_outputs = [helper.make_tensor_value_info(output, TensorProto.UNDEFINED, None) for output in outputs]
    graph = helper.make_graph(nodes, name, graph_inputs, graph_outputs, weights)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    for key, value in (settings or {}).items():
        model.metadata_props.add(key=key, value=value)
    return model


def save_model(name, model):
    onnx.save(model, OUTPUT_FOLDER / f"{name}.onnx")


def save_case(name, nodes, inputs, outputs, weights, opset=OPSET, listed_weights=()):
    model = build_model(name, nodes, inputs, outputs, weights, opset, listed_weights=listed_weights)
    save_model(name, model)
    output_values = ReferenceEvaluator(model).run(None, inputs)
    check = {
        "inputs": [get_listed(key, value) for key, value in inputs.items()],
        "outputs": [get_listed(key, value) for key, value in zip(outputs, output_values)],
    }
    (OUTPUT_FOLDER / f"{name}.json").write_text(json.dumps(check))


def get_listed(name, values):
    values = np.asarray(values)
    is_int = np.issubdtype(values.dtype, np.integer) or values.dtype == np.bool_
    return {
        "name": name,
        "kind": "int" if is_int else "float",
        "shape": list(values.shape),
        "values": values.reshape(-1).astype(np.int64 if is_int else np.float32).tolist(),
    }


def make_math_case():
    nodes = [
        helper.make_node("Relu", ["x"], ["relu"]),
        helper.make_node("LeakyRelu", ["x"], ["leaky_relu"], alpha=0.2),
        helper.make_node("Elu", ["x"], ["elu"], alpha=0.5),
        helper.make_node("Sigmoid", ["x"], ["sigmoid"]),
        helper.make_node("HardSigmoid", ["x"], ["hard_sigmoid"], alpha=0.3, beta=0.4),
        helper.make_node("Tanh", ["x"], ["tanh"]),
        helper.make_node("Softplus", ["x"], ["softplus"]),
        helper.make_node("Erf", ["x"], ["erf"]),
        helper.make_node("Gelu", ["x"], ["gelu"]),
        helper.make_node("Gelu", ["x"], ["gelu_tanh"], approximate="tanh"),
        helper.make_node("Exp", ["x"], ["exp"]),
        helper.make_node("Abs", ["x"], ["abs"]),
        helper.make_node("Add", ["abs", "one"], ["abs_plus_one"]),
        helper.make_node("Log", ["abs_plus_one"], ["log"]),
        helper.make_node("Sqrt", ["abs"], ["sqrt"]),
        helper.make_node("Reciprocal", ["abs_plus_one"], ["reciprocal"]),
        helper.make_node("Neg", ["x"], ["neg"]),
        helper.make_node("Floor", ["x"], ["floor"]),
        helper.make_node("Ceil", ["x"], ["ceil"]),
        helper.make_node("Round", ["halves"], ["round"]),
        helper.make_node("Sin", ["x"], ["sin"]),
        helper.make_node("Cos", ["x"], ["cos"]),
        helper.make_node("Clip", ["x", "low", "high"], ["clip"]),
        helper.make_node("Clip", ["x", "low"], ["clip_low"]),
        helper.make_node("Clip", ["x", "", "high"], ["clip_high"]),
        helper.make_node("Abs", ["counts"], ["abs_int"]),
        helper.make_node("Neg", ["counts"], ["neg_int"]),
    ]
    outputs = [node.output[0] for node in nodes if node.output[0] != "abs_plus_one"]
    inputs = {
        "x": np.linspace(-3, 3, 15, dtype=np.float32).reshape(3, 5),
        "halves": np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32),
    }
    weights = [get_float("one", 1.0), get_float("low", -0.5), get_float("high", 0.8), get_ints("counts", [-3, 0, 4])]
    save_case("math", nodes, inputs, outputs, weights, opset=GELU_OPSET)


def make_arithmetic_case():
    nodes = [
        helper.make_node("Add", ["x", "x"], ["same_shape"]),
        helper.make_node("Sub", ["x", "row"], ["per_row"]),
        helper.make_node("Mul", ["x", "channel"], ["per_channel"]),
        helper.make_node("Sub", ["x", "two"], ["scalar_right"]),
        helper.make_node("Div", ["two", "x"], ["scalar_left"]),
        helper.make_node("Add", ["y", "column"], ["both_broadcast"]),
        helper.make_node("Add", ["x", "y"], ["split_broadcast"]),
        helper.make_node("Pow", ["x", "two"], ["square"]),
        helper.make_node("Abs", ["x"], ["abs"]),
        helper.make_node("Pow", ["abs", "half"], ["power_half"]),
        helper.make_node("Pow", ["abs", "row"], ["power_row"]),
        helper.make_node("Pow", ["x", "three_int"], ["power_int_exponent"]),
        helper.make_node("Max", ["x", "row"], ["max"]),
        helper.make_node("Min", ["x", "channel"], ["min"]),
        helper.make_node("Sum", ["x", "row", "channel"], ["sum_three"]),
        helper.make_node("Sum", ["x"], ["sum_one"]),
        helper.make_node("Add", ["left_int", "right_int"], ["add_int"]),
        helper.make_node("Sub", ["left_int", "right_int"], ["sub_int"]),
        helper.make_node("Mul", ["left_int", "right_int"], ["mul_int"]),
        helper.make_node("Div", ["left_int", "right_int"], ["div_int"]),
        helper.make_node("Max", ["left_int", "right_int"], ["max_int"]),
        helper.make_node("Min", ["left_int", "right_int"], ["min_int"]),
        helper.make_node("Pow", ["left_int", "two_int"], ["pow_int"]),
    ]
    outputs = [node.output[0] for node in nodes if node.output[0] != "abs"]
    inputs = {
        "x": RANDOM.standard_normal((2, 3, 4)).astype(np.float32),
        "y": RANDOM.standard_normal((2, 1, 4)).astype(np.float32),
    }
    weights = [
        get_weight("row", np.abs(RANDOM.standard_normal(4)).astype(np.float32) + 0.5),
        get_random("channel", 1, 3, 1),
        get_random("column", 3, 1),
        get_float("two", 2.0),
        get_float("half", 0.5),
        get_ints("three_int", 3),
        get_ints("two_int", 2),
        get_ints("left_int", [[7, 9, 12], [5, 20, 3]]),
        get_ints("right_int", [2, 4, 3]),
    ]
    save_case("arithmetic", nodes, inputs, outputs, weights)


def make_compare_case():
    nodes = [
        helper.make_node("Equal", ["a", "b"], ["equal"]),
        helper.make_node("Less", ["a", "b"], ["less"]),
        helper.make_node("LessOrEqual", ["a", "b"], ["less_or_equal"]),
        helper.make_node("Greater", ["a", "b"], ["greater"]),
        helper.make_node("GreaterOrEqual", ["a", "b"], ["greater_or_equal"]),
        helper.make_node("And", ["less_or_equal", "greater_or_equal"], ["and"]),
        helper.make_node("Or", ["less", "greater"], ["or"]),
        helper.make_node("Not", ["equal"], ["not"]),
        helper.make_node("Where", ["less", "a", "b"], ["where_float"]),
        helper.make_node("Less", ["left_int", "right_int"], ["less_int"]),
        helper.make_node("Where", ["less_int", "left_int", "right_int"], ["where_int"]),
    ]
    inputs = {
        "a": np.round(RANDOM.standard_normal((3, 4)) * 1.5).astype(np.float32),
        "b": np.round(RANDOM.standard_normal(4) * 1.5).astype(np.float32),
    }
    weights = [get_ints("left_int", [[1, 5, 3], [4, 2, 6]]), get_ints("right_int", [3, 3, 3])]
    save_case("compare", nodes, inputs, [node.output[0] for node in nodes], weights)


def make_conv_case():
    nodes = [
        helper.make_node("Conv", ["x", "weight", "bias"], ["dilated"], dilations=[2], pads=[2, 2]),
        helper.make_node("Conv", ["x", "weight"], ["same_upper"], strides=[2], auto_pad="SAME_UPPER"),
        helper.make_node("Conv", ["x", "even_weight"], ["same_lower"], auto_pad="SAME_LOWER"),
        helper.make_node("Conv", ["x", "weight"], ["valid"], strides=[3], auto_pad="VALID"),
        helper.make_node("Conv", ["image", "group_weight"], ["grouped_2d"], pads=[1, 0, 1, 2], dilations=[1, 2], group=2),
        helper.make_node("Conv", ["image", "square_weight"], ["strided_2d"], strides=[2, 1]),
        helper.make_node("Conv", ["wide", "wide_weight", "wide_bias"], ["kernel_blocks"], pads=[1, 1]),
    ]
    inputs = {
        "x": RANDOM.standard_normal((2, 3, 11)).astype(np.float32),
        "image": RANDOM.standard_normal((1, 4, 5, 6)).astype(np.float32),
        "wide": RANDOM.standard_normal((1, 5, 21)).astype(np.float32),
    }
    weights = [
        get_random("weight", 4, 3, 3), get_random("bias", 4), get_random("even_weight", 2, 3, 4),
        get_random("group_weight", 6, 2, 3, 2), get_random("square_weight", 3, 4, 2, 2),
        get_random("wide_weight", 11, 5, 3), get_random("wide_bias", 11),
    ]
    save_case("conv", nodes, inputs, [node.output[0] for node in nodes], weights)


def make_linear_case():
    nodes = [
        helper.make_node("MatMul", ["left", "right"], ["matmul_blocks"]),
        helper.make_node("MatMul", ["batch_left", "batch_right"], ["matmul_batches"]),
        helper.make_node("MatMul", ["vector", "matrix"], ["vector_matrix"]),
        helper.make_node("MatMul", ["flipped_matrix", "vector"], ["matrix_vector"]),
        helper.make_node("MatMul", ["vector", "vector"], ["vector_vector"]),
        helper.make_node("Gemm", ["small", "flipped_dense", "dense_bias"], ["gemm_flip_right"], transB=1, alpha=0.5, beta=2.0),
        helper.make_node("Gemm", ["flipped_small", "dense", "scalar_bias"], ["gemm_flip_left"], transA=1),
        helper.make_node("Gemm", ["small", "dense", "matrix_bias"], ["gemm_matrix_bias"]),
        helper.make_node("Gemm", ["small", "dense"], ["gemm_no_bias"]),
    ]
    inputs = {
        "left": RANDOM.standard_normal((11, 13)).astype(np.float32),
        "batch_left": RANDOM.standard_normal((2, 1, 3, 4)).astype(np.float32),
        "vector": RANDOM.standard_normal(4).astype(np.float32),
        "small": RANDOM.standard_normal((3, 4)).astype(np.float32),
    }
    weights = [
        get_random("right", 13, 21), get_random("batch_right", 5, 4, 2), get_random("matrix", 4, 5),
        get_random("flipped_matrix", 5, 4), get_random("flipped_dense", 5, 4), get_random("dense_bias", 5),
        get_random("flipped_small", 4, 3), get_random("dense", 4, 5), get_random("scalar_bias", 1),
        get_random("matrix_bias", 3, 5),
    ]
    save_case("linear", nodes, inputs, [node.output[0] for node in nodes], weights)


def make_pool_case():
    nodes = [
        helper.make_node("MaxPool", ["x"], ["max_pool"], kernel_shape=[3], strides=[2], pads=[1, 1]),
        helper.make_node("MaxPool", ["x"], ["max_pool_round_up"], kernel_shape=[2], strides=[2], ceil_mode=1),
        helper.make_node("MaxPool", ["x"], ["max_pool_dilated"], kernel_shape=[2], dilations=[3]),
        helper.make_node("AveragePool", ["x"], ["average_without_pads"], kernel_shape=[3], pads=[1, 1]),
        helper.make_node("AveragePool", ["x"], ["average_with_pads"], kernel_shape=[3], pads=[1, 1], count_include_pad=1),
        helper.make_node("GlobalAveragePool", ["x"], ["global_average"]),
        helper.make_node("GlobalMaxPool", ["image"], ["global_max"]),
        helper.make_node("MaxPool", ["image"], ["max_pool_2d"], kernel_shape=[2, 3], strides=[1, 2]),
        helper.make_node("AveragePool", ["image"], ["average_2d"], kernel_shape=[2, 2], auto_pad="SAME_UPPER"),
    ]
    inputs = {
        "x": RANDOM.standard_normal((1, 2, 9)).astype(np.float32),
        "image": RANDOM.standard_normal((1, 2, 5, 6)).astype(np.float32),
    }
    save_case("pool", nodes, inputs, [node.output[0] for node in nodes], [])


def make_reduce_case():
    nodes = [
        helper.make_node("ReduceMean", ["x"], ["mean_middle"], axes=[1]),
        helper.make_node("ReduceMean", ["x"], ["mean_first"], axes=[0], keepdims=0),
        helper.make_node("ReduceMax", ["x"], ["max_last"], axes=[-1], keepdims=0),
        helper.make_node("ReduceMin", ["x"], ["min_two_axes"], axes=[0, 2]),
        helper.make_node("ReduceSum", ["x", "middle_axis"], ["sum_middle"], keepdims=0),
        helper.make_node("ReduceSum", ["x"], ["sum_all"]),
        helper.make_node("ReduceSum", ["x"], ["sum_nothing"], noop_with_empty_axes=1),
        helper.make_node("ReduceSum", ["counts", "middle_axis"], ["sum_int"]),
        helper.make_node("CumSum", ["x", "middle_scalar"], ["cumsum"]),
        helper.make_node("CumSum", ["x", "last_axis"], ["cumsum_exclusive"], exclusive=1),
        helper.make_node("CumSum", ["x", "last_axis"], ["cumsum_reverse"], reverse=1),
        helper.make_node("CumSum", ["x", "last_axis"], ["cumsum_exclusive_reverse"], exclusive=1, reverse=1),
    ]
    inputs = {"x": RANDOM.standard_normal((2, 3, 4)).astype(np.float32)}
    weights = [
        get_ints("middle_axis", [1]), get_ints("middle_scalar", 1), get_ints("last_axis", -1),
        get_ints("counts", [[1, 2, 3], [4, 5, 6]]),
    ]
    save_case("reduce", nodes, inputs, [node.output[0] for node in nodes], weights)


def make_normalize_case():
    nodes = [
        helper.make_node("Softmax", ["x"], ["softmax"]),
        helper.make_node("Softmax", ["x"], ["softmax_middle"], axis=1),
        helper.make_node("LogSoftmax", ["x"], ["log_softmax"]),
        helper.make_node("LayerNormalization", ["x", "scale", "shift"], ["layer_norm"]),
        helper.make_node("LayerNormalization", ["x", "wide_scale"], ["layer_norm_two_axes"], axis=1, epsilon=1e-3),
        helper.make_node("BatchNormalization", ["x", "channel_scale", "channel_shift", "channel_mean", "channel_variance"], ["batch_norm"]),
    ]
    inputs = {"x": RANDOM.standard_normal((2, 3, 4)).astype(np.float32)}
    weights = [
        get_random("scale", 4), get_random("shift", 4), get_random("wide_scale", 3, 4),
        get_random("channel_scale", 3), get_random("channel_shift", 3), get_random("channel_mean", 3),
        get_weight("channel_variance", np.abs(RANDOM.standard_normal(3)).astype(np.float32) + 0.5),
    ]
    save_case("normalize", nodes, inputs, [node.output[0] for node in nodes], weights)


def make_shape_case():
    nodes = [
        helper.make_node("Reshape", ["x", "keep_and_infer"], ["reshape"]),
        helper.make_node("Flatten", ["x"], ["flatten_start"], axis=0),
        helper.make_node("Flatten", ["x"], ["flatten_last"], axis=2),
        helper.make_node("Unsqueeze", ["x", "outer_axes"], ["unsqueeze"]),
        helper.make_node("Squeeze", ["unsqueeze", "first_axis"], ["squeeze_axis"]),
        helper.make_node("Squeeze", ["unsqueeze"], ["squeeze_all"]),
        helper.make_node("Transpose", ["x"], ["transpose"], perm=[1, 2, 0]),
        helper.make_node("Transpose", ["x"], ["transpose_reverse"]),
        helper.make_node("Expand", ["row", "full_shape"], ["expand"]),
        helper.make_node("Shape", ["x"], ["shape"]),
        helper.make_node("Shape", ["x"], ["shape_part"], start=1, end=-1),
        helper.make_node("Cast", ["x"], ["cast_int"], to=TensorProto.INT64),
        helper.make_node("Cast", ["cast_int"], ["cast_float"], to=TensorProto.FLOAT),
        helper.make_node("ConstantOfShape", ["fill_shape"], ["filled"], value=helper.make_tensor("value", TensorProto.FLOAT, [1], [1.5])),
        helper.make_node("ConstantOfShape", ["fill_shape"], ["zeros"]),
        helper.make_node("Range", ["range_start", "range_limit", "range_step"], ["range_int"]),
        helper.make_node("Range", ["float_start", "float_limit", "float_step"], ["range_float"]),
        helper.make_node("Identity", ["x"], ["identity"]),
        helper.make_node("Dropout", ["x"], ["dropout"]),
    ]
    inputs = {"x": (RANDOM.standard_normal((2, 3, 4)) * 3).astype(np.float32)}
    weights = [
        get_ints("keep_and_infer", [0, -1]), get_ints("outer_axes", [0, -1]), get_ints("first_axis", [0]),
        get_random("row", 4), get_ints("full_shape", [2, 3, 4]), get_ints("fill_shape", [2, 3]),
        get_ints("range_start", 1), get_ints("range_limit", 10), get_ints("range_step", 3),
        get_float("float_start", 0.5), get_float("float_limit", 2.0), get_float("float_step", 0.25),
    ]
    save_case("shape", nodes, inputs, [node.output[0] for node in nodes], weights)


def make_select_case():
    nodes = [
        helper.make_node("Concat", ["x", "x"], ["concat"], axis=-1),
        helper.make_node("Concat", ["first_ints", "second_ints"], ["concat_int"], axis=0),
        helper.make_node("Slice", ["x", "negative_start", "large_end", "last_axis"], ["slice_negative"]),
        helper.make_node("Slice", ["x", "default_starts", "default_ends"], ["slice_default"]),
        helper.make_node("Slice", ["x", "zero", "large_end", "last_axis", "two"], ["slice_steps"]),
        helper.make_node("Gather", ["x", "negative_indices"], ["gather_negative"], axis=1),
        helper.make_node("Gather", ["x", "scalar_index"], ["gather_scalar"]),
        helper.make_node("GatherElements", ["x", "row_indices"], ["gather_elements_rows"], axis=0),
        helper.make_node("GatherElements", ["x", "column_indices"], ["gather_elements_columns"], axis=2),
        helper.make_node("Split", ["x", "split_sizes"], ["split_first", "split_second"], axis=2),
        helper.make_node("Split", ["x"], ["split_equal_0", "split_equal_1", "split_equal_2"], axis=1),
        helper.make_node("Pad", ["x", "pads", "pad_value"], ["pad"]),
        helper.make_node("Pad", ["x", "pads"], ["pad_zero"]),
        helper.make_node("Tile", ["x", "repeats"], ["tile"]),
    ]
    outputs = [output for node in nodes for output in node.output]
    inputs = {"x": RANDOM.standard_normal((2, 3, 4)).astype(np.float32)}
    weights = [
        get_ints("first_ints", [1, 2]), get_ints("second_ints", [3]),
        get_ints("negative_start", [-3]), get_ints("large_end", [100]), get_ints("last_axis", [-1]),
        get_ints("default_starts", [0, 1]), get_ints("default_ends", [1, 3]), get_ints("zero", [0]), get_ints("two", [2]),
        get_ints("negative_indices", [-1, 0]), get_ints("scalar_index", 1),
        get_ints("row_indices", [[[1, 0, 1, 1], [0, 1, 1, -1], [1, 1, 0, 0]]]),
        get_ints("column_indices", [[[3, 0], [1, -2], [0, 0]], [[2, 2], [-1, 1], [3, 3]]]),
        get_ints("split_sizes", [1, 3]), get_ints("pads", [0, 1, 0, 1, 0, 2]), get_float("pad_value", 0.5),
        get_ints("repeats", [1, 2, 1]),
    ]
    save_case("select", nodes, inputs, outputs, weights)


def make_constant_case():
    nodes = [
        helper.make_node("Constant", [], ["value_tensor"], value=helper.make_tensor("value", TensorProto.FLOAT, [2], [1.5, -2.0])),
        helper.make_node("Constant", [], ["value_float"], value_float=3.0),
        helper.make_node("Constant", [], ["value_floats"], value_floats=[0.25, 0.75]),
        helper.make_node("Constant", [], ["value_int"], value_int=4),
        helper.make_node("Constant", [], ["value_ints"], value_ints=[5, 6]),
        helper.make_node("Add", ["x", "value_tensor"], ["add_tensor"]),
        helper.make_node("Mul", ["x", "value_float"], ["mul_float"]),
        helper.make_node("Sub", ["x", "value_floats"], ["sub_floats"]),
        helper.make_node("Add", ["value_ints", "value_int"], ["add_ints"]),
    ]
    inputs = {"x": RANDOM.standard_normal(2).astype(np.float32)}
    outputs = ["add_tensor", "mul_float", "sub_floats", "add_ints", "value_tensor"]
    save_case("constant", nodes, inputs, outputs, [])


def get_raw(name, kind, values, dtype):
    return helper.make_tensor(name, kind, [len(values)], np.asarray(values, dtype=dtype).tobytes(), raw=True)


def make_weights_case():
    halves = [0.5, -2.0, 6e-8, 65504.0, 0.333]
    weights = [
        get_raw("float_raw", TensorProto.FLOAT, [1.5, -2.25], np.float32),
        helper.make_tensor("float_list", TensorProto.FLOAT, [2], [1.5, -2.25]),
        get_raw("double_raw", TensorProto.DOUBLE, [0.1, -7.5], np.float64),
        helper.make_tensor("double_list", TensorProto.DOUBLE, [2], [0.1, -7.5]),
        get_raw("half_raw", TensorProto.FLOAT16, halves, np.float16),
        helper.make_tensor("half_list", TensorProto.FLOAT16, [len(halves)], np.asarray(halves, dtype=np.float16)),
        get_raw("int64_raw", TensorProto.INT64, [-9, 1 << 40], np.int64),
        helper.make_tensor("int64_list", TensorProto.INT64, [2], [-9, 1 << 40]),
        get_raw("uint64_raw", TensorProto.UINT64, [3, 1 << 40], np.uint64),
        get_raw("int32_raw", TensorProto.INT32, [-70000, 12], np.int32),
        helper.make_tensor("int32_list", TensorProto.INT32, [2], [-70000, 12]),
        get_raw("uint32_raw", TensorProto.UINT32, [70000, 1], np.uint32),
        get_raw("int16_raw", TensorProto.INT16, [-300, 7], np.int16),
        get_raw("uint16_raw", TensorProto.UINT16, [60000, 7], np.uint16),
        get_raw("int8_raw", TensorProto.INT8, [-100, 100], np.int8),
        get_raw("uint8_raw", TensorProto.UINT8, [200, 3], np.uint8),
        get_raw("bool_raw", TensorProto.BOOL, [True, False], np.bool_),
        helper.make_tensor("bool_list", TensorProto.BOOL, [2], [False, True]),
    ]
    nodes = [helper.make_node("Identity", [weight.name], [f"{weight.name}_output"]) for weight in weights]
    save_case("weights", nodes, {}, [node.output[0] for node in nodes], weights)


def make_graph_case():
    nodes = [
        helper.make_node("Relu", ["x"], ["shared"]),
        helper.make_node("Add", ["shared", "listed_bias"], ["first_use"]),
        helper.make_node("Mul", ["shared", "first_use"], ["second_use"]),
    ]
    inputs = {"x": RANDOM.standard_normal((2, 3)).astype(np.float32)}
    weights = [get_random("listed_bias", 3)]
    outputs = ["first_use", "second_use", "listed_bias"]
    save_case("graph", nodes, inputs, outputs, weights, listed_weights=["listed_bias"])


def make_small_model_case(points=64, channels=8, blocks=3, segments=4, context_count=3, target_count=5):
    node_count = segments + 1
    node_step = 0.3
    weights = [
        get_ints("axis_0", [0]), get_ints("axis_1", [1]), get_ints("last_axis", -1), get_ints("last_axis_list", [-1]),
        get_float("half", 0.5), get_float("one", 1.0), get_float("scale_start", 0.3), get_float("shift_start", 0.5),
        get_float("node_step", node_step), get_float("max_correction", 0.12), get_float("min_sigma", 0.001),
        get_float("sigma_span", 0.199), get_float("sqrt_half", np.sqrt(0.5)),
        get_ints("segment_shape", [1, channels, segments, points // segments]),
        get_ints("rt_column", [1]),
        get_random("encoder_in", channels, 1, 3, scale=0.3), get_random("encoder_in_bias", channels, scale=0.3),
    ]
    nodes = [
        helper.make_node("Concat", ["target_eics", "context_eics"], ["all_eics"], axis=0),
        helper.make_node("ReduceSum", ["all_eics", "axis_0"], ["chromatogram_flat"], keepdims=1),
        helper.make_node("Unsqueeze", ["chromatogram_flat", "axis_0"], ["chromatogram"]),
        helper.make_node("Conv", ["chromatogram", "encoder_in", "encoder_in_bias"], ["feature_0"], pads=[1, 1]),
    ]
    for block in range(blocks):
        dilation = 2 ** block
        weights += [get_random(f"block_{block}", channels, channels, 3, scale=0.1), get_random(f"block_{block}_bias", channels, scale=0.1)]
        nodes += [
            helper.make_node("Conv", [f"feature_{block}", f"block_{block}", f"block_{block}_bias"], [f"block_{block}_conv"], dilations=[dilation], pads=[dilation, dilation]),
            helper.make_node("Mul", [f"block_{block}_conv", "sqrt_half"], [f"block_{block}_scaled"]),
            helper.make_node("Erf", [f"block_{block}_scaled"], [f"block_{block}_erf"]),
            helper.make_node("Add", [f"block_{block}_erf", "one"], [f"block_{block}_erf_one"]),
            helper.make_node("Mul", [f"block_{block}_conv", f"block_{block}_erf_one"], [f"block_{block}_gelu_double"]),
            helper.make_node("Mul", [f"block_{block}_gelu_double", "half"], [f"block_{block}_gelu"]),
            helper.make_node("Add", [f"feature_{block}", f"block_{block}_gelu"], [f"feature_{block + 1}"]),
        ]
    feature_map = f"feature_{blocks}"
    weights += [
        get_random("start_dense", 2 * channels, 1, scale=0.3), get_random("start_bias", 1, scale=0.3),
        get_random("step_conv", channels, 3 * channels, 1, scale=0.3), get_random("step_conv_bias", channels, scale=0.3),
        get_random("step_out", 1, channels, 1, scale=0.3), get_random("step_out_bias", 1, scale=0.3),
        get_ints("global_expand_shape", [1, 2 * channels, segments]),
        get_float("zero_column", np.zeros((1, 1), dtype=np.float32)),
        get_weight("reference_nodes", (np.arange(node_count, dtype=np.float32) * node_step + 0.5).reshape(1, node_count)),
        get_ints("first_node", [0]), get_ints("last_index", [node_count - 2]), get_ints("one_int", 1),
    ]
    nodes += [
        helper.make_node("GlobalAveragePool", [feature_map], ["mean_pool"]),
        helper.make_node("ReduceMax", [feature_map], ["max_pool"], axes=[2]),
        helper.make_node("Concat", ["mean_pool", "max_pool"], ["global_column"], axis=1),
        helper.make_node("Flatten", ["global_column"], ["global_context"], axis=1),
        helper.make_node("Reshape", [feature_map, "segment_shape"], ["segments_split"]),
        helper.make_node("ReduceMean", ["segments_split"], ["segments_mean"], axes=[3], keepdims=0),
        helper.make_node("Unsqueeze", ["global_context", "last_axis_list"], ["global_column_3d"]),
        helper.make_node("Expand", ["global_column_3d", "global_expand_shape"], ["global_repeated"]),
        helper.make_node("Concat", ["segments_mean", "global_repeated"], ["segment_input"], axis=1),
        helper.make_node("Conv", ["segment_input", "step_conv", "step_conv_bias"], ["step_hidden"]),
        helper.make_node("Relu", ["step_hidden"], ["step_active"]),
        helper.make_node("Conv", ["step_active", "step_out", "step_out_bias"], ["step_raw"]),
        helper.make_node("Softplus", ["step_raw"], ["step_sizes_3d"]),
        helper.make_node("Flatten", ["step_sizes_3d"], ["step_sizes"], axis=1),
        helper.make_node("Gemm", ["global_context", "start_dense", "start_bias"], ["start_raw"]),
        helper.make_node("Tanh", ["start_raw"], ["start_tanh"]),
        helper.make_node("Mul", ["start_tanh", "scale_start"], ["start_scaled"]),
        helper.make_node("Add", ["start_scaled", "shift_start"], ["start"]),
        helper.make_node("Mul", ["step_sizes", "node_step"], ["step_times"]),
        helper.make_node("CumSum", ["step_times", "last_axis"], ["step_total"]),
        helper.make_node("Concat", ["zero_column", "step_total"], ["node_offsets"], axis=1),
        helper.make_node("Add", ["node_offsets", "start"], ["warped_nodes"]),
        helper.make_node("Gather", ["targets", "rt_column"], ["reference_rt"], axis=1),
        helper.make_node("Greater", ["reference_rt", "reference_nodes"], ["is_after"]),
        helper.make_node("Cast", ["is_after"], ["is_after_count"], to=TensorProto.INT64),
        helper.make_node("ReduceSum", ["is_after_count", "axis_1"], ["after_count"], keepdims=0),
        helper.make_node("Sub", ["after_count", "one_int"], ["left_raw"]),
        helper.make_node("Max", ["left_raw", "first_node"], ["left_low"]),
        helper.make_node("Min", ["left_low", "last_index"], ["left_index"]),
        helper.make_node("Add", ["left_index", "one_int"], ["right_index"]),
        helper.make_node("Squeeze", ["reference_nodes", "axis_0"], ["reference_list"]),
        helper.make_node("Squeeze", ["warped_nodes", "axis_0"], ["warped_list"]),
        helper.make_node("Gather", ["reference_list", "left_index"], ["reference_left"]),
        helper.make_node("Gather", ["reference_list", "right_index"], ["reference_right"]),
        helper.make_node("Gather", ["warped_list", "left_index"], ["warped_left"]),
        helper.make_node("Gather", ["warped_list", "right_index"], ["warped_right"]),
        helper.make_node("Squeeze", ["reference_rt", "axis_1"], ["reference_rt_list"]),
        helper.make_node("Sub", ["reference_rt_list", "reference_left"], ["rt_offset"]),
        helper.make_node("Sub", ["reference_right", "reference_left"], ["reference_gap"]),
        helper.make_node("Div", ["rt_offset", "reference_gap"], ["fraction"]),
        helper.make_node("Sub", ["warped_right", "warped_left"], ["warped_gap"]),
        helper.make_node("Mul", ["fraction", "warped_gap"], ["warped_offset"]),
        helper.make_node("Add", ["warped_left", "warped_offset"], ["warped_rt_list"]),
        helper.make_node("Unsqueeze", ["warped_rt_list", "axis_1"], ["warped_rt"]),
    ]
    weights += [
        get_random("eic_dense", points, channels, scale=0.05), get_random("eic_bias", channels, scale=0.3),
        get_random("correction_dense", 4 * channels + 2, channels, scale=0.1), get_random("correction_bias", channels, scale=0.3),
        get_random("query", channels, channels, scale=0.1), get_random("key", channels, channels, scale=0.1),
        get_random("value", channels, channels, scale=0.1),
        get_random("correction_out", channels, 1, scale=0.3), get_random("correction_out_bias", 1, scale=0.3),
        get_random("sigma_dense", 4 * channels + 2, channels, scale=0.1), get_random("sigma_bias", channels, scale=0.3),
        get_random("sigma_out", channels, 1, scale=0.3), get_random("sigma_out_bias", 1, scale=0.3),
        get_ints("target_global_shape", [target_count, 2 * channels]),
        get_float("attention_scale", 1.0 / np.sqrt(channels)),
    ]
    nodes += [
        helper.make_node("MatMul", ["target_eics", "eic_dense"], ["eic_raw"]),
        helper.make_node("Add", ["eic_raw", "eic_bias"], ["eic_features"]),
        helper.make_node("Expand", ["global_context", "target_global_shape"], ["target_global"]),
        helper.make_node("Concat", ["eic_features", "eic_features", "target_global", "warped_rt", "reference_rt"], ["target_input"], axis=1),
        helper.make_node("MatMul", ["target_input", "correction_dense"], ["correction_raw"]),
        helper.make_node("Add", ["correction_raw", "correction_bias"], ["correction_hidden"]),
        helper.make_node("Tanh", ["correction_hidden"], ["correction_active"]),
        helper.make_node("MatMul", ["correction_active", "query"], ["queries"]),
        helper.make_node("MatMul", ["correction_active", "key"], ["keys"]),
        helper.make_node("MatMul", ["correction_active", "value"], ["values"]),
        helper.make_node("Transpose", ["keys"], ["keys_flipped"], perm=[1, 0]),
        helper.make_node("MatMul", ["queries", "keys_flipped"], ["scores_raw"]),
        helper.make_node("Mul", ["scores_raw", "attention_scale"], ["scores"]),
        helper.make_node("Softmax", ["scores"], ["attention"], axis=-1),
        helper.make_node("MatMul", ["attention", "values"], ["attended"]),
        helper.make_node("Add", ["correction_active", "attended"], ["correction_mixed"]),
        helper.make_node("MatMul", ["correction_mixed", "correction_out"], ["correction_one"]),
        helper.make_node("Add", ["correction_one", "correction_out_bias"], ["correction_one_bias"]),
        helper.make_node("Tanh", ["correction_one_bias"], ["correction_tanh"]),
        helper.make_node("Mul", ["correction_tanh", "max_correction"], ["correction"]),
        helper.make_node("Add", ["warped_rt", "correction"], ["predicted_column"]),
        helper.make_node("Flatten", ["predicted_column"], ["predicted_flat"], axis=0),
        helper.make_node("Squeeze", ["predicted_flat", "axis_0"], ["predicted_rt"]),
        helper.make_node("MatMul", ["target_input", "sigma_dense"], ["sigma_raw"]),
        helper.make_node("Add", ["sigma_raw", "sigma_bias"], ["sigma_hidden"]),
        helper.make_node("Relu", ["sigma_hidden"], ["sigma_active"]),
        helper.make_node("Gemm", ["sigma_active", "sigma_out", "sigma_out_bias"], ["sigma_one"]),
        helper.make_node("Sigmoid", ["sigma_one"], ["sigma_part"]),
        helper.make_node("Mul", ["sigma_part", "sigma_span"], ["sigma_spread"]),
        helper.make_node("Add", ["sigma_spread", "min_sigma"], ["sigma_column"]),
        helper.make_node("Squeeze", ["sigma_column", "axis_1"], ["uncertainty"]),
    ]
    inputs = {
        "target_eics": np.abs(RANDOM.standard_normal((target_count, points))).astype(np.float32),
        "targets": np.column_stack([RANDOM.uniform(100, 400, target_count), RANDOM.uniform(0.6, 1.6, target_count)]).astype(np.float32),
        "context_eics": np.abs(RANDOM.standard_normal((context_count, points))).astype(np.float32),
    }
    save_case("small_model", nodes, inputs, ["predicted_rt", "uncertainty"], weights)


def make_error_models():
    relu = [helper.make_node("Relu", ["x"], ["y"])]
    settings = {"points": "64", "time_from": " 0.5 "}
    save_model("settings", build_model("settings", relu, {"x": np.zeros(1, np.float32)}, ["y"], [], settings=settings))
    save_model("old_opset", build_model("old_opset", relu, {"x": np.zeros(1, np.float32)}, ["y"], [], opset=12))
    unknown = [helper.make_node("Unknown", ["x"], ["y"])]
    save_model("unknown_layer", build_model("unknown_layer", unknown, {"x": np.zeros(1, np.float32)}, ["y"], []))
    backwards = [helper.make_node("Relu", ["middle"], ["y"]), helper.make_node("Relu", ["x"], ["middle"])]
    save_model("wrong_order", build_model("wrong_order", backwards, {"x": np.zeros(1, np.float32)}, ["y"], []))
    bad_reshape = [helper.make_node("Reshape", ["x", "wrong_shape"], ["y"])]
    save_model("bad_reshape", build_model("bad_reshape", bad_reshape, {"x": np.zeros(1, np.float32)}, ["y"], [get_ints("wrong_shape", [7])]))
    outside = helper.make_tensor("outside", TensorProto.FLOAT, [2], [1.0, 2.0])
    outside.ClearField("float_data")
    outside.data_location = TensorProto.EXTERNAL
    outside.external_data.add(key="location", value="outside.bin")
    outside_nodes = [helper.make_node("Add", ["x", "outside"], ["y"])]
    save_model("outside_weights", build_model("outside_weights", outside_nodes, {"x": np.zeros(2, np.float32)}, ["y"], [outside]))


if __name__ == "__main__":
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    for old_file in list(OUTPUT_FOLDER.glob("*.onnx")) + list(OUTPUT_FOLDER.glob("*.json")):
        old_file.unlink()
    make_math_case()
    make_arithmetic_case()
    make_compare_case()
    make_conv_case()
    make_linear_case()
    make_pool_case()
    make_reduce_case()
    make_normalize_case()
    make_shape_case()
    make_select_case()
    make_constant_case()
    make_weights_case()
    make_graph_case()
    make_small_model_case()
    make_error_models()
