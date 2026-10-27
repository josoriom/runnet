# runnet

Run ONNX neural network models in plain Rust.

## Run a model

```rust,no_run
use runnet::{Model, Tensor};

fn main() -> Result<(), runnet::Error> {
    let model = Model::load("model.onnx")?;
    let input = Tensor::from_floats([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let outputs = model.run([input])?;
    println!("{:?}", outputs[0].floats()?);
    Ok(())
}
```

## API

| Signature | Input | Output |
|---|---|---|
| `Model::load(path: impl AsRef<Path>) -> Result<Model, Error>` | Path to an `.onnx` file | The model, ready to run |
| `Model::from_bytes(bytes: &[u8]) -> Result<Model, Error>` | The bytes of an `.onnx` file | The model, ready to run |
| `model.input_names(&self) -> &[String]` | Nothing | The input names, in the order `run` takes them |
| `model.output_names(&self) -> &[String]` | Nothing | The output names, in the order `run` gives them |
| `model.get_keys(&self) -> &[String]` | Nothing | The keys stored in the file, in file order |
| `model.get_values(&self) -> &[String]` | Nothing | The values stored in the file, in the same order as the keys |
| `model.get_value(&self, key: &str) -> Option<&str>` | A key, like `"points"` | The key's text from the file, or `None` if it is missing |
| `model.run(&self, inputs: impl IntoIterator<Item = Tensor>) -> Result<Vec<Tensor>, Error>` | One `Tensor` per input name, in the same order | One `Tensor` per output name, in the same order |
| `Tensor::from_floats(shape: impl Into<Vec<usize>>, values: impl Into<Vec<f32>>) -> Tensor` | The size of each axis, like `[2, 3]`, and the values row by row | A tensor of decimals |
| `Tensor::from_ints(shape: impl Into<Vec<usize>>, values: impl Into<Vec<i64>>) -> Tensor` | The size of each axis and the values row by row | A tensor of whole numbers |
| `tensor.shape(&self) -> &[usize]` | Nothing | The size of each axis |
| `tensor.floats(&self) -> Result<&[f32], Error>` | Nothing | The decimals, borrowed; `WrongType` if the tensor holds whole numbers |
| `tensor.ints(&self) -> Result<&[i64], Error>` | Nothing | The whole numbers, borrowed; `WrongType` if the tensor holds decimals |
| `tensor.into_floats(self) -> Result<Vec<f32>, Error>` | Nothing | The decimals, taken without a copy |
| `tensor.into_ints(self) -> Result<Vec<i64>, Error>` | Nothing | The whole numbers, taken without a copy |

`Model` can be shared between threads: use one `&Model` or `Arc<Model>`.

## Layers

| Group | Layers |
|---|---|
| Math | Relu, LeakyRelu, Elu, Sigmoid, HardSigmoid, Tanh, Softplus, Gelu, Erf, Exp, Log, Sqrt, Reciprocal, Abs, Neg, Floor, Ceil, Round, Sin, Cos, Clip |
| Arithmetic | Add, Sub, Mul, Div, Pow, Max, Min, Sum |
| Compare | Equal, Less, LessOrEqual, Greater, GreaterOrEqual, And, Or, Not, Where |
| Linear | Conv (1D and 2D), MatMul, Gemm |
| Pool | MaxPool, AveragePool, GlobalMaxPool, GlobalAveragePool |
| Reduce | ReduceMean, ReduceMax, ReduceMin, ReduceSum, CumSum |
| Normalize | Softmax, LogSoftmax, LayerNormalization, BatchNormalization |
| Shape | Reshape, Flatten, Squeeze, Unsqueeze, Transpose, Expand, Shape, Cast, ConstantOfShape, Range, Identity, Dropout, Constant |
| Select | Concat, Slice, Gather, GatherElements, Split, Pad, Tile |
