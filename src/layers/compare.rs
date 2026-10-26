use crate::{
    error::Error,
    layers::{Layer, arithmetic::combine, get_input},
    model_file::NodeFile,
    tensor::{Tensor, Values, get_broadcast_positions, get_broadcast_shape},
};

#[derive(Clone, Copy)]
enum CompareKind {
    Equal,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
    And,
    Or,
}

struct CompareLayer {
    kind: CompareKind,
}

struct NotLayer;

struct WhereLayer;

pub(crate) fn find_layer(node: &NodeFile) -> Option<Box<dyn Layer>> {
    let kind = match node.kind.as_str() {
        "Equal" => CompareKind::Equal,
        "Less" => CompareKind::Less,
        "LessOrEqual" => CompareKind::LessOrEqual,
        "Greater" => CompareKind::Greater,
        "GreaterOrEqual" => CompareKind::GreaterOrEqual,
        "And" => CompareKind::And,
        "Or" => CompareKind::Or,
        "Not" => return Some(Box::new(NotLayer)),
        "Where" => return Some(Box::new(WhereLayer)),
        _ => return None,
    };
    Some(Box::new(CompareLayer { kind }))
}

impl Layer for CompareLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let left = get_input(inputs, 0)?;
        let right = get_input(inputs, 1)?;
        let kind = self.kind;
        let (shape, values) = match (&left.values, &right.values) {
            (Values::Int(_), Values::Int(_)) => combine(left, right, |a: i64, b: i64| {
                is_true(kind, a as f64, b as f64)
            })?,
            _ => {
                let left = Tensor::from_floats(left.shape.clone(), left.to_floats());
                let right = Tensor::from_floats(right.shape.clone(), right.to_floats());
                combine(&left, &right, |a: f32, b: f32| {
                    is_true(kind, a as f64, b as f64)
                })?
            }
        };
        let values: Vec<_> = values.into_iter().map(i64::from).collect();
        Ok(vec![Tensor::from_ints(shape, values)])
    }
}

fn is_true(kind: CompareKind, a: f64, b: f64) -> bool {
    match kind {
        CompareKind::Equal => a == b,
        CompareKind::Less => a < b,
        CompareKind::LessOrEqual => a <= b,
        CompareKind::Greater => a > b,
        CompareKind::GreaterOrEqual => a >= b,
        CompareKind::And => a != 0.0 && b != 0.0,
        CompareKind::Or => a != 0.0 || b != 0.0,
    }
}

impl Layer for NotLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let input = get_input(inputs, 0)?;
        let values: Vec<_> = input
            .to_ints()
            .iter()
            .map(|value| i64::from(*value == 0))
            .collect();
        Ok(vec![Tensor::from_ints(input.shape.clone(), values)])
    }
}

impl Layer for WhereLayer {
    fn run(&self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, Error> {
        let condition = get_input(inputs, 0)?;
        let if_true = get_input(inputs, 1)?;
        let if_false = get_input(inputs, 2)?;

        let shape = get_broadcast_shape(&condition.shape, &if_true.shape)?;
        let shape = get_broadcast_shape(&shape, &if_false.shape)?;
        let condition_values = condition.to_ints();
        let condition_positions = get_broadcast_positions(&condition.shape, &shape);
        let true_positions = get_broadcast_positions(&if_true.shape, &shape);
        let false_positions = get_broadcast_positions(&if_false.shape, &shape);
        let use_true = |at: usize| condition_values[condition_positions[at]] != 0;

        let values = match (&if_true.values, &if_false.values) {
            (Values::Int(a), Values::Int(b)) => Values::Int(
                (0..shape.iter().product())
                    .map(|at| {
                        if use_true(at) {
                            a[true_positions[at]]
                        } else {
                            b[false_positions[at]]
                        }
                    })
                    .collect(),
            ),
            _ => {
                let a = if_true.to_floats();
                let b = if_false.to_floats();
                Values::Float(
                    (0..shape.iter().product())
                        .map(|at| {
                            if use_true(at) {
                                a[true_positions[at]]
                            } else {
                                b[false_positions[at]]
                            }
                        })
                        .collect(),
                )
            }
        };
        Ok(vec![Tensor { shape, values }])
    }
}
