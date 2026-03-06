# FUNCTIONS MODULE

Activation and cost functions with derivatives for backpropagation.

## OVERVIEW

Pure functions for forward pass (activation/cost) and backward pass (derivatives). Used by layer.rs via select_activation().

## FILES

- `activation.rs` (327L): Relu, Sigmoid, Softmax, Tanh, Linear + derivatives
- `cost.rs` (483L): MSE, MAE, BinaryCrossEntropy, CategoricalCrossEntropy, HuberLoss, HingeLoss + derivatives
- `mod.rs`: Module exports

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| **Add activation** | activation.rs | Add function + derivative, update layer.rs select_activation |
| **Add cost** | cost.rs | Add function + derivative, update layer.rs output layer logic |
| **Derivative bugs** | activation.rs/cost.rs | Check gradient calculations for backprop |
| **Numerical stability** | activation.rs softmax | exp() overflow protection via max subtraction |

## ACTIVATION FUNCTIONS

- `Relu`: max(0, x), derivative: 1 if x>0 else 0
- `Sigmoid`: 1/(1+e^-x), derivative: sigmoid(x) * (1 - sigmoid(x))
- `Softmax`: e^xi / Σe^xj, derivative: softmax(x) * (1 - softmax(x))
- `Tanh`: tanh(x), derivative: 1 - tanh²(x)
- `Linear`: x, derivative: 1

## COST FUNCTIONS

- `MSE`: mean((y_pred - y_true)²), derivative: 2(y_pred - y_true) / n
- `MAE`: mean(|y_pred - y_true|), derivative: sign(y_pred - y_true) / n
- `BinaryCrossEntropy`: -mean(y*log(p) + (1-y)*log(1-p)), derivative: -(y/p - (1-y)/(1-p)) / n
- `CategoricalCrossEntropy`: -mean(Σ y*log(p)), derivative: -y/p / n
- `HuberLoss`: MSE if |error|<δ else MAE, derivative: conditional
- `HingeLoss`: mean(max(0, 1 - y*p)), derivative: -y if y*p<1 else 0

## CONVENTIONS

- All functions take `&Array1<f64>` or `&Array2<f64>`
- Derivatives return same shape as input
- Error handling: unwrap() present in cost.rs (should propagate)

## NOTES

- Softmax uses numerical stability trick (subtract max before exp)
- Cost derivatives assume reduction by mean (division by n)
- layer.rs select_activation() maps Activation enum to these functions
