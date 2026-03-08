# NEOMATRIX/CORE — HIGH-LEVEL PYTHON API

## OVERVIEW

High-level Python training API. Wraps Rust extension (`neomatrix._backend` / currently `rustybrain`) for model building, training loops, and prediction.

## FILES

| File | Lines | Purpose |
|------|-------|---------|
| `model.py` | 267 | `NeuralNetwork`, `LinearRegression`, `LogisticRegression`, `SoftmaxRegression` |
| `optimizer.py` | 44 | `SGD`, `BatchGD`, `MiniBatchGD` |
| `__init__.py` | — | Re-exports from `rustybrain`: `Tensor`, `Layer`, `Activation`, `Cost`, `get_cost` |

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Training loop | `model.py fit()` |
| Prediction | `model.py predict()` |
| Save/load | `model.py save() / load()` — JSON via `to_dict`/`from_dict` |
| Add optimizer | `optimizer.py` — subclass `Optimizer`, implement `params_update` |
| Public exports | `__init__.py` |

## CORRECT API USAGE

```python
from neomatrix.core import Layer, Activation, Cost
from neomatrix.core.model import NeuralNetwork
from neomatrix.core.optimizer import MiniBatchGD

# Layer(out_nodes, in_nodes, Activation.X)  ← output first!
layers = [
    Layer(16, 2, Activation.Relu),
    Layer(1, 16, Activation.Sigmoid),
]
model = NeuralNetwork(layers=layers, cost_function=Cost.BinaryCrossEntropy(), learning_rate=0.01)
optimizer = MiniBatchGD(training_batch_size=32, validation_batch_size=16)
model.fit(X_train, y_train, X_val, y_val, optimizer=optimizer, epochs=50, patience=5, parallel=True)
predictions = model.predict(X_test)    # returns Tensor
model.save("model.json")
model2 = NeuralNetwork.load("model.json")
```

## OPTIMIZER API

```python
# params_update(layer, w_grads, b_grads, lr) → layer.weights -= w_grads*lr
SGD()                                          # 1 sample per update
BatchGD()                                      # all samples
MiniBatchGD(training_batch_size, validation_batch_size)
```
**TODO:** Adam, RMSprop, Adagrad

## PRE-BUILT MODELS

- `LinearRegression(input_nodes, output_nodes, lr)` — 1 Linear layer + MSE
- `LogisticRegression(input_nodes, lr)` — 1 Sigmoid(1 output) + BCE
- `SoftmaxRegression(input_nodes, output_nodes, lr)` — 1 Softmax + CCE

## CONVENTIONS

- User imports: `from neomatrix.core import Tensor, Layer, NeuralNetwork`
- Type hints required for all public methods
- Explicit bool defaults: `parallel=False`, `batch_processing=True`
- `fit()` does not return self (contrary to earlier AGENTS note)
- JSON persistence: `to_dict()` / `from_dict()` / `from_json()` / `save()` / `load()`

## ANTI-PATTERNS

- **`os.system('clear')` in `fit()`** — do not add more console manipulation
- **`exit(1)`** — use `raise ValueError` instead
- No new `print()` in library code — use `logging`

## NOTES

- `__init__.py` imports from `rustybrain` (old module name) — will switch to `neomatrix._backend` once `neomatrix-python-wrapper/src/lib.rs` is complete
- No input validation at Python boundary — invalid inputs propagate to Rust as `PyErr`
- `HuberLoss` cannot be reconstructed from string — not deserializable via `TryFrom<&str>`
- `get_batches(tensor, batch_size)` in `utils/dataset.py`: 1D batch slices are reshaped to `[length, 1]`
