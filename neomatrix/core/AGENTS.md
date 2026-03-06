# NEOMATRIX CORE

High-level Python API for neural network training and inference.

## OVERVIEW

Python wrapper around rustybrain extension. Orchestrates model building, training loops, and prediction.

## FILES

- `model.py` (267L): NeuralNetwork class - fit(), predict(), save/load
- `optimizer.py`: SGD, BatchGD, MiniBatchGD implementations
- `__init__.py`: Re-exports from rustybrain (Tensor, Layer, Activation, Cost)

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| **Training loop** | model.py fit() | Batch iteration, forward/backward, weight updates |
| **Prediction** | model.py predict() | Forward pass only, no backprop |
| **Save/load** | model.py save/load | JSON serialization via to_dict/from_dict |
| **Add optimizer** | optimizer.py | Implement update() method, integrate with fit() |
| **API changes** | __init__.py | Update __all__ for public exports |

## NEURAL NETWORK API

```python
from neomatrix.core import NeuralNetwork, Activation, Cost

model = NeuralNetwork()
model.add(Layer(input_size, hidden_size, Activation.Relu))
model.add(Layer(hidden_size, output_size, Activation.Softmax))
model.fit(X_train, y_train, epochs=100, lr=0.01, optimizer='sgd')
predictions = model.predict(X_test)
model.save('model.json')
```

## OPTIMIZER API

- `SGD`: Stochastic Gradient Descent (1 sample per update)
- `BatchGD`: Full batch gradient descent (all samples per update)
- `MiniBatchGD`: Mini-batch gradient descent (batch_size samples per update)

**TODO:** Adam, RMSprop, Adagrad

## CONVENTIONS

- User imports: `from neomatrix.core import Tensor, Layer, NeuralNetwork`
- Type hints required for public methods
- Booleans: explicit defaults (parallel=False, batch_processing=True)
- fit() returns self for chaining

## NOTES

- __init__.py imports from rustybrain extension module (built from src/)
- NeuralNetwork wraps Rust Layer objects, orchestrates training in Python
- No validation layer - invalid inputs propagate to Rust (PyErr)
- save/load uses JSON (to_dict/from_dict from Rust)
