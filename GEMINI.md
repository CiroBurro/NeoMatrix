# NeoMatrix Project

## Project Overview

This project, NeoMatrix, is a machine learning library inspired by Keras. It combines a high-performance Rust backend with a user-friendly Python API. The core computations are handled by a Rust crate named `rustybrain`, which is then exposed as a Python module. This hybrid approach aims to provide both the speed of Rust and the ease of use of Python for building and training neural networks.

**Key Technologies:**

*   **Python:** For the high-level API and model definition.
*   **Rust:** For the core numerical computations and performance-critical operations.
*   **PyO3 & Maturin:** For the interoperability between Python and Rust.
*   **NumPy:** For numerical operations in Python and compatibility with the Rust backend.

**Architecture:**

*   **`neomatrix/`:** This directory contains the Python package that users interact with. It provides classes for building models (`NeuralNetwork`, `Layer`), defining cost functions, and running training loops.
*   **`src/`:** This directory contains the Rust source code for the `rustybrain` library. It implements the low-level tensor operations, neural network layers, and activation functions.
*   **`tests/`:** This directory contains Python-based tests for the library's functionality.

## Building and Running

**Building the Project:**

This project uses `maturin` to build the Rust library and make it available to Python. To build the project for development, you can run:

```bash
maturin develop
```

Or, to install it as a package:

```bash
pip install .
```

**Running Tests:**

The tests are written using Python's `unittest` framework. To run the tests, you can use the following command from the project's root directory:

```bash
python -m unittest discover
```

## Development Conventions

*   **Code Style:** The Python code follows standard Python conventions with type hinting and docstrings. The Rust code follows standard Rust conventions.
*   **Testing:** Tests are written in Python and are located in the `tests/` directory. New features should be accompanied by corresponding tests.
*   **Documentation:** The `README.md` file provides a good overview of the project and its usage. Docstrings in the Python code provide more detailed information about the classes and functions.
