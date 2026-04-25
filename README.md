# Autograd Toy Engine

A tiny, scalar-valued autograd engine with a small neural network library on top. This project implements backpropagation (reverse-mode autodiff) over a dynamic computational graph, similar in spirit to Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

## Features

- **Scalar-valued Autograd**: Supports basic arithmetic operations (`+`, `-`, `*`, `**`) and activation functions like `tanh`.
- **Dynamic Computational Graph**: Builds a graph as operations are performed, allowing for easy `backward()` calls to compute gradients.
- **Neural Network Library**: Includes basic building blocks for deep learning:
  - `Neuron`: A single artificial neuron with weights and bias.
  - `Layer`: A collection of neurons forming a single layer.
  - `MLP` (Multi-Layer Perceptron): A sequence of layers.

## File Structure

- `value.py`: Core engine implementing the `Value` class and backpropagation logic.
- `nn.py`: Neural network abstractions (`Neuron`, `Layer`, `MLP`).
- `Untitled3.ipynb`: A demonstration notebook showing how to train a small neural network to fit a simple dataset.

## Installation

Ensure you have Python installed. You may also need `graphviz` if you want to visualize the computational graph in the notebook.

```bash
pip install graphviz
```

## Usage

### Core Engine

```python
from value import Value

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = a * b
d = c.tanh()
d.backward()

print(f"Gradient of a: {a.grad}")
print(f"Gradient of b: {b.grad}")
```

### Neural Network

```python
from nn import MLP

# 3 inputs, two hidden layers of 4, and 1 output
model = MLP(3, [4, 4, 1])

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

# Simple training loop
for k in range(20):
    # Forward pass
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    # Backward pass
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()
    
    # Update weights
    for p in model.parameters():
        p.data -= 0.01 * p.grad
    
    print(k, loss.data)
```

## Visualization

The notebook provides a `draw_dot` function to visualize the complexity of the computational graph, showing each node's data and its relative gradient.

---
Inspired by the "Building Micrograd" series.
