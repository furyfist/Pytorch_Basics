# Pytorch_Basics

This repository contains a basic introduction to PyTorch, covering fundamental concepts such as Tensors, basic operations, and building simple neural networks :))

## Table of Contents

- [Tensors](#tensors)
- [Operations](#operations)
- [Dynamic Computation Graph and Backpropagation](#dynamic-computation-graph-and-backpropagation)
- [The Model](#the-model)
- [Loss Modules](#loss-modules)
- [Optimizers](#optimizers)

## Tensors

PyTorch's Tensors are similar to NumPy arrays but are designed for use with GPUs, enabling faster computation. The notebook covers:

- Creating Tensors (`torch.tensor`, `torch.rand`, `torch.randn`, `torch.zeros`)
- Differences between `torch.tensor` and `torch.Tensor` (data types)
- Creating multi-dimensional tensors (e.g., for images)
- Converting between Tensors and NumPy arrays (`.numpy()`, `torch.from_numpy()`, `torch.tensor()`)

## Operations

The notebook demonstrates basic tensor operations, including:

- Element-wise addition
- In-place operations (e.g., `add_`)
- Reshaping tensors using `view()` and `reshape()`, highlighting their key differences (contiguous vs. non-contiguous memory)
- Matrix multiplication (`torch.matmul`)

## Dynamic Computation Graph and Backpropagation

PyTorch uses a define-by-run approach to build a dynamic computation graph. This allows for automatic calculation of gradients using backpropagation. The notebook shows:

- How to specify which tensors require gradients (`.requires_grad`, `.requires_grad_()`)
- An example of a simple function and how PyTorch tracks operations for gradient calculation.

## The Model

The `nn.Module` is the base class for all neural network components in PyTorch. The notebook illustrates how to build a simple classifier using `nn.Module`, including:

- Defining the network structure in the `__init__` method
- Defining the forward pass in the `forward` method
- Using `nn.Linear` for linear transformations and `nn.Tanh` as an activation function
- Inspecting model parameters using `model.named_parameters()`

## Loss Modules

Loss functions measure the error between the model's predictions and the actual values. The notebook introduces:

- `nn.BCEWithLogitsLoss` which combines a sigmoid layer and the Binary Cross Entropy loss.

## Optimizers

Optimizers update the model's parameters based on the calculated gradients to minimize the loss. The notebook shows how to use:

- `torch.optim.SGD` (Stochastic Gradient Descent) to update model parameters.
