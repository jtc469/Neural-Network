# Neural Network From Scratch

This project implements a neural network in pure Python and NumPy as part of my university project on "The Mathematics of Neural Networks".
The aim is to show the complete mathematical behind the training of a neural network, with no external libraries.
It applies the theory explored in “The Mathematics of Neural Networks.pdf”.

The implementation includes:
- Forward propagation
- Backpropagation
- Gradient based optimisation
- Mean squared error loss
- ReLU, Sigmoid and Linear activations
- Saving and loading model parameters
- Training a network to approximate a nonlinear function

## Training Example

In main.py the network is trained to approximate the function f(x) = x² with added Gaussian noise.

The network architecture is:
- 1 → 16 with ReLU  
- 16 → 16 with ReLU  
- 16 → 1 with Linear  
## Key Learning Outcomes
