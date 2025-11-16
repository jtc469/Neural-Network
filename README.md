# Neural Network From Scratch

This project implements a neural network in pure Python and NumPy as part of my university project on "The Mathematics of Neural Networks".
The aim is to show the complete mathematical pipeline behind neural networks without using high level libraries.
It forms a practical companion to the theory explored in “The Mathematics of Neural Networks.pdf”.

The implementation includes:
- Forward propagation
- Backpropagation using the chain rule
- Gradient based optimisation
- Mean squared error loss
- ReLU, Sigmoid and Linear activations
- Saving and loading model parameters
- Training a network to approximate a nonlinear function

Everything is built explicitly to expose the mathematics behind each computation.

## Project Structure

main.py            Training script  
test.py            Loads a trained model and prints predictions  
network.py         Layer and Network classes  
activations.py     Activation functions  
losses.py          MSE loss and derivative  
models/model.npz   Saved weights and biases  

## Mathematical Overview

### Forward Pass
Each layer applies the affine map  
z = W x + b  
followed by the activation  
a = σ(z).

### Cost Function (MSE)
C = (1/n) Σ (a_i − y_i)^2.

### Backpropagation
Given ∂C/∂a at the output, gradients flow backward through:

∂C/∂z = (∂C/∂a) σ’(z)  
∂C/∂W = (∂C/∂z) xᵀ  
∂C/∂b = ∂C/∂z  
∂C/∂x = Wᵀ (∂C/∂z)

Parameters update via gradient descent:  
W ← W − η ∂C/∂W.

This shows the mathematical structure of the learning process explicitly.

## Training Example

In main.py the network is trained to approximate the function f(x) = x² with added Gaussian noise. Inputs are normalised to stabilise optimisation.

The network architecture is:
- 1 → 16 with ReLU  
- 16 → 16 with ReLU  
- 16 → 1 with Linear  

A simple training loop performs online gradient descent, logs the loss and stops early if training plateaus. The final model is saved to models/model.npz.

## Running the Code

Train:
python main.py

Evaluate:
python test.py

## Key Learning Outcomes

This project demonstrates:
- How neural networks are built using linear algebra and nonlinear activations  
- How backpropagation applies the chain rule across layers  
- How training stability depends on learning rate, scaling and activation choice  
- How to implement a complete training loop from scratch  
- How optimisation behaviour emerges from the underlying mathematics  
