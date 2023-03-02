from enum import Enum

class Activations(Enum):
    relu = "0000" #0
    sigmoid = "0001" #1
    softmax = "0010" #2
    softplus = "0011" #3
    softsign = "0100" #4
    tanh = "0101" #5
    selu = "0110" #6

class Optiizers(Enum):
    rmsprop = "0000" #0
    adam = "0001" #1

# Activation functions of neural networks
def activations():
    return ["relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu"]

# Optimizer functions of neural networks
def optimizers():
    return ["rmsprop", "adam"]