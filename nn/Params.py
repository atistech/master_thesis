import random

# Layer numbers of neural networks
layerNums = [1, 2, 3, 4]
def randomLayerNums():
    return random.choice(layerNums)

# Layer types of neural networks
layerTypes = ["dense"]

layerOutputs = [2, 4, 8, 16, 32, 64, 128, 256, 512]
def randomLayerOutputs():
    return random.choice(layerOutputs)

# Activation functions of neural networks
layerActivations = ["relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu"]
def randomLayerActivations():
    return random.choice(layerActivations)

# Optimizer functions of neural networks
ModelOptimizers = ["rmsprop", "adam"]

# Loss functions of neural networks
ModelLossFunctions = ["categorical_crossentropy"]