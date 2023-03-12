import random
import nn.nn_params as nn_params

class Layer():
    def __init__(self, name, output, activation, bitParts):
        self.name = name
        self.output = output
        self.activation = activation
        self.bitParts = bitParts

layerNums = nn_params.layerNums()
layerTypes = nn_params.layerTypes()
layerActivations = nn_params.activations()
layerOutputs = nn_params.layerOutputs()

def encode_node(input):
    randrange = random.randrange(1, len(input)+1)
    return f'{randrange:04b}'

def random_encode():
    result = ""
    layerNum = random.choice(layerNums)
    for i in range(layerNum):
        result += encode_node(layerTypes)
        result += encode_node(layerOutputs)
        result += encode_node(layerActivations)
    for i in range(4-layerNum):
        result += "0000"+"0000"+"0000"
    return result

def splitStringByN(input, n):
    return [input[i:i+n] for i in range(0, len(input), n)]

def decode_node(input, output):
    index = int(input, 2)
    return output[index-1]

def decode(input):
    layers = []
    for splittedInput in splitStringByN(input, 12):
        param = splitStringByN(splittedInput, 4)
        if param[0] != "0000":
            name = decode_node(param[0], layerTypes)
            output = decode_node(param[1], layerOutputs)
            activation = decode_node(param[2], layerActivations)
            bitParts = splittedInput
            layers.append(Layer(name, output, activation, bitParts))
    return layers