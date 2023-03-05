import random
import nn_params
from individual import Model, Layer

layerNums = nn_params.layerNums()
layerTypes = nn_params.layerTypes()
layerActivations = nn_params.activations()
layerOutputs = nn_params.layerOutputs()
modelOptimizers = nn_params.optimizers()

def decimalToBinary(input):
    return f'{input:04b}'

def encode_node(input):
    randrange = random.randrange(1, len(input)+1)
    return decimalToBinary(randrange)

def random_encode():
    result = ""
    layerNum = random.choice(layerNums)
    result += decimalToBinary(layerNum)
    for i in range(layerNum):
        result += encode_node(layerTypes)
        result += encode_node(layerOutputs)
        result += encode_node(layerActivations)
    for i in range(4-layerNum):
        result += "0000"+"0000"+"0000"
    result += encode_node(modelOptimizers)
    return result


def splitStringByN(input, n):
    return [input[i:i+n] for i in range(0, len(input), n)]

def binaryToDecimal(input):
    return int(input, 2)

def decode_node(input, output):
    index = binaryToDecimal(input)
    return output[index-1]

def decode(input):
    bitString = splitStringByN(input, 4)
    layerNum = binaryToDecimal(bitString[0])
    layers = []
    for i in range(4):
        if bitString[1+(i*3)] != "0000":
            name = decode_node(bitString[1+(i*3)], layerTypes)
            output = decode_node(bitString[2+(i*3)], layerOutputs)
            activation = decode_node(bitString[3+(i*3)], layerActivations)
            layers.append(Layer(name, output, activation))
    optimizer = decode_node(bitString[13], modelOptimizers)
    return Model(layerNum, layers, optimizer)