import random
import nn_params

def splitStringByN(input, n):
    return [input[i:i+n] for i in range(0, len(input), n)]

def decimalToBinary(input):
    return f'{input:04b}'

def binaryToDecimal(input):
    return int(input, 2)

def random_encode():
    result = ""

    layerNums_length = len(nn_params.layerNums())
    layerNums_random = random.randrange(layerNums_length)
    result += decimalToBinary(layerNums_random)

    layerTypes_length = len(nn_params.layerTypes())
    layerTypes_random = random.randrange(layerTypes_length)
    result += decimalToBinary(layerTypes_random)

    layerOutputs_length = len(nn_params.layerOutputs())
    layerOutputs_random = random.randrange(layerOutputs_length)
    result += decimalToBinary(layerOutputs_random)

    activations_length = len(nn_params.activations())
    activations_random = random.randrange(activations_length)
    result += decimalToBinary(activations_random)

    optimizers_length = len(nn_params.optimizers())
    optimizers_random = random.randrange(optimizers_length)
    result += decimalToBinary(optimizers_random)

    return result

def decode(input):
    splittedInput = splitStringByN(input, 4)

    layerNums_index = binaryToDecimal(splittedInput[0])
    layerNums = nn_params.layerNums()[layerNums_index]

    layerTypes_index = binaryToDecimal(splittedInput[1])
    layerTypes = nn_params.layerTypes()[layerTypes_index]

    layerOutputs_index = binaryToDecimal(splittedInput[2])
    layerOutputs = nn_params.layerOutputs()[layerOutputs_index]

    activation_index = binaryToDecimal(splittedInput[3])
    activation = nn_params.activations()[activation_index]
    
    optimizer_index = binaryToDecimal(splittedInput[4])
    optimizer = nn_params.optimizers()[optimizer_index]

    return str(layerNums)+"-"+str(layerTypes)+"-"+str(layerOutputs)+"-"+str(activation)+"-"+str(optimizer)



model = random_encode()
print("Encoded: "+model)
decoded = decode(model)
print("Decoded: "+str(decoded))