import nn.Params as Params
import random

class NetworkModel():
    
    def __init__(self, isRandom, layers):
        if isRandom:
            self.layers = []
            layersCount = random.choice(Params.layerNums)
            for i in range(layersCount):
                self.layers.append({
                    "activation": random.choice(Params.layerActivations),
                    "output": random.choice(Params.layerOutputs)
                })
            for i in range(4-layersCount):
                self.layers.append({
                    "activation": "",
                    "output": ""
                })
        else:
            self.layers = layers

    def updateLayersRandomly(self):
        layersLength = len(self.layers)
        for i in range(random.randint(1, layersLength)):
            index = random.randint(0, layersLength-1)
            self.layers[index] = {
                    "activation": random.choice(Params.layerActivations),
                    "output": random.choice(Params.layerOutputs)
                }
            
    def toString(self):
        toString = ""
        for layer in self.layers:
            if layer["activation"] != "":
                toString += "dense-{}-{}/".format(layer["activation"], layer["output"])
        return toString