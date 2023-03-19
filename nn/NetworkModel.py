import nn.Params as Params
import random

class NetworkModel():
    
    def __init__(self):
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

    def updateLayersRandomly(self):
        howManyTimes = random.randint(0,3)
        for i in range(howManyTimes):
            index = random.randint(0,3)
            self.layers[index] = {
                    "activation": random.choice(Params.layerActivations),
                    "output": random.choice(Params.layerOutputs)
                }
            
    def toString(self):
        toString = ""
        for layer in self.layers:
            if layer["activation"] != "":
                toString += "dense-{}-{}/".format(layer["activation"], layer["output"])
            else:
                toString += "empty_layer/"
        return toString