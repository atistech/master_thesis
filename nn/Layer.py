import nn.Params as Params

class Layer():
    def __init__(self, 
            units = Params.randomLayerOutputs(), 
            activation = Params.randomLayerActivations()):
        self.units = units
        self.activation = activation