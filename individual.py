import encoder_decoder as ED
import random
import nn.nn_test as nn

class Individual():
    fitness = 0
    bitString = ""
    output = object()

    def __init__(self, *args):
        if len(args) == 1:
            self.chromose = ED.random_encode()
            self.genes = self.chromosomeSplitToGenes()
            self.output = nn.createModel(args[0])
        elif len(args) == 2:
            self.chromose = args[1]
            self.genes = self.chromosomeSplitToGenes()
            self.output = nn.createModel(args[0])

    def chromosomeSplitToGenes(self):
        input = self.chromose
        n = 12
        return [input[i:i+n] for i in range(0, len(input), n)]
    
    def mutation():
        r = random.randint(0,3)
        createNewLayerRandomly()