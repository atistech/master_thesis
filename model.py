from keras.layers import Dense
import encoder_decoder as ED

class Model():
    fitness = 0
    bitString = ""
    output = object()

    def __init__(self, bitString):
        self.bitString = bitString
        self.layers = ED.decode(bitString)