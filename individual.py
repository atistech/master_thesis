from keras.layers import Dense
import encoder_decoder as ED

class Individual():
    fitness = 0

    def __init__(self, input, bitString):
        self.bitString = bitString
        layers = ED.decode(bitString)
        last = input
        for layer in layers:
            last = Dense(units=layer.output, activation=layer.activation)(last)
        self.output = Dense(units=10, activation="softmax")(last)