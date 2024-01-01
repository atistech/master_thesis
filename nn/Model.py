import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from keras.layers import Dense, Input
from keras.models import Model as KerasModel
import nn.Params as Params
import random

class Layer():
    def __init__(self, units, activation):
        self.units = units
        self.activation = activation

class Model(object):
    
    def __init__(self, isRandom, layers):
        if isRandom:
            self.layers = []
            layersCount = random.choice(Params.layerNums)
            for i in range(layersCount):
                self.layers.append(
                    Layer(
                        units=random.choice(Params.layerOutputs),
                        activation=random.choice(Params.layerActivations)
                    )
                )
            for i in range(4-layersCount):
                self.layers.append(Layer("",""))
            self.layers.append(
                    Layer(
                        units=1,
                        activation="sigmoid"
                    )
                )
            self.toKeras()
        else:
            self.layers = layers
            self.toKeras()

    def toKeras(self):
        input = Input(shape=(3, ))
        last = input
        for layer in self.layers:
            if layer.activation != "":
                last = Dense(
                    units=layer.units, 
                    activation=layer.activation
                )(last)
        self.model = KerasModel(inputs=input, outputs=last)

    def updateLayersRandomly(self):
        layersLength = len(self.layers)
        for i in range(random.randint(1, layersLength)):
            index = random.randint(0, layersLength-1)
            random_activation = random.choice(Params.layerActivations)
            random_output = random.choice(Params.layerOutputs)
            self.layers[index] = {
                    "activation": random_activation,
                    "output": random_output
                }
    
    def toString(self):
        toString = ""
        for layer in self.layers:
            if layer.activation != "":
                toString += "dense-{}-{}/".format(layer.activation, layer.units)
            else:
                toString += "empty_layer/"
        return toString
    
    """
    def toPythonCode(self):
        stringModel = "from keras import Sequential\n"
        stringModel += "from keras.layers import Input, Dense\n\n"
        stringModel += "train_x, train_y, test_x, test_y = [], [], [], []\n\n"
        stringModel += "model = Sequential()\n"
        stringModel += "model.add(Input(shape=({}, )))\n".format(self.dataset["input"].shape[1])
        for layer in self.layers:
            if layer.activation != "":
                stringModel += "model.add(Dense(units={}, activation='{}'))\n".format(layer.units, layer.activation)
        
        stringModel += "model.add(Dense(units={}, activation='{}'))\n\n".format(self.dataset["output"]["output"], self.dataset["output"]["activation"])
        
        stringModel += "model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])\n\n"

        stringModel += "history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=1, batch_size=32, verbose=0)"

        file = open('deneme.py', 'w')
        file.write(stringModel)
        file.close()

        return stringModel
    """