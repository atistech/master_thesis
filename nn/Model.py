from keras.layers import Dense
from keras.models import Model as KerasModel
import nn.Params as Params
import random

class Layer():
    def __init__(self, units, activation):
        self.units = units
        self.activation = activation

class Model():
    optimizer = "adam"
    lossFunction = "categorical_crossentropy"
    epochs = 5
    batchSize = 600

    loss, accuracy, val_loss, val_accuracy = "", "", "", ""
    
    def __init__(self, isRandom, layers, dataset):
        self.dataset = dataset
        if isRandom:
            self.layers = []
            layersCount = random.choice(Params.layerNums)
            for i in range(layersCount):
                self.layers.append(
                    Layer(
                        random.choice(Params.layerOutputs),
                        random.choice(Params.layerActivations)
                    )
                )
            for i in range(4-layersCount):
                self.layers.append(Layer("",""))
            self.toKeras()
        else:
            self.layers = layers
            self.toKeras()

    def toKeras(self):
        last = self.dataset["input"]
        for layer in self.layers:
            if layer.activation != "":
                last = Dense(
                    units=layer.units, 
                    activation=layer.activation
                )(last)
        
        self.output = Dense(
            units=self.dataset["output"]["output"], 
            activation=self.dataset["output"]["activation"]
        )(last)
        self.model = KerasModel(inputs=self.dataset["input"], outputs=self.output)

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
            
    def calculateResult(self):
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.lossFunction, 
            metrics=["accuracy"]
        )

        history = self.model.fit(
            self.dataset["train_x"], self.dataset["train_y"],
            validation_data=(self.dataset["test_x"], self.dataset["test_y"]),
            epochs=self.epochs, 
            batch_size=self.batchSize, 
            verbose=0
        )

        self.loss = float("%.2f" % history.history['loss'][0])
        self.accuracy = float("%.2f" % (history.history['accuracy'][0]*100))
        self.val_loss = float("%.2f" % history.history['val_loss'][0])
        self.val_accuracy = float("%.2f" % (history.history['val_accuracy'][0]*100))
            
    def toString(self):
        toString = ""
        for layer in self.layers:
            if layer.activation != "":
                toString += "dense-{}-{}/".format(layer.activation, layer.units)
        return toString
    
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