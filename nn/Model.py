import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from keras.layers import Dense, Input
from keras import metrics
from keras.models import Model as KerasModel
import nn.Params as Params
from nn.Layer import Layer
import random

class Model(object):
    
    def __init__(self, isRandom, layers, isRegression):
        if isRandom:
            self.layers = []
            layersCount = Params.randomLayerNums()
            for i in range(layersCount):
                self.layers.append(Layer())
            for i in range(4-layersCount):
                self.layers.append(Layer("",""))
            self.layers.append(Layer(units=1, activation="sigmoid"))
            self.toKeras()
        else:
            self.layers = layers
            self.toKeras()
        self.isRegression = isRegression

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

    def calculateResult(self, dataset):
        if(self.isRegression):
            self.model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=[
                    metrics.MeanAbsolutePercentageError(), 
                    metrics.MeanSquaredError(),
                    metrics.MeanAbsoluteError()
                ]
            )

            history = self.model.fit(
                dataset["x"], dataset["y"],
                validation_split=0.2,
                epochs=5, 
                batch_size=600,
                verbose=0
            )

            self.loss = float("%.2f" % history.history['loss'][0])
            self.val_loss = float("%.2f" % history.history['val_loss'][0])
            self.mape = float("%.2f" % (history.history['mean_absolute_percentage_error'][0]))
            self.val_mape = float("%.2f" % history.history['val_mean_absolute_percentage_error'][0])
            self.mse = float("%.2f" % (history.history['mean_squared_error'][0]))
            self.val_mse = float("%.2f" % history.history['val_mean_squared_error'][0])
            self.mae = float("%.2f" % (history.history['mean_absolute_error'][0]))
            self.val_mae = float("%.2f" % history.history['mean_absolute_error'][0])
            mean = (self.loss + self.val_loss + self.mape + self.val_mape + self.mse + self.val_mse)/6
            self.fitnessScore = float("%.2f" % mean)
        else:
            self.model.compile(
                optimizer="adam",
                loss="categorical_crossentropy", 
                metrics=["accuracy"]
            )

            history = self.model.fit(
                self.dataset["x"], self.dataset["y"],
                validation_split=0.2,
                epochs=5, 
                batch_size=600, 
                verbose=0
            )

            self.loss = float("%.2f" % history.history['loss'][0])
            self.accuracy = float("%.2f" % (history.history['accuracy'][0]*100))
            self.val_loss = float("%.2f" % history.history['val_loss'][0])
            self.val_accuracy = float("%.2f" % (history.history['val_accuracy'][0]*100))
            self.fitnessScore = float("%.2f" % ((self.accuracy+self.val_accuracy)/2))
    
    def serialize(self):
        if(self.isRegression):
            return {
                'mape': self.mape,
                'val_mape': self.val_mape,
                'mse': self.mse,
                'val_mse': self.val_mse,
                'mae': self.mae,
                'val_mae': self.val_mae,
                'loss': self.loss,
                'val_loss': self.val_loss,
                'fitnessScore': self.fitnessScore,
                'optimizer': "adam",
                'architecture': self.toString()
            }
        else:
            return {
                'accuracy': self.accuracy,
                'val_accuracy': self.val_accuracy,
                'average_accuracy': self.fitnessScore,
                'loss': self.loss,
                'val_loss': self.val_loss,
                'optimizer': self.optimizer,
                'architecture': self.toString()
            }