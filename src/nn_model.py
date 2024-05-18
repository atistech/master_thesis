import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from keras.models import Model as KerasModel
from keras.layers import Dense, Input
from keras import metrics
import src.utils as utils
import random

class Layer():
    def __init__(self, units = utils.randomLayerOutputs(), activation = utils.randomLayerActivations()):
        self.units = units
        self.activation = activation

class NNModel():
    def __init__(self, isRandom, layers, isRegression):
        if isRandom:
            self.layers = []
            layersCount = utils.randomLayerNums()
            for i in range(layersCount):
                self.layers.append(Layer())
            self.layers.append(Layer(units=1, activation="sigmoid"))
        else:
            self.layers = layers
        self.toKeras()
        self.optimizer = utils.randomModelOptimizers()
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
            self.layers[index] = Layer()

    def calculateResult(self, dataset):
        if(self.isRegression):
            self.model.compile(
                optimizer=self.optimizer,
                loss="binary_crossentropy",
                metrics=[
                    metrics.MeanAbsolutePercentageError(), 
                    metrics.MeanSquaredError(),
                    metrics.MeanAbsoluteError(),
                    metrics.RootMeanSquaredError()
                ]
            )

            history = self.model.fit(
                dataset["x"], dataset["y"],
                validation_split=0.2,
                epochs=5, 
                batch_size=600,
                verbose=0
            )

            print(history.history)

            self.loss = float("%.2f" % history.history['loss'][0])
            self.val_loss = float("%.2f" % history.history['val_loss'][0])
            self.mape = float("%.2f" % (history.history['mean_absolute_percentage_error'][0]))
            self.val_mape = float("%.2f" % history.history['val_mean_absolute_percentage_error'][0])
            self.mse = float("%.2f" % (history.history['mean_squared_error'][0]))
            self.val_mse = float("%.2f" % history.history['val_mean_squared_error'][0])
            self.mae = float("%.2f" % (history.history['mean_absolute_error'][0]))
            self.val_mae = float("%.2f" % history.history['val_mean_absolute_error'][0])
            mean = (self.loss + self.val_loss + self.mape + self.val_mape + self.mse + self.val_mse)/6
            self.fitnessScore = float("%.2f" % mean)
        else:
            self.model.compile(
                optimizer=self.optimizer,
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
    
    def architectureString(self):
        architectureString = ""
        for layer in self.layers:
            architectureString += f"{layer.activation}-{layer.units}/"
        architectureString += f"{self.optimizer}"
        return architectureString
    
    def toDict(self):
        if(self.isRegression):
            return {
                'history': f"mape: {self.mape},"+
                f"'val_mape': {self.val_mape},"+
                f"mse: {self.mse},"+
                f"val_mse: {self.val_mse},"+
                f"mae: {self.mae},"+
                f"val_mae: {self.val_mae},"+
                f"loss: {self.loss},"+
                f"val_loss: {self.val_loss}",
                'fitnessScore': self.fitnessScore,
                'architecture': self.architectureString()
            }
        else:
            return {
                'history': f"accuracy: {self.accuracy},"+
                f"val_accuracy: {self.val_accuracy},"+
                f"loss: {self.loss},"+
                f"val_loss: {self.val_loss}",
                'fitnessScore': self.fitnessScore,
                'architecture': self.architectureString()
            }