import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from keras.callbacks import EarlyStopping
from keras.models import Model as KerasModel
from keras.layers import Dense, Input
import keras
import src.utils as utils
import random

class Layer():
    def __init__(self, isRandom, units = 0, activation = ""):
        if isRandom:
            self.units = utils.randomLayerOutputs()
            self.activation = utils.randomLayerActivations()
        else:
            self.units = units
            self.activation = activation

class NNModel():
    def __init__(self, params):
        self.taskType = params["taskType"]
        self.input_len = params["input"]
        if params["isRandom"]:
            self.layers = []
            layersCount = utils.randomLayerNums()
            for i in range(layersCount):
                self.layers.append(Layer(isRandom=True))
            self.layers.append(Layer(isRandom=False, units=params["output"], activation="sigmoid"))
            self.optimizer = utils.randomModelOptimizers()
        else:
            self.layers = params["newLayers"]
            self.optimizer = params["optimizer"]
        self.toKeras()

    def toKeras(self):
        input = Input(shape=(self.input_len, ))
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
        for i in range(random.randint(1, 4)):
            if layersLength > 2:
                index = random.randint(0, layersLength-2)
            else:
                index = 0
            self.layers[index] = Layer(isRandom=True)
        if random.randint(0, 1) == 1:
            self.optimizer = utils.randomModelOptimizers()

    def calculateResult(self, dataset):
        loss, monitor = "", ""
        metrics = []
        if(self.taskType == 1):
            loss="mean_absolute_error"
            metrics.append(keras.metrics.MeanAbsoluteError())
            monitor = "loss"
        elif(self.taskType == 2):
            loss="binary_crossentropy"
            metrics.append("accuracy")
            monitor = "accuracy"
        elif(self.taskType == 3):
            loss="categorical_crossentropy"
            metrics.append("accuracy")
            monitor = "accuracy"
        
        self.model.compile(
            optimizer=self.optimizer,
            loss=loss, 
            metrics=metrics
        )
        
        early_stopping = EarlyStopping(monitor=monitor, patience=5)
        history = self.model.fit(
            dataset["x"], dataset["y"],
            validation_split=0.2,
            epochs=20,
            verbose=0,
            callbacks=early_stopping
        )
        
        if(self.taskType == 1):
            self.loss = float("%.4f" % history.history['loss'][-1])
            self.val_loss = float("%.4f" % history.history['val_loss'][-1])
            self.fitnessScore = self.val_loss
        else:
            self.loss = float("%.2f" % history.history['loss'][-1])
            self.accuracy = float("%.2f" % (history.history['accuracy'][-1]*100))
            self.val_loss = float("%.2f" % history.history['val_loss'][-1])
            self.val_accuracy = float("%.2f" % (history.history['val_accuracy'][-1]*100))
            self.fitnessScore = self.val_accuracy
    
    def architectureString(self):
        architectureString = ""
        for layer in self.layers:
            architectureString += f"{layer.activation}-{layer.units}/"
        architectureString += f"{self.optimizer}"
        return architectureString
    
    def toDict(self):
        if(self.taskType == 1):
            history = f"Loss: {self.loss},   Validation Loss: {self.val_loss}"
        else:
            history = f"Accuracy: {self.accuracy}, Val Accuracy: {self.val_accuracy}",
        
        return {
            "history": history,
            "fitnessScore": self.fitnessScore,
            "architecture": self.architectureString()
        }