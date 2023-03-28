import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import enum
from keras.models import Model
from keras.layers import Dense
import nn.Datasets as Datasets
from nn.NetworkModel import NetworkModel

class Network:
    models = []

    def __init__(self, datasetSelection):
        if datasetSelection == 0:
            self.dataset = Datasets.MnistDataset()

    def __modelToKeras(self, model):
        last = self.dataset["input"]
        model.layers.append(self.dataset["output"])
        for layer in model.layers:
            if layer["activation"] != "":
                last = Dense(units=layer["output"], activation=layer["activation"])(last)
        model.output = last
        return model

    def createRandomModels(self, howManyPiece):
        for i in range(howManyPiece):
            self.models.append(self.__modelToKeras(NetworkModel(True, [])))

    def createModels(self, models):
        for model in models:
            self.models.append(self.__modelToKeras(model))

    def calculateResults(self):
        outputs = [i.output for i in self.models]
        model = Model(self.dataset["input"], outputs)

        model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=["accuracy"]
        )

        history = model.fit(self.dataset["train_x"], self.dataset["train_y"],
            validation_data=(self.dataset["test_x"], self.dataset["test_y"]),
            epochs=5, batch_size=600, verbose=0)

        index = 0
        for h in history.history:
            if h.startswith("val") and h.endswith("accuracy"):
                self.models[index].fitness = history.history[h][-1]*100
                index += 1