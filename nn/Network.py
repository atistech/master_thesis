import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import enum
from keras.models import Model
from keras.layers import Dense
import nn.Datasets as Datasets
from nn.NetworkModel import NetworkModel
import uuid

class Network:
    models = []

    def __init__(self, datasetSelection):
        if datasetSelection == "Mnist":
            self.dataset = Datasets.MnistDataset()
        elif datasetSelection == "Fashion Mnist":
            self.dataset = Datasets.FashionMnistDataset()

    def __modelToKeras(self, model):
        model.name = str(uuid.uuid1())
        last = self.dataset["input"]
        #model.layers.append(self.dataset["output"])
        for layer in model.layers:
            if layer["activation"] != "":
                last = Dense(units=layer["output"], activation=layer["activation"])(last)
        model.output = Dense(units=self.dataset["output"]["output"], 
                            activation=self.dataset["output"]["activation"], 
                            name=model.name)(last)
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

        for i in range(len(self.models)):
            keys = [h for h in history.history if "{}_".format(self.models[i].name) in h]
            self.models[i].loss = "%.2f" % history.history[keys[0]][-1]
            self.models[i].accuracy = "%.2f" % (history.history[keys[1]][-1]*100)
            self.models[i].val_loss = "%.2f" % history.history[keys[2]][-1]
            self.models[i].val_accuracy = "%.2f" % (history.history[keys[3]][-1]*100)
            self.models[i].fitness = "%.2f" % (history.history[keys[3]][-1]*100)