import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Model
from keras.layers import Dense
from nn.ClassificationModel import ClassificationModel
from nn.RegressionModel import RegressionModel
from nn.Datasets import readCSVDataset

class ModelController:
    models = []

    def __init__(self, dataset, isRegression):
        self.dataset = readCSVDataset(dataset)
        self.isRegression = isRegression
    
    def createInitialModels(self, maxCount):
        for i in range(maxCount):
            if(self.isRegression):
                n = RegressionModel(True, [])
            else:
                n = ClassificationModel(True, [])
            self.models.append(n)

    def calculateModelResults(self):
        for model in self.models:
            model.calculateResult(self.dataset)
        self.models.sort(key=lambda i: i.fitnessScore, reverse=True)
        return self.models


"""
    def __modelToKeras(self, model):
        last = self.dataset["input"]
        for layer in model.layers:
            if layer["activation"] != "":
                last = Dense(units=layer["output"], activation=layer["activation"])(last)
        model.output = Dense(units=10, activation="softmax")(last)
        return model

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
"""