import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
from nn.Model import Model
from nn.Datasets import readCSVDataset

class GeneticAlgorithm():
    models =  []
    
    def __init__(self, param_dict):
        self.dataset = readCSVDataset(param_dict["dataset"])
        self.isRegression = param_dict["IsRegression"]
        self.popSize = int(param_dict["populationSize"])

    def initialPopulation(self):
        # creating new models
        for i in range(self.popSize):
            self.models.append(Model(True, [], self.isRegression))
        
        # return new models with results
        for model in self.models:
            model.calculateResult(self.dataset)
        self.models.sort(key=lambda i: i.fitnessScore, reverse=True)
        return self.models

    def callback(self):
        # selection best two models
        self.firstFittestModel = self.models[0]
        self.secondFittestModel = self.models[1]
        
        # crossover
        self.models.clear()
        self.models = [self.firstFittestModel, self.secondFittestModel]

        layersPool = []
        layersPool.extend(self.firstFittestModel.layers)
        layersPool.extend(self.secondFittestModel.layers)

        for i in range(len(self.models)):
            random.shuffle(layersPool)
            amount = random.randint(1,3)
            newLayers = []
            for i in range(amount):
                newLayers.append(random.choice(layersPool))
            self.models.append(Model(False, newLayers))
        
        # mutation
        for model in self.models[2:]:
            model.updateLayersRandomly()
        
        # return models with results
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