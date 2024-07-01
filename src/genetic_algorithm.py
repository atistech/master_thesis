from src.nn_model import NNModel
import src.utils as utils
import threading
import random

class GeneticAlgorithm():
    def __init__(self, param_dict):
        self.models = []
        self.dataset = utils.readCSVDataset(param_dict["dataset"])
        self.taskType = param_dict["taskType"]
        self.populationSize = int(param_dict["populationSize"])
        for i in range(self.populationSize):
            params_dict = {
                "taskType": self.taskType,
                "input": self.dataset["input"],
                "output": self.dataset["output"],
                "newLayers": [],
                "optimizer": "",
                "isRandom": True
            }
            self.models.append(NNModel(params=params_dict))

    def selection(self):
        self.firstFittestModel = self.models[0]
        self.secondFittestModel = self.models[1]

    def crossover(self):
        lastLayer = self.models[0].layers[-1]
        self.models.clear()
        layersPool = []
        layersPool.extend(self.firstFittestModel.layers)
        layersPool.extend(self.secondFittestModel.layers)
        optimizersPool = []
        optimizersPool.append(self.firstFittestModel.optimizer)
        optimizersPool.append(self.secondFittestModel.optimizer)

        for i in range(self.populationSize):
            random.shuffle(layersPool)
            random.shuffle(optimizersPool)
            amount = random.randint(1,3)
            newLayers = []
            for i in range(amount):
                newLayers.append(random.choice(layersPool))
            newLayers.append(lastLayer)
            params_dict = {
                "taskType": self.taskType,
                "input": self.dataset["input"],
                "output": self.dataset["output"],
                "newLayers": newLayers,
                "optimizer": random.choice(optimizersPool),
                "isRandom": False
            }
            self.models.append(NNModel(params=params_dict))
        
    def mutation(self):
        for model in self.models:
            model.updateLayersRandomly()
    
    def calculation(self):
        threads = []
        for model in self.models:
            t1 = threading.Thread(target=model.calculateResult, args=(self.dataset,))
            t1.start()
            threads.append(t1)
        for t in threads:
            t.join()
        
        if(self.taskType == 1):
            reverse = False
        else:
            reverse = True
        self.models.sort(key=lambda i: i.fitnessScore, reverse=reverse)
        return self.models