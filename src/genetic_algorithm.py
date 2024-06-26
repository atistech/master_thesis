from src.nn_model import NNModel
import src.utils as utils
import random

class GeneticAlgorithm():
    def __init__(self, param_dict):
        self.models = []
        self.dataset = utils.readCSVDataset(param_dict["dataset"])
        self.isRegression = param_dict["IsRegression"]
        self.populationSize = int(param_dict["populationSize"])
        for i in range(self.populationSize):
            self.models.append(NNModel(True, [], self.isRegression))

    def selection(self):
        self.firstFittestModel = self.models[0]
        self.secondFittestModel = self.models[1]

    def crossover(self):
        self.models.clear()
        #self.models = [self.firstFittestModel, self.secondFittestModel]

        layersPool = []
        layersPool.extend(self.firstFittestModel.layers)
        layersPool.extend(self.secondFittestModel.layers)

        for i in range(self.populationSize):
            random.shuffle(layersPool)
            amount = random.randint(1,3)
            newLayers = []
            for i in range(amount):
                newLayers.append(random.choice(layersPool))
            self.models.append(NNModel(False, newLayers, self.isRegression))
        
    def mutation(self):
        for model in self.models[2:]:
            model.updateLayersRandomly()
    
    def calculation(self):
        for model in self.models:
            model.calculateResult(self.dataset)
        if(self.isRegression):
            reverse = False
        else:
            reverse = True
        self.models.sort(key=lambda i: i.fitnessScore, reverse=reverse)
        return self.models