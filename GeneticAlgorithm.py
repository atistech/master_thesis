import random
from nn.Model import Model
from nn.ModelController import ModelController

class GeneticAlgorithm():
    
    def __init__(self, param_dict):
        self.dataset = param_dict["dataset"]
        self.isRegression = param_dict["IsRegression"]
        self.popSize = int(param_dict["populationSize"])
        self.controller = ModelController(self.dataset, self.isRegression)

    def initialPopulation(self):
        self.controller.createInitialModels(self.popSize)
        return self.__calculateFitness()

    def callback(self):
        self.__selection()
        self.__crossOver()
        self.__mutation()
        return self.__calculateFitness()

    def __selection(self):
        self.firstFittestModel, self.secondFittestModel = self.controller.getBestModels()

    def __crossOver(self):
        self.individuals.clear()
        newModels = [self.firstFittestModel, self.secondFittestModel]

        for i in range(int((self.popSize/2) - 1)):
            crossPoint = random.randint(1,3)
            
            firstLayers = []
            firstLayers.extend(self.secondFittestModel.layers[:crossPoint])
            firstLayers.extend(self.firstFittestModel.layers[crossPoint:])
            newModels.append(Model(False, firstLayers, self.dataset))

            secondLayers = []
            secondLayers.extend(self.firstFittestModel.layers[:crossPoint])
            secondLayers.extend(self.secondFittestModel.layers[crossPoint:])
            newModels.append(Model(False, secondLayers, self.dataset))

        self.individuals.extend(newModels)
    
    def __mutation(self):
        for i in self.individuals[2:]:
            i.updateLayersRandomly()

    def __calculateFitness(self):
        return self.controller.calculateModelResults()