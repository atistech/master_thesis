import random
from nn.Network import Network
from nn.NetworkModel import NetworkModel

class GeneticAlgorithm():

    def __init__(self, popSize, param_dict):
        self.popSize = popSize
        self.network = Network(param_dict["datasetSelection"])

    def initialPopulation(self):
        self.network.createRandomModels(self.popSize)

    def selection(self):
        self.firstFittestModel = self.network.models[0]
        self.secondFittestModel = self.network.models[1]

    def crossOver(self):
        self.network.models.clear()
        newModels = [self.firstFittestModel, self.secondFittestModel]

        for i in range(4):
            crossPoint = random.randint(1,3)
            
            firstLayers = []
            firstLayers.extend(self.secondFittestModel.layers[:crossPoint])
            firstLayers.extend(self.firstFittestModel.layers[crossPoint:])
            newModels.append(NetworkModel(False, firstLayers))

            secondLayers = []
            secondLayers.extend(self.firstFittestModel.layers[:crossPoint])
            secondLayers.extend(self.secondFittestModel.layers[crossPoint:])
            newModels.append(NetworkModel(False, secondLayers))

        self.network.createModels(newModels)
    
    def mutation(self):
        for model in self.network.models[2:]:
            model.updateLayersRandomly()

    def calculateFitness(self):
        self.network.calculateResults()
        self.network.models.sort(key=lambda i: i.fitness, reverse=True)
        calculatedResults = []
        calculatedResults.extend(self.network.models)
        return calculatedResults

    def populationResult(self):
        return self.network.models[0]