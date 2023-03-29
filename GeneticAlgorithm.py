import random
from nn.Network import Network
from nn.NetworkModel import NetworkModel

class GeneticAlgorithm():

    def __init__(self, popSize, datasetSelection):
        self.popSize = popSize
        self.network = Network(datasetSelection)

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
        for model in self.network.models:
            print("{} {}".format(model.fitness, model.toString()))

    def populationResult(self, generationCount):
        return "Generation: {}  Fittest Score: {} Fittest Model: {}".format(
            generationCount, self.network.models[0].fitness, self.network.models[0].toString())