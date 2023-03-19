import random
from nn.Network import Network

class GeneticAlgorithm():

    def __init__(self, popSize):
        self.popSize = popSize
        self.network = Network(datasetSelection=0)

    def initialPopulation(self):
        self.network.createRandomModels(self.popSize)

    def selection(self):
        self.selecteds = [self.firstFittestModel, self.secondFittestModel]
        for i in range(self.popSize-2):
            r = random.randint(2, len(self.network.models)-1)
            self.selecteds.append(self.network.models[r])

    def crossOver(self):
        self.network.models.clear()
        
        for i in range(2, 6):
            crossPoint = random.randint(1,3)
            first = self.selecteds[i]
            second = self.selecteds[i+4]

            firstLayers = []
            firstLayers.extend(second.layers[:crossPoint])
            firstLayers.extend(first.layers[crossPoint:])

            secondLayers = []
            secondLayers.extend(first.layers[:crossPoint])
            secondLayers.extend(second.layers[crossPoint:])

            first.layers = firstLayers
            second.layers = secondLayers

        self.network.createModels(self.selecteds)

    def mutation(self):
        notMutates = [self.selecteds[0].toString(), self.selecteds[1].toString()]
        for model in self.network.models:
            if model.toString() not in notMutates:
                model.updateLayersRandomly()

    def calculateFitness(self):
        self.network.calculateResults()
        self.network.models.sort(key=lambda i: i.fitness, reverse=True)
        for model in self.network.models:
            print("{} {}".format(model.fitness, model.toString()))
        self.firstFittestModel = self.network.models[0]
        self.secondFittestModel = self.network.models[1]

    def populationResult(self, generationCount):
        return "Generation: {}  Fittest Score: {} Fittest Model: {}".format(
            generationCount, self.firstFittestModel.fitness, self.firstFittestModel.toString())