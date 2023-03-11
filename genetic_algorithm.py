import random
from individual import Individual
from nn.nn import NN
import encoder_decoder as ED


class GeneticAlgorithm():
    popSize = 10
    population = []

    def createRandomPopulation(self, dataset):
        self.neuralNetwork = NN(dataset)
        bitStrings = [ED.random_encode() for i in range(self.popSize)]
        self.neuralNetwork.createModels(bitStrings)
        self.population = self.neuralNetwork.models

    def calculateFitness(self):
        models = self.neuralNetwork.calculateResults()
        models.sort(key=lambda i: i.fitness, reverse=True)
        for model in models:
            print("{} {}".format(model.fitness, model.bitString))
        self.fittest_model = models[0].bitString
        self.fittest = models[0].fitness

    def populationResult(self, generationCount):
        return "Generation: {}  Fittest Score: {} Fittest Model: {}".format(
            generationCount, self.fittest, self.fittest_model)

    def selection(self):
        self.population.sort(key=lambda i: i.fitness, reverse=True)
        self.selecteds = [self.population[0], self.population[1]]

    def changeValues(self, first, second):
        temp = first
        first = second
        second =  temp

    def crossOver(self):
        crossPoint = random.randint(1,3)
        first_cross_part_1 = self.selecteds[0].bitString[:crossPoint*12]
        first_cross_part_2 = self.selecteds[0].bitString[crossPoint*12:]
        second_cross_part_1 = self.selecteds[1].bitString[:crossPoint*12]
        second_cross_part_2 = self.selecteds[1].bitString[crossPoint*12:]
        self.changeValues(first_cross_part_1, second_cross_part_1)
        
        bitString_1 = first_cross_part_1 + first_cross_part_2
        bitString_2 = second_cross_part_1 + second_cross_part_2

        self.population.clear()
        self.neuralNetwork.models.clear()
        self.neuralNetwork.createModels([bitString_1, bitString_2])
        self.population = self.neuralNetwork.models