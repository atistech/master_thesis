import random
from nn.nn import NN
import encoder_decoder as ED

class GeneticAlgorithm():
    popSize = 10
    population = []

    def __init__(self, dataset):
        self.neuralNetwork = NN(dataset)

    def createRandomPopulation(self):
        bitStrings = [ED.random_encode() for i in range(self.popSize)]
        self.population = self.neuralNetwork.createModels(bitStrings)

    def calculateFitness(self):
        models = self.neuralNetwork.calculateResults(self.population)
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
        for i in range(self.popSize-2):
            r = random.randint(2, len(self.population)-1)
            self.selecteds.append(self.population[r])

    def changeValues(self, first, second):
        temp = first
        first = second
        second =  temp

    def crossOver(self):
        self.population.clear()
        self.population = self.neuralNetwork.createModels([self.selecteds[0].bitString, self.selecteds[1].bitString])

        for i in range(2, 6):
            crossPoint = random.randint(1,3)
            first_cross_part_1 = self.selecteds[i].bitString[:crossPoint*12]
            first_cross_part_2 = self.selecteds[i].bitString[crossPoint*12:]
            second_cross_part_1 = self.selecteds[i+4].bitString[:crossPoint*12]
            second_cross_part_2 = self.selecteds[i+4].bitString[crossPoint*12:]
            self.changeValues(first_cross_part_1, second_cross_part_1)
            bitString_1 = first_cross_part_1 + first_cross_part_2
            bitString_2 = second_cross_part_1 + second_cross_part_2

            self.population.extend(self.neuralNetwork.createModels([bitString_1, bitString_2]))
    
    def mutation(self):
        for p in self.population[2:]:
            r = random.randint(0,4)
            ED.random_encode()
            p = self.neuralNetwork.createModels([bitString_1, bitString_2])