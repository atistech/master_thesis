import random
from nn.nn import NN
import encoder_decoder as ED
from individual import Individual

class GeneticAlgorithm():
    popSize = 10
    population = []

    def __init__(self, dataset):
        self.dataset = dataset
        self.input = self.dataset["input"]

    def createRandomPopulation(self):
        for i in range(self.popSize):
            self.population.append(Individual(self.input))

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

    def __changeGenesRandomly(self, first, second):
        crossPoint = random.randint(1,3)
        first_cross_part_1 = first.genes[:crossPoint]
        first_cross_part_2 = first.genes[crossPoint:]
        second_cross_part_1 = second.genes[:crossPoint]
        second_cross_part_2 = second.genes[crossPoint:]
        return [
            second_cross_part_1 + first_cross_part_2,
            first_cross_part_1 + second_cross_part_2
        ]


    def crossOver(self):
        self.population.clear()
        self.population.append(Individual(self.input, self.selecteds[0].chromosome))
        self.population.append(Individual(self.input, self.selecteds[1].chromosome))

        for i in range(2, 6):
            for newChromosome in self.__changeGenesRandomly(self.selecteds[i], self.selecteds[i+4]):
                offspring = Individual(self.input, newChromosome)
                offspring.mutation()
                self.population.append(offspring)