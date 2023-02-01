import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import datasets
dataset = datasets.MnistDataset()

from population import Population
import time

class Main:
    firstFittest = object()
    secondFittest = object()
    generationCount = 0

    def __init__(self):
        self.population = Population()
        self.population.calculateFitness(dataset)
        print("Generation: {}  Fittest: {}"
            .format(self.generationCount, self.population.fittest))

        while self.population.fittest < 93:
            self.generationCount += 1

            self.selection()
            self.crossOver()
            self.mutation()
            self.addFittestOffspring()
    
            self.population.calculateFitness(dataset)
            print("Generation: {} Fittest: {}".format(self.generationCount, self.population.fittest))

            time.sleep(10)

        print("Solution found in generation " + str(self.generationCount))
        print("Fitness: " + str(self.population.getMostFittests()[0].fitness))
        print("Model: " + str(self.population.getMostFittests()[0].model_text))

    def selection(self):
        first, second = self.population.getMostFittests()
        self.firstFittest = first
        self.secondFittest = second

    def crossOver(self):
        pass

    def mutation(self):
        pass

    def addFittestOffspring(self):
        # Update fitness values of offspring
        print("update")
        self.firstFittest.calcFitness(dataset)
        self.secondFittest.calcFitness(dataset)
        print("update")

        # Replace least fittest individual from most fittest offspring
        minFitIndex = self.population.getLeastFittestIndex()
        self.population.individuals[minFitIndex] = self.getFittestOffspring()

    def getFittestOffspring(self):
        if self.firstFittest.fitness > self.secondFittest.fitness:
            return self.firstFittest
        return self.secondFittest

if __name__ == "__main__":
    Main()