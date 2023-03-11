import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import datasets
from genetic_algorithm import GeneticAlgorithm

dataset = datasets.MnistDataset()
generationCount = 0

algorithm = GeneticAlgorithm()

algorithm.createRandomPopulation(dataset)
algorithm.calculateFitness()
print(algorithm.populationResult(generationCount))

while generationCount < 10:
    generationCount += 1
    algorithm.selection()
    algorithm.crossOver()
    algorithm.calculateFitness()
    print(algorithm.populationResult(generationCount))

print("Solution found.")
print(algorithm.populationResult(generationCount))