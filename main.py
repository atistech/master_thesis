import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import datasets
dataset = datasets.MnistDataset()

from population import Population
population = Population()

fittest = None
secondFittest = None
generationCount = 0

population.calculateFitness(dataset)
print("Generation: {}  Fittest: {}"
    .format(generationCount, population.fittest))

while population.fittest < 92:
    ++generationCount

    # selection()
    # crossover()
    # mutation()
    # addFittestOffspring()
    
    population.calculateFitness(dataset)
    print("Generation: {} Fittest: {}".format(generationCount, population.fittest))

print("Solution found in generation " + str(generationCount))
print("Fitness: " + str(population.getFittest().fitness))
print("Model: " + str(population.getFittest().model_text))