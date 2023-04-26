from GeneticAlgorithm import GeneticAlgorithm

generationCount = 0

algorithm = GeneticAlgorithm(popSize=10, datasetSelection=1)

algorithm.initialPopulation()
print(algorithm.calculateFitness())
print(algorithm.populationResult(generationCount))

while generationCount < 10:
    generationCount += 1
    algorithm.selection()
    algorithm.crossOver()
    algorithm.mutation()
    print(algorithm.calculateFitness())
    print(algorithm.populationResult(generationCount))

print("Solution found.")
print(algorithm.populationResult(generationCount))