from genetic_algorithm import GeneticAlgorithm

generationCount = 0

algorithm = GeneticAlgorithm(0)

algorithm.createRandomPopulation()
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