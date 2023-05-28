from GeneticAlgorithm import GeneticAlgorithm

generationCount = 0
params_dict = {
    "datasetSelection": "Mnist",
    "populationSize": 2
}
ga = GeneticAlgorithm(param_dict=params_dict)

ga.initialPopulation()
ga.calculateFitness()
print(ga.populationToString())
print(ga.populationResultToString(generationCount))

while generationCount < 10:
    generationCount += 1
    ga.selection()
    ga.crossOver()
    ga.mutation()

    ga.calculateFitness()
    print(ga.populationToString())
    print(ga.populationResultToString(generationCount))

print("Solution found.")
print(ga.populationResultToString(generationCount))