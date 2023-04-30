from GeneticAlgorithm import GeneticAlgorithm

generationCount = 0
params_dict = {
    "datasetSelection": "Mnist",
    "populationSize": 10
}
algorithm = GeneticAlgorithm(param_dict=params_dict)

algorithm.initialPopulation()
for result in algorithm.calculateFitness():
    print("Acc:{} Val_Acc:{} Model:{}".format(result.accuracy, result.val_accuracy, result.toString()))
result = algorithm.populationResult()
print("Generation: {}  Fittest Score: {} Fittest Model: {}"
    .format(generationCount, result.fitness, result.toString()))

while generationCount < 10:
    generationCount += 1
    algorithm.selection()
    algorithm.crossOver()
    algorithm.mutation()
    for result in algorithm.calculateFitness():
        print("Acc:{} Val_Acc:{} Model:{}".format(result.accuracy, result.val_accuracy, result.toString()))
    result = algorithm.populationResult()
    print("Generation: {}  Fittest Score: {} Fittest Model: {}"
        .format(generationCount, result.fitness, result.toString()))

print("Solution found.")
result = algorithm.populationResult()
print("Generation: {}  Fittest Score: {} Fittest Model: {}"
    .format(generationCount, result.fitness, result.toString()))