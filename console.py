from GeneticAlgorithm import GeneticAlgorithm

generationCount = 0
params_dict = {
    "datasetSelection": "Num",
    "populationSize": 2,
    "IsRegression": True
}
ga = GeneticAlgorithm(param_dict=params_dict)

results = ga.initialPopulation()

print("Generation: "+str(generationCount))
print([r.serialize() for r in results]) 

while generationCount < 2:
    results = ga.callback()
    print("Generation: "+str(generationCount))
    print([r.serialize() for r in results])   
    generationCount += 1

print("Solution found.")
print(ga.individuals[0].serialize())