from GeneticAlgorithm import GeneticAlgorithm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

generationCount = 0
params_dict = {
    "dataset": "nn/sample_dataset1.csv",
    "populationSize": 10,
    "IsRegression": True
}
ga = GeneticAlgorithm(param_dict=params_dict)

results = ga.initialPopulation()

print("Generation: "+str(generationCount))
for r in results:
    print(r.serialize())
"""
while generationCount < 10:
    results = ga.callback()
    print("Generation: "+str(generationCount))
    for r in results:
        print(r.serialize()) 
    generationCount += 1

print("Solution found.")
print(ga.individuals[0].serialize())"""