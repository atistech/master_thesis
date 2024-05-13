from Tool import Tool

params_dict = {
    "dataset": "nn/sample_dataset1.csv",
    "populationSize": 4,
    "IsRegression": True
}
tool = Tool(params_dict)
results = tool.firstModels()

print("Generation: {tool.generationCount}")
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
print(ga.individuals[0].serialize())
"""