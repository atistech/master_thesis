from src.nn_search_engine import NNSearchEngine

params_dict = {
    "dataset": "sample_datasets/sample_dataset1.csv",
    "populationSize": 4,
    "maxGenerationCount": 2,
    "IsRegression": True
}
search_engine = NNSearchEngine(params_dict)

for iteration_models in search_engine:
    print(f"Generation: {search_engine.generationCount}")
    for model in iteration_models:
        print(model.toDict())

print("Solution found.")
print(search_engine.finalBestFoundModel().toDict())