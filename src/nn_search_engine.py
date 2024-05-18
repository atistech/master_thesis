from src.genetic_algorithm import GeneticAlgorithm

class NNSearchEngine():
    def __init__(self, params_dict):
        self.generationCount = 0
        self.maxGenerationCount = int(params_dict["maxGenerationCount"])
        self.ga = GeneticAlgorithm(param_dict=params_dict)
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.generationCount < self.maxGenerationCount:
            if(self.generationCount != 0):
                self.ga.selection()
                self.ga.crossover()
                self.ga.mutation()
            self.generationCount += 1
            return self.ga.calculation()
        else:
            raise StopIteration
        
    def finalBestFoundModel(self):
        return self.ga.models[0]