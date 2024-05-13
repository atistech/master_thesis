from GeneticAlgorithm import GeneticAlgorithm

class Tool(object):
    generationCount = 0

    def __init__(self, params_dict):
        self.ga = GeneticAlgorithm(param_dict=params_dict)

    def firstModels(self):
        return self.ga.initialPopulation()