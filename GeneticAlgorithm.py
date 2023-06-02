import random
from nn.Model import Model
import nn.Datasets as Datasets

class GeneticAlgorithm():

    def __init__(self, param_dict):
        self.individuals = []
        self.param_dict = param_dict
        self.popSize = int(param_dict["populationSize"])
        if self.param_dict["datasetSelection"] == "Mnist":
            self.dataset = Datasets.MnistDataset()
        elif self.param_dict["datasetSelection"] == "Fashion Mnist":
            self.dataset = Datasets.FashionMnistDataset()

    def initialPopulation(self):
        for i in range(self.popSize):
            n = Model(True, [], self.dataset)
            self.individuals.append(n)

    def selection(self):
        self.firstFittestModel = self.individuals[0]
        self.secondFittestModel = self.individuals[1]

    def crossOver(self):
        self.individuals.clear()
        newModels = [self.firstFittestModel, self.secondFittestModel]

        for i in range(int((self.popSize/2) - 1)):
            crossPoint = random.randint(1,3)
            
            firstLayers = []
            firstLayers.extend(self.secondFittestModel.layers[:crossPoint])
            firstLayers.extend(self.firstFittestModel.layers[crossPoint:])
            newModels.append(Model(False, firstLayers, self.dataset))

            secondLayers = []
            secondLayers.extend(self.firstFittestModel.layers[:crossPoint])
            secondLayers.extend(self.secondFittestModel.layers[crossPoint:])
            newModels.append(Model(False, secondLayers, self.dataset))

        self.individuals.extend(newModels)
    
    def mutation(self):
        for i in self.individuals[2:]:
            i.updateLayersRandomly()

    def calculateFitness(self):
        for i in self.individuals:
            i.calculateResult()

        self.individuals.sort(key=lambda i: i.val_accuracy, reverse=True)
        calculatedResults = []
        calculatedResults.extend(self.individuals)
        return calculatedResults

    def populationResult(self):
        return self.individuals[0]
    
    def populationToString(self):
        resultString = ""
        for i in self.individuals:
            resultString += "Accucracy:{} Val_Accuracy:{} Model:{}\n".format(
                i.accuracy, 
                i.val_accuracy, 
                i.toString()
            )
        return resultString
    
    def populationResultToString(self, generationCount):
        result = self.populationResult()
        return "Generation: {}  Fittest Score: {} Fittest Model: {}".format(
            generationCount, 
            result.val_accuracy, 
            result.toString()
        )