import random
from nn.ClassificationModel import ClassificationModel
from nn.Model import Model
import nn.Datasets as Datasets
import os
from nn.RegressionModel import RegressionModel

class GeneticAlgorithm():

    def __init__(self, param_dict):
        self.individuals = []
        self.param_dict = param_dict
        self.popSize = int(param_dict["populationSize"])

        if(self.param_dict["IsRegression"]):
            #self.dataset = Datasets.readCSVDataset(os.getcwd()+"/nn/sample_dataset1.csv")
            self.dataset = Datasets.readCSVDataset(param_dict["dataset"])
        else:
            self.dataset = Datasets.MnistDataset()
            #self.dataset = Datasets.FashionMnistDataset()

    def initialPopulation(self):
        for i in range(self.popSize):
            if(self.param_dict["IsRegression"]):
                n = RegressionModel(True, [], self.dataset)
                self.individuals.append(n)
            else:
                n = ClassificationModel(True, [], self.dataset)
                self.individuals.append(n)
        return self.__calculateFitness()
        

    def callback(self):
        self.__selection()
        self.__crossOver()
        self.__mutation()
        return self.__calculateFitness()

    def __selection(self):
        self.firstFittestModel = self.individuals[0]
        self.secondFittestModel = self.individuals[1]

    def __crossOver(self):
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
    
    def __mutation(self):
        for i in self.individuals[2:]:
            i.updateLayersRandomly()

    def __calculateFitness(self):
        for i in self.individuals:
            i.calculateResult()

        self.individuals.sort(key=lambda i: i.fitnessScore, reverse=True)
        return self.individuals