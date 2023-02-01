from individual import Individual
import threading

class Population:
    individuals = []
    fittest = 0

    def __init__(self):
        activations = ["relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu"]
        activations2 = ["relu", "sigmoid"]
        optimizers = ["rmsprop", "adam"]

        for activation in activations2:
            for optimizer in optimizers:
                model_text = "784&dense_10_{}&{}/categorical_crossentropy&5".format(activation, optimizer)
                self.individuals.append(Individual(model_text))
    
    # Get the most fittest individuals
    def getMostFittests(self):
        self.individuals.sort(key=lambda i: i.fitness, reverse=True)
        self.fittest = self.individuals[0].fitness
        return self.individuals[0], self.individuals[1]

    # Get index of the least fittest individual
    def getLeastFittestIndex(self):
        minFitVal = 100
        minFitIndex = 0
        for index, individual in enumerate(self.individuals):
            if minFitVal >= individual.fitness:
                minFitVal = individual.fitness
                minFitIndex = index
        return minFitIndex
    
    # Calculate fitness of each individual
    def calculateFitness(self, dataset):
        threads = []
        for individual in self.individuals:
            thread = threading.Thread(target=individual.calcFitness, args=(dataset,))
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        self.getMostFittests()