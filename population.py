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
    
    # Get the fittest individual
    def getFittest(self):
        maxFit = 90
        maxFitIndex = 0
        for index, individual in enumerate(self.individuals):
            if maxFit <= individual.fitness:
                maxFit = individual.fitness
                maxFitIndex = index
        self.fittest = self.individuals[maxFitIndex].fitness
        return self.individuals[maxFitIndex]
    
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

        self.getFittest()