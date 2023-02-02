import individual
import nn_params
import threading
import itertools

class Population:
    individuals = []
    fittest = 0

    def __init__(self, dataset):
        # Define dataset for all individuals
        self.dataset = dataset
        # Create a list with dot product
        productList = list(itertools.product(nn_params.activations()[:2], nn_params.optimizers()))
        for item in productList:
            layer = individual.Layer(output=10, activation=item[0])
            new_individual = individual.Individual(input=self.dataset["input"], layers=[layer], 
                optimizer=item[1], epochs=5)
            self.individuals.append(new_individual)
    
    # Genetic Crossing Over
    def geneticCrossingOver(self, threshold):
        fittests = [ i for i in self.individuals if i.fitness>=threshold ]
        self.individuals.clear()
        
        productList = list(itertools.product(nn_params.activations(), nn_params.optimizers(), fittests))
        
        for item in productList:
            layers = [individual.Layer(output=10, activation=item[0]),]
            layers.extend(item[2].layers)
            new_individual = individual.Individual(input=self.dataset["input"], layers=layers, 
                optimizer=item[1], epochs=5)
            self.individuals.append(new_individual)
    
    # Calculate fitness of all individuals
    def calculateFitness(self):
        threads = []
        # Calculate fitness of each individual with multi-threading
        for individual in self.individuals:
            threads.append(threading.Thread(target=individual.calcFitness, args=(self.dataset,)))

        # Start for all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to executed
        for thread in threads:
            thread.join()
        
        # Save the fittest score and fittest model
        self.individuals.sort(key=lambda i: i.fitness, reverse=True)
        self.fittest_model = self.individuals[0].toString()
        self.fittest = self.individuals[0].fitness