from individual import Individual,Layer
import nn_params
import threading
import itertools
import encoder_decoder as ed

class Population:
    popSize = 10
    individuals = []
    fittest = 0

    def __init__(self, dataset):
        # Define dataset for all individuals
        self.dataset = dataset
        # Create random individuals 
        for i in range(self.popSize):
            bitString = ed.random_encode()
            model = ed.decode(bitString)
            new_individual = Individual(input=self.dataset["input"], bitString=bitString, model=model)
            self.individuals.append(new_individual)
    
    # Genetic Crossing Over
    def geneticCrossingOver(self):
        self.individuals.sort(key=lambda i: i.fitness, reverse=True)
        first = self.individuals[0]
        second = self.individuals[1]
        self.individuals.clear()

        for i in range(self.popSize):
            bitString = ed.random_encode()
            model = ed.decode(bitString)
            new_individual = Individual(input=self.dataset["input"], bitString=bitString, model=model)
            self.individuals.append(new_individual)
    
    # Calculate fitness of all individuals
    def calculateFitness(self):
        threads = []
        # Calculate fitness of each individual with multi-threading
        for individual in self.individuals:
            threads.append(threading.Thread(target=individual.calculateFitness, args=(self.dataset,)))

        # Start for all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to executed
        for thread in threads:
            thread.join()
        
        # Save the fittest score and fittest model
        self.individuals.sort(key=lambda i: i.fitness, reverse=True)
        self.fittest_model = self.individuals[0].bitString
        self.fittest = self.individuals[0].fitness