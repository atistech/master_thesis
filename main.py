import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import datasets
import population
import genetic_algorithm as GA
# import time
# time.sleep(10)

class Main:
    # Define a dataset
    dataset = datasets.MnistDataset()

    # Define generation count as 0
    generationCount = 0

    # Main() constructor
    def __init__(self):
        # Define a population with a dataset
        self.population = population.Population(self.dataset)

        # Calculate fitness of population's all individuals
        self.population.calculateFitness()

        # Show the highest fitness score of current generation
        print("Generation: {}  Fittest Score: {} Fittest Model: {}"
            .format(self.generationCount, self.population.fittest, self.population.fittest_model))

        # Generate new generations along the population's highest fitness score less than 93
        while self.generationCount < 10:
            # Increase the generation count by 1
            self.generationCount += 1

            #selection
            selectedIndividuals = GA.selection(self.population.individuals)

            # cross over
            self.population.individuals.clear()
            offsprings = GA.crossOver(self.dataset["input"], selectedIndividuals)
            self.population.individuals.extend(offsprings)

            
            # Calculate fitness of population's all individuals
            self.population.calculateFitness()

            # Show the highest fitness score of current generation
            print("Generation: {}  Fittest Score: {} Fittest Model: {}"
                .format(self.generationCount, self.population.fittest, self.population.fittest_model))

        # Show the solution found
        print("Solution found in generation " + str(self.generationCount))
        print("Fitness: " + str(self.population.fittest))
        print("Model: " + str(self.population.fittest_model))

if __name__ == "__main__":
    Main()