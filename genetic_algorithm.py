import random
from individual import Individual

def selection(array):
    array.sort(key=lambda i: i.fitness, reverse=True)
    return [array[0], array[1]]

def changeValues(first, second):
    temp = first
    first = second
    second =  temp

def crossOver(population, parents):
    crossPoint = random.randint(1,3)
    first_cross_part_1 = parents[0].bitString[:crossPoint*12]
    first_cross_part_2 = parents[0].bitString[crossPoint*12:]
    second_cross_part_1 = parents[1].bitString[:crossPoint*12]
    second_cross_part_2 = parents[1].bitString[crossPoint*12:]
    changeValues(first_cross_part_1, second_cross_part_1)
        
    bitString_1 = first_cross_part_1 + first_cross_part_2
    bitString_2 = second_cross_part_1 + second_cross_part_2
    offspringIndividuals = []
    #offspringIndividuals.append(Individual(input, bitString_1))
    #offspringIndividuals.append(Individual(input, bitString_2))
    offspringIndividuals.append(population.createNewIndividual(bitString_1))
    offspringIndividuals.append(population.createNewIndividual(bitString_2))
    return offspringIndividuals