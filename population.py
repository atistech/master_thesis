from individual import Individual
import encoder_decoder as ED
import encoder_decoder as ed
from keras.models import Model
from keras.layers import Dense, Input

class Population:
    popSize = 10
    individuals = []
    fittest = 0

    def __init__(self, dataset):
        # Define dataset for all individuals
        self.dataset = dataset
        self.input = Input(shape=(self.dataset["input"], ))
        # Create random individuals 
        for i in range(self.popSize):
            bitString = ed.random_encode()
            #new_individual = Individual(input=self.dataset["input"], bitString=bitString)
            new_individual = self.createNewIndividual(bitString)
            self.individuals.append(new_individual)

    def createNewIndividual(self, bitString):
        newIndividual = Individual()
        newIndividual.bitString = bitString
        layers = ED.decode(bitString)
        last = self.input
        for layer in layers:
            last = Dense(units=layer.output, activation=layer.activation)(last)
        newIndividual.output = Dense(units=10, activation="softmax")(last)
        return newIndividual

    # Calculate fitness of all individuals
    def calculateFitness(self):
        
        outputs = [i.output for i in self.individuals]
        model = Model(inputs=self.input, outputs=outputs)

        model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=["accuracy"]
        )

        history = model.fit(self.dataset["train_x"], self.dataset["train_y"],
            validation_data=(self.dataset["test_x"], self.dataset["test_y"]),
            epochs=5, batch_size=600, verbose=0)

        index = 0
        for h in history.history:
            if h.startswith("val") and h.endswith("accuracy"):
                score = history.history[h][-1]*100
                self.individuals[index].fitness = score
                print(str(score) + " " + self.individuals[index].bitString)
                index += 1
        
        # Save the fittest score and fittest model
        self.individuals.sort(key=lambda i: i.fitness, reverse=True)
        self.fittest_model = self.individuals[0].bitString
        self.fittest = self.individuals[0].fitness

    def populationResult(self, generationCount):
        return "Generation: {}  Fittest Score: {} Fittest Model: {}".format(
            generationCount, self.fittest, self.fittest_model)