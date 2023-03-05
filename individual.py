from keras.models import Sequential
from keras.layers import Input, Dense

class Model():
    # Model() constructor
    def __init__(self, layerNum, layers, optimizer):
        self.layerNum = layerNum
        self.layers = layers
        self.optimizer = optimizer

class Layer():
    # Layer() constructor
    def __init__(self, name, output, activation):
        # Define parameters of layer
        self.name = name
        self.output = output
        self.activation = activation

class Individual():
    # Define fitness score as 0
    fitness = 0
    bitString = ""

    # Individual() constructor
    def __init__(self, input, bitString, model):
        self.bitString = bitString
        # Define a sequential model
        self.model = Sequential()
        # Add input layer to model
        self.model.add(Input(shape=(input,)))

        # Add all layers to model
        for layer in model.layers:
            self.model.add(Dense(units=layer.output, activation=layer.activation))
        self.model.add(Dense(10, activation="softmax"))

        # Configures the model for training
        self.model.compile(optimizer=model.optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Calculate Fitness Score
    def calculateFitness(self, dataset):
        # Train the model
        history = self.model.fit(dataset["train_x"], dataset["train_y"], 
            validation_data=(dataset["test_x"], dataset["test_y"]), 
            epochs=5, batch_size=600, verbose=0)
        # Set last accuracy of the model as fitness score
        self.fitness = int(history.history["val_accuracy"][-1]*100)