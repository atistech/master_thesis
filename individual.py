from keras.models import Sequential
from keras.layers import Input, Dense

class Layer():
    # Layer() constructor
    def __init__(self, output, activation):
        # Define parameters of layer
        self.output = output
        self.activation = activation

class Individual():
    # Define fitness score as 0
    fitness = 0

    # Individual() constructor
    def __init__(self, input, layers, optimizer, epochs):
        # Define parameters of model architecture
        self.input = input
        self.layers = layers
        
        # Define parameters of compile function
        self.optimizer = optimizer
        self.loss = "categorical_crossentropy"
        self.metrics = ["accuracy"]

        # Define parameters of fit function
        self.epochs = epochs
        self.batchSize = 600

        # Define a sequential model
        self.model = Sequential()
        # Add input layer to model
        self.model.add(Input(shape=(self.input,)))

        # Add all layers to model
        for layer in self.layers:
            self.model.add(Dense(layer.output, activation=layer.activation))

        # Configures the model for training
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    # Calculate Fitness Score
    def calcFitness(self, dataset):
        # Define x and y of dataset
        trainX = dataset["train_x"]
        trainY = dataset["train_y"]
        valX = dataset["test_x"]
        valY = dataset["test_y"]
        # Train the model
        history = self.model.fit(trainX, trainY, validationX=valX, validationY=valY,epochs=self.epochs, batch_size=self.batchSize, verbose=0)
        # Set last accuracy of the model as fitness score
        self.fitness = int(history.history["val_accuracy"][-1]*100)
        # Show 
        print("Fitness: {} Model: {}".format(self.fitness, self.toString()))

    # Convert model of individual to string
    def toString(self):
        # Example: 784&dense_10_relu&adam/categorical_crossentropy&10
        layer_str = ""
        for l in self.layers:
            layer_str += "dense_{}_{}/".format(l.output, l.activation)
        return "{}&{}&{}/{}&{}".format(self.input, layer_str, self.optimizer, self.loss, self.epochs)