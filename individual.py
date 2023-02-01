from keras.models import Sequential
from keras.layers import Input, Dense

class Individual():
    fitness = 0

    genes_for_layers = []
    genes_for_fit_params = []

    def __init__(self, model_text):
        self.model_text = model_text
        parser = self.model_text_parser(model_text)
        self.input = parser[0]
        self.layers = parser[1]
        self.compile_params = parser[2]
        self.fit_params = parser[3]

        self.model = Sequential()
        self.model.add(Input(shape=(self.input,)))

        for layer in self.layers:
            layer_type = layer.split("_")[0]
            output = int(layer.split("_")[1])
            activation = layer.split("_")[2]
            
            if layer_type=="dense":
                self.model.add(Dense(output, activation=activation))

        optimizer = self.compile_params[0]
        loss = self.compile_params[1]
        self.model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    # Calculate Fitness Score
    def calcFitness(self, dataset):
        trainX = dataset["train_x"]
        trainY = dataset["train_y"]
        epochs = int(self.fit_params[0])
        history = self.model.fit(trainX, trainY, epochs=epochs, batch_size=128, verbose=0)
        self.fitness = int(history.history["accuracy"][-1]*100)
        print("Fitness: {} Model: {}".format(self.fitness, self.model_text))

    def model_text_parser(self, text):
        #Exp. 784&dense_10_relu&adam/categorical_crossentropy&10
        input = int(text.split("&")[0])
        layers = text.split("&")[1].split("/")
        compile_params= text.split("&")[2].split("/")
        fit_params= text.split("&")[3].split("/")
        return (input, layers, compile_params, fit_params)