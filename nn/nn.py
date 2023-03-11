from keras.models import Model
from keras.layers import Dense, Input
from individual import Individual
import encoder_decoder as ED

class NN:
    models = []

    def __init__(self, dataset):
        self.dataset = dataset
        self.input = Input(shape=(self.dataset["input"], ))

    def createModels(self, bitStrings):
        for bitString in bitStrings:
            newIndividual = Individual()
            newIndividual.bitString = bitString
            layers = ED.decode(bitString)
            last = self.input
            for layer in layers:
                last = Dense(units=layer.output, activation=layer.activation)(last)
            newIndividual.output = Dense(units=10, activation="softmax")(last)
            self.models.append(newIndividual)

    def calculateResults(self):
        outputs = [i.output for i in self.models]
        model = Model(self.input, outputs)

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
                self.models[index].fitness = history.history[h][-1]*100
                index += 1
        return self.models