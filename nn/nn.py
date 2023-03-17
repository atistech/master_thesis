from keras.models import Model
from keras.layers import Dense, Input
from individual import Individual as EntityModel
import nn.nn_datasets as nn_datasets

class NN:
    def __init__(self, datasetIndex):
        if(datasetIndex == 0):
            self.dataset = nn_datasets.MnistDataset()
        self.input = Input(shape=(self.dataset["input"], ))

    def createModels(self, bitStrings):
        newModels = []
        for bitString in bitStrings:
            newModel = EntityModel(bitString)
            last = self.input
            for layer in newModel.layers:
                last = Dense(units=layer.output, activation=layer.activation)(last)
            newModel.output = Dense(units=10, activation="softmax")(last)
            newModels.append(newModel)
        return newModels

    def calculateResults(self, models):
        outputs = [i.output for i in models]
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
                models[index].fitness = history.history[h][-1]*100
                index += 1
        return models