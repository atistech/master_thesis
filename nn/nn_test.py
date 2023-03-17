from keras.models import Model
from keras.layers import Dense, Input

def createModel(input, layers):
    last = input
    for layer in layers:
        last = Dense(units=layer.output, activation=layer.activation)(last)
    return Dense(units=10, activation="softmax")(last)

def createNewLayerRandomly():


def calculateResults(input, dataset, models):
    outputs = [i.output for i in models]
    model = Model(input, outputs)

    model.compile(optimizer='adam',
        loss='categorical_crossentropy',
        metrics=["accuracy"]
    )

    history = model.fit(dataset["train_x"], dataset["train_y"],
        validation_data=(dataset["test_x"], dataset["test_y"]),
        epochs=5, batch_size=600, verbose=0)

    index = 0
    for h in history.history:
        if h.startswith("val") and h.endswith("accuracy"):
            models[index].fitness = history.history[h][-1]*100
            index += 1
    return models