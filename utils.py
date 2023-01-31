from keras.models import Sequential
from keras.layers import Input, Dense

def sample_model_texts():
    samples = []
    activations1 = ["relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu"]
    activations2 = ["relu", "sigmoid"]
    optimizers = ["rmsprop", "adam"]

    for activation in activations2:
        for optimizer in optimizers:
            samples.append("dense_10_{}&784/{}/categorical_crossentropy/5"
                .format(activation, optimizer))
    return samples

def text_to_model(model_text):
    layers_part = model_text.split("&")[0]
    config_part = model_text.split("&")[1]
    model_input = int(config_part.split("/")[0])
    
    model = Sequential()
    model.add(Input(shape=(model_input,)))

    for layer_part in layers_part.split("/"):
        layer_part_list = layer_part.split("_")
        layer_type = layer_part_list[0]
        output = int(layer_part_list[1])
        activation = layer_part_list[2]
            
        if layer_type=="dense":
            model.add(Dense(output, activation=activation))

    optimizer = config_part.split("/")[1]
    loss = config_part.split("/")[2]
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    
    return model

def model_run(dataset, model_text, model):
    print("The {} has started.".format(model_text))
    epochs = int(model_text.split("&")[1].split("/")[3])
    history = model.fit(dataset["train_x"], dataset["train_y"], epochs=epochs, batch_size=128, verbose=0)
    lastAccuracy = history.history["accuracy"][-1]
    #model_dict.accuracy_results.append(round(lastAccuracy*100,2))
    print("The last training accuracy of {} is {:.0%}.".format(model_text, lastAccuracy))