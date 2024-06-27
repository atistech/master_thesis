import keras

model = keras.models.load_model("bestModel.keras")
model.summary()

keras.utils.plot_model(model, to_file="model.png", show_shapes=True)