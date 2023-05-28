from keras import Sequential
from keras.layers import Input, Dense
import nn.Datasets as datasets

train_x, train_y, test_x, test_y = [], [], [], []

model = Sequential()
model.add(Input(shape=(784, )))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=64, activation='selu'))
model.add(Dense(units=256, activation='softplus'))
model.add(Dense(units=16, activation='sigmoid'))
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

dataset = datasets.MnistDataset()

history = model.fit(dataset["train_x"], dataset["train_y"], validation_data=(dataset["test_x"], dataset["test_y"]), epochs=1, batch_size=32, verbose=0)

print(history.history)
print(type(history.history['loss']))
print(history.history[1])
print(history.history[2])
print(history.history[3])