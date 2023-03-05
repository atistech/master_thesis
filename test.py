import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import Input, Dense
from keras.models import Model
import datasets

dataset = datasets.MnistDataset()
main_input = Input(shape=(dataset["input"], ))


def deneme(x):
    deneme = [10, 10, 10]
    last = main_input
    for i in range(3):
        last = Dense(deneme[i])(last)
    return Dense(10, activation="softmax")(last)

outputs = []
for i in range(10):
    outputs.append(deneme(str(i)+"-"))

model = Model(inputs=main_input,
              outputs=outputs,
)

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=["accuracy"]
             )

#model.summary()



history = model.fit(dataset["train_x"], dataset["train_y"], 
            epochs=5, batch_size=600, verbose=0)

for key,value in enumerate(history.history):
    if value.endswith("accuracy"):
        print(str(history.history[value][-1]*100))