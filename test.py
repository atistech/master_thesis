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
        temp = Dense(deneme[i], name=x+str(i))(last)
        last = temp
    return Dense(1, name=x+str(3))(last)

outputs = []
for i in range(10):
    outputs.append(deneme(str(i)+"-"))

model = Model(inputs=main_input,
              outputs=outputs,
)

model.compile(optimizer='adam',
             loss='mean_squared_error'
             )

model.summary()



history = model.fit(dataset["train_x"], dataset["train_y"], 
            epochs=5, batch_size=600, verbose=0)

print(history.history)