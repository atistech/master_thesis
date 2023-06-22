from keras.datasets import mnist, fashion_mnist
from keras.utils import to_categorical
from keras.layers import Dense, Input
import pandas as pd
import os

def MnistDataset():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x = train_x.reshape((60000, 28 * 28))
    train_x = train_x.astype('float32') / 255
    test_x = test_x.reshape((10000, 28 * 28))
    test_x = test_x.astype('float32') / 255
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    return {
        "input":  Input(shape=(28*28, )),
        "x": train_x,
        "y": train_y,
        "test_x": test_x,
        "test_y": test_y,
        "output": {
                    "activation": "softmax",
                    "output": 10
                }
    }

def FashionMnistDataset():
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
    train_x = train_x.reshape((60000, 28 * 28))
    train_x = train_x.astype('float32') / 255
    test_x = test_x.reshape((10000, 28 * 28))
    test_x = test_x.astype('float32') / 255
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    return {
        "input":  Input(shape=(28*28, )),
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
        "output": {
                    "activation": "softmax",
                    "output": 10
                }
    }

def readCSVDataset(file_path):
    df = pd.read_csv(file_path)
    input_heads = [c for c in df.columns if c.startswith('x')]
    output_heads = [c for c in df.columns if c.startswith('y')]

    inputs = df[input_heads].to_numpy()
    outputs = df[output_heads].to_numpy()
    
    #print(inputs.shape)
    #print(outputs.shape)

    return {
        "input":  Input(shape=(len(input_heads), )),
        "x": inputs,
        "y": outputs,
        "output": {
                    "activation": "sigmoid",
                    "output": len(output_heads)
                }
    }

#readCSVDataset(os.getcwd()+"/nn/sample_dataset1.csv")