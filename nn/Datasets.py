from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Dense, Input

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
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y
    }