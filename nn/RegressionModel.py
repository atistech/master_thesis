from nn.Model import Model
from keras import metrics
import nn.Datasets as Datasets

class RegressionModel(Model):
    def __init__(self, isRandom, layers, dataset):
        #self.dataset = Datasets.readCSVDataset(os.getcwd()+"/nn/sample_dataset1.csv")
        self.dataset = Datasets.readCSVDataset(dataset)
        super().__init__(isRandom, layers, self.dataset)

    def calculateResult(self):
        self.model.compile(
            optimizer=self.optimizer,
            loss="binary_crossentropy",
            metrics=[metrics.MeanAbsolutePercentageError(), 
                     metrics.MeanSquaredError(),
                     metrics.MeanAbsoluteError()]
        )

        history = self.model.fit(
            self.dataset["x"], self.dataset["y"],
            validation_split=0.2,
            epochs=self.epochs, 
            batch_size=self.batchSize, 
            verbose=0
        )

        self.loss = float("%.2f" % history.history['loss'][0])
        self.val_loss = float("%.2f" % history.history['val_loss'][0])
        self.mape = float("%.2f" % (history.history['mean_absolute_percentage_error'][0]))
        self.val_mape = float("%.2f" % history.history['val_mean_absolute_percentage_error'][0])
        self.mse = float("%.2f" % (history.history['mean_squared_error'][0]))
        self.val_mse = float("%.2f" % history.history['val_mean_squared_error'][0])
        self.mae = float("%.2f" % (history.history['mean_absolute_error'][0]))
        self.val_mae = float("%.2f" % history.history['mean_absolute_error'][0])
        mean = (self.loss + self.val_loss + self.mape + self.val_mape + self.mse + self.val_mse)/6
        self.fitnessScore = float("%.2f" % mean)

    def serialize(self):
        return {
            'mape': self.mape,
            'val_mape': self.val_mape,
            'mse': self.mse,
            'val_mse': self.val_mse,
            'mae': self.mae,
            'val_mae': self.val_mae,
            'loss': self.loss,
            'val_loss': self.val_loss,
            'fitnessScore': self.fitnessScore,
            'optimizer': self.optimizer,
            'architecture': super().toString()
        }
    
