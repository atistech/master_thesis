from nn.Model import Model
from keras import metrics
import nn.Datasets as Datasets

class RegressionModel(Model):
    def __init__(self, isRandom, layers):
        super().__init__(isRandom, layers)

    def calculateResult(self, dataset):
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[metrics.MeanAbsolutePercentageError(), 
                     metrics.MeanSquaredError(),
                     metrics.MeanAbsoluteError()]
        )

        history = self.model.fit(
            dataset["x"], dataset["y"],
            validation_split=0.2,
            epochs=5, 
            batch_size=600,
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
            'optimizer': "adam",
            'architecture': super().toString()
        }
    
