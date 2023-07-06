from nn.Model import Model
import nn.Datasets as Datasets

class ClassificationModel(Model):

    def __init__(self, isRandom, layers, dataset):
        self.dataset = Datasets.MnistDataset()
        super().__init__(isRandom, layers, dataset)

    def calculateResult(self):
        self.model.compile(
            optimizer=self.optimizer,
            loss="categorical_crossentropy", 
            metrics=["accuracy"]
        )

        history = self.model.fit(
            self.dataset["x"], self.dataset["y"],
            validation_split=0.2,
            epochs=self.epochs, 
            batch_size=self.batchSize, 
            verbose=0
        )

        self.loss = float("%.2f" % history.history['loss'][0])
        self.accuracy = float("%.2f" % (history.history['accuracy'][0]*100))
        self.val_loss = float("%.2f" % history.history['val_loss'][0])
        self.val_accuracy = float("%.2f" % (history.history['val_accuracy'][0]*100))
        self.fitnessScore = float("%.2f" % ((self.accuracy+self.val_accuracy)/2))

    def serialize(self):
        return {
            'accuracy': self.accuracy,
            'val_accuracy': self.val_accuracy,
            'average_accuracy': self.fitnessScore,
            'loss': self.loss,
            'val_loss': self.val_loss,
            'optimizer': self.optimizer,
            'architecture': super().toString()
        }