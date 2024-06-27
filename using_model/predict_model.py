import pandas as pd
import keras

df = pd.read_csv("C:/Users/bluea/OneDrive/Masaüstü/sample_regression_dataset.csv")
#df = pd.read_csv("C:/Users/bluea/OneDrive/Masaüstü/sample_binary_classification_dataset.csv")
input_heads = [c for c in df.columns if c.startswith('x')]
inputs = df[input_heads].to_numpy()
output_heads = [c for c in df.columns if c.startswith('y')]
outputs = df[output_heads].to_numpy()

model = keras.models.load_model("bestModel.keras")
results = model.predict(inputs)
print(f"Target Value: {outputs[0]} and Predicted Value: {results[0]}")