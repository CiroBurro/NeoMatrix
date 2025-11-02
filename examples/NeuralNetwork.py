from neomatrix.core import model, Tensor, Layer, Cost, optimizer, Activation
import pandas as pd

df = pd.read_csv("datasets/sinx.csv")
df_val = pd.read_csv("datasets/sinx_val.csv")

# --- Data Normalization ---
# Calculate mean and std from the training set
mean_x = df['X'].mean()
std_x = df['X'].std()
mean_y = df['Y'].mean()
std_y = df['Y'].std()

# Apply scaling
df_x_scaled = (df['X'] - mean_x) / std_x
df_val_x_scaled = (df_val['X'] - mean_x) /std_x

df_y_scaled = (df['Y'] - mean_y) / std_y
df_val_y_scaled = (df_val['Y'] - mean_y) /std_y

num_training_samples = len(df)
num_val_samples = len(df_val)


# Instance training and validation tensors
training_x = Tensor([num_training_samples], df_x_scaled.tolist())
training_y = Tensor([num_training_samples], df_y_scaled.tolist())

val_x = Tensor([num_val_samples], df_val_x_scaled.tolist())
val_y = Tensor([num_val_samples], df_val_y_scaled.tolist())


# Instant a neural network model
layers = [
    Layer(32, 1, Activation.Relu),
    Layer(32,32,Activation.Relu),
    Layer(1,32,Activation.Tanh)
]
model = model.NeuralNetwork(layers, Cost.MeanSquaredError(), learning_rate=0.01)

# Training
model.fit(training_set=training_x, training_targets=training_y, val_set=val_x, val_targets=val_y, optimizer=optimizer.MiniBatchGD(32, 32), epochs=300, patience=10, parallel=True)
model.save("sinx.json")