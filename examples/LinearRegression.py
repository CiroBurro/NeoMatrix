from neomatrix.core import Tensor, model, optimizer
import pandas as pd

df = pd.read_csv("datasets/linearreg.csv")
df_val = pd.read_csv("datasets/linearreg_val.csv")

# --- Data Normalization ---
# Calculate mean and std from the training set
mean_x = df[['impegno', 'cfu']].mean()
std_x = df[['impegno', 'cfu']].std()
mean_y = df['risultato'].mean()
std_y = df['risultato'].std()

# Apply scaling
df_x_scaled = (df[['impegno', 'cfu']] - mean_x) / std_x
df_val_x_scaled = (df_val[['impegno', 'cfu']] - mean_x) / std_x

df_y_scaled = (df['risultato'] - mean_y) / std_y
df_val_y_scaled = (df_val['risultato'] - mean_y) / std_y

num_training_samples = len(df)
num_val_samples = len(df_val)

# Instance training and validation tensors
training_x = Tensor([num_training_samples], df_x_scaled["impegno"].tolist())
t1 = Tensor([num_training_samples], df_x_scaled["cfu"].tolist())
training_x.push_column(t1)
training_y = Tensor([num_training_samples], df_y_scaled.tolist())


val_x = Tensor([num_val_samples], df_val_x_scaled["impegno"].tolist())
t2 = Tensor([num_val_samples], df_val_x_scaled["cfu"].tolist())
val_x.push_column(t2)
val_y = Tensor([num_val_samples], df_val_y_scaled.tolist())

# Instant a linear regression model
model = model.LinearRegression(2,1,0.01)

# Training
model.fit(training_set=training_x, training_targets=training_y, val_set=val_x, val_targets=val_y, optimizer=optimizer.MiniBatchGD(training_batch_size=16, validation_batch_size=16), epochs=500, patience=10)
model.save("model.json")