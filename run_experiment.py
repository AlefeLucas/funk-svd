import pandas as pd
import numpy as np

from funk_svd.dataset import fetch_ml_ratings
from funk_svd.dataset import VARIANTS
from funk_svd import SVD

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt

variant = 'bx'

print("Obtaining data...")
df = fetch_ml_ratings(variant=variant)
print("Sampling train and test...")
train = df.sample(frac=0.8, random_state=7)
val = df.drop(train.index.tolist()).sample(frac=0.5, random_state=8)
test = df.drop(train.index.tolist()).drop(val.index.tolist())

svd = SVD(learning_rate=0.001, regularization=0.005, n_epochs=100,
          n_factors=15, variant=variant)

print("Training...")
svd.fit(X=train, X_val=val, early_stopping=True, shuffle=False)

pred = svd.predict(test)
mae = mean_absolute_error(test["rating"], pred)
mse = mean_squared_error(test["rating"], pred)
rmse = sqrt(mse)

print(f'Test MAE: {mae:.2f}')
print(f'Test MSE: {mse:.2f}')
print(f'Test RMSE: {rmse:.2f}')

