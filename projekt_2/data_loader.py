from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os


# Step 2: Prepare your data
# Let's assume your data is in a variable named 'data'
# data = ...
file_path = os.path.join("kindey_stone_urine_analysis.csv")
data = pd.read_csv(file_path)
predicted_vaiable = "calc"
X = data.drop([predicted_vaiable], axis=1)
y = data[predicted_vaiable]

y = np.array(y).reshape(-1, 1)
# It's important to standardize your data before applying PCA
# scaler = StandardScaler()
# standardized_data = scaler.fit_transform(data)
scalar = StandardScaler()
st_X = scalar.fit_transform(X)
st_y = scalar.fit_transform(y)