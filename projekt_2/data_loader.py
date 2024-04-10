from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os


# Step 2: Prepare your data
# Let's assume your data is in a variable named 'data'
# data = ...
file_path = os.path.join("kindey_stone_urine_analysis.csv")
data = pd.read_csv(file_path)
data = data.sample(frac = 1)
predicted_vaiable = "calc"
X = data.drop([predicted_vaiable], axis=1)
y_reg = data[predicted_vaiable]

y_reg = np.array(y_reg).reshape(-1, 1)
scalar = StandardScaler()
st_X_reg = scalar.fit_transform(X)

data = pd.read_csv(file_path)
X = data.drop(["target"], axis=1)
y_class = data["target"]
st_X_class = scalar.fit_transform(X)
y_class = np.array(y_class).reshape(-1, 1)


