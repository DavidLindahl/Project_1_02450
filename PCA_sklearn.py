# Step 1: Import necessary libraries
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Step 2: Prepare your data
# Let's assume your data is in a variable named 'data'
# data = ...
file_path = '/Users/m.brochlips/Documents/2 MachLearn DATA/02450Toolbox_Python/Scripts/kindey stone urine analysis.csv'
data = pd.read_csv(file_path)

X = data.drop('target', axis=1)
y = data['target']

# It's important to standardize your data before applying PCA
# scaler = StandardScaler()
# standardized_data = scaler.fit_transform(data)
scalar = StandardScaler()
st_X = scalar.fit_transform(X)

# Step 3: Initialize PCA
# Decide how many principal components you want. For demonstration, let's use 2.
# pca = PCA(n_components=2)
pca = PCA()

# Step 4: Fit PCA on your standardized data
# pca.fit(standardized_data)
# Step 5: Transform the data to the new PCA space
# transformed_data = pca.transform(standardized_data)
trans_data = pca.fit_transform(st_X)

print("Ratio:")
print(pca.explained_variance_ratio_)
print("Principal components (each row corresponds to a component, each column to a feature):")
print(pca.components_)

