import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Loading Data
filepath = os.path.join('kindey_stone_urine_analysis.csv')
data = pd.read_csv(filepath)

# Names our data and making y (our prediction variable)
X = data.values
y = X[:,-1]
attributeNames = data.columns.values.tolist()

# Making the variables from the book
N = X.shape[0]
M = X.shape[1]

# Making classLabels
classLabels = []
classNames = ['No kidney stone' , 'Kidney stone']
classDict = {0:classNames[0] , 1:classNames[1]}

for value in y:
    label = classDict.get(value)
    classLabels.append(label)

C = len(attributeNames)


print('data loaded')


