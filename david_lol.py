from dataloader import *

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# split data into X and y

# X = train_data.drop(['target'], axis=1)
# y = train_data['target'] 

# split data into train and test sets
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.24, random_state=7)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: {}%".format(round(accuracy * 100.0 ,2),))




