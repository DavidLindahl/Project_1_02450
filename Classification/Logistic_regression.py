import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import cross_val_score, StratifiedKFold
from dataloader import *

# Define the logistic regression model
model = lm.LogisticRegression()

# Define the outer cross-validation
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists to store accuracy and misclassification rate
accuracy_list = []
misclass_rate_list = []

# Outer cross-validation loop
for train_index, test_index in outer_cv.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Inner cross-validation for hyperparameter tuning
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=inner_cv, scoring='accuracy')
    
    # Fit the model on the training set with tuned hyperparameters
    model.fit(X_train, y_train)
    
    # Evaluate the model on the test set
    y_est = model.predict(X_test)
    
    # Calculate accuracy and misclassification rate
    accuracy = scores.mean()
    misclass_rate = np.sum(y_est != y_test) / float(len(y_est))
    
    # Append results to lists
    accuracy_list.append(accuracy)
    misclass_rate_list.append(misclass_rate)

# Display average accuracy and misclassification rate across folds
print("Average Accuracy:", np.mean(accuracy_list))
print("Average Misclassification Rate:", np.mean(misclass_rate_list))
