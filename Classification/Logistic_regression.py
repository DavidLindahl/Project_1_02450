import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from dataloader import *

# Setup 10-fold cross-validation
K = 10
kf = KFold(n_splits=K, shuffle=True, random_state=1)

# Prepare to store error rates and coefficients for each fold and lambda value
lambda_interval = np.logspace(-8, 2, 200)
errors = np.zeros((K, len(lambda_interval)))
coefficient_norms = np.zeros((K, len(lambda_interval)))

# Arrays to store the optimal lambda and its error for each fold
optimal_lambs = []
errors_real = []

# Perform cross-validation
k = 0
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Standardize the training and test sets
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    # Initialize arrays to store errors for the current fold for each lambda
    fold_errors = np.zeros(len(lambda_interval))

    # Train models and calculate errors for each lambda
    for i, lambd in enumerate(lambda_interval):
        mdl = LogisticRegression(penalty='l2', C=1/lambd, solver='lbfgs', max_iter=10000)
        mdl.fit(X_train, y_train)

        # Calculate error rate
        y_test_est = mdl.predict(X_test)
        fold_errors[i] = np.sum(y_test_est != y_test) / len(y_test)

    # Find the optimal lambda for the current fold
    min_error_idx = np.argmin(fold_errors)
    opt_lambda = lambda_interval[min_error_idx]
    min_error = fold_errors[min_error_idx]

    # Store the optimal lambda and its error for the fold
    optimal_lambs.append(opt_lambda)
    errors_real.append(min_error)

    k += 1
# Print or use the optimal_lambs and errors_real as needed
print("Optimal lambdas per fold:", optimal_lambs)
print("Minimum errors per fold:", errors_real)

print('average error rate: ', np.mean(errors_real))
print('best lambda: ', optimal_lambs[np.argmin(errors_real)])
