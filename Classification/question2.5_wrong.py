# Now we will train a logistic regression model using 200 different values of lambda, and
# Find the value of lambda with the minium error:

# Import relevant packages
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from dataloader import *

# Prepare to store error rates and coefficients for each fold and lambda value
lambda_interval = np.logspace(-8, 2, 200)
errors = np.zeros(len(lambda_interval))
coefficient_norms = np.zeros((K, len(lambda_interval)))

# Split the set
X_train, X_test, y_train, y_test = train_test_split(X ,y ,test_size=0.10, random_state=int(np.random.rand()*100))

# Train models and calculate errors for each lambda
for i, lambd in enumerate(lambda_interval):
    mdl = LogisticRegression(penalty='l2', C=1/lambd, solver='lbfgs', max_iter=10000)
    mdl.fit(X_train, y_train)

    # Calculate error rate
    y_test_est = mdl.predict(X_test)
    errors[i] = np.sum(y_test_est != y_test) / len(y_test)

optimal_lambda = lambda_interval[np.argmin(errors)]
min_error = np.min(errors)

print('Optimal lambda:', optimal_lambda)
print('Error for optimal lambda:', min_error)

