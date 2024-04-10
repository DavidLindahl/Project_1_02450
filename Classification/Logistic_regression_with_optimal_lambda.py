import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from dataloader import *

# Setup 10-fold cross-validation
K = 10
kf = KFold(n_splits=K, shuffle=True, random_state=1)

# Now we already know the optimal lambda:
optimal_lambda = 2.2

# Prepare to store error rates and coefficients for each fold
errors = np.zeros(K)
coefficient_norms = np.zeros((K, X.shape[1]))

# Train models and calculate errors for each fold  
k = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    mdl = LogisticRegression(penalty='l2', C=1/optimal_lambda, solver='lbfgs', max_iter=10000)
    mdl.fit(X_train, y_train)

    # Calculate error rate
    y_test_est = mdl.predict(X_test)
    errors[k] = np.sum(y_test_est != y_test) / len(y_test)

    # Store coefficients
    coefficient_norms[k, :] = np.linalg.norm(mdl.coef_, axis=0)

    k += 1

median_coefficient_norms = np.median(coefficient_norms, axis=0)

# Display results
print('Error rates:', errors)
print('Mean error rate:', np.mean(errors))
print('Standard deviation of error rates:', np.std(errors))
print('Median coefficient norms:', median_coefficient_norms)
print('Coefficient norms:', coefficient_norms)


# Plot the coefficients
plt.figure()
plt.boxplot(coefficient_norms)
plt.xlabel('Attribute index')
plt.ylabel('Attribute weight coefficient')
plt.title('Coefficient norms')

plt.savefig('coefficient_norms.png')
plt.show()


