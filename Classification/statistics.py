import numpy as np
from dtuimldmtools.statistics.statistics import correlated_ttest
# ANN test errors
ann_errors = np.array([0.38, 0.38, 0.25, 0.38, 0.5, 0.25, 0.12, 0.12, 0.5, 0.0])

# Logistic regression test errors
logistic_regression_errors = np.array([0.0, 0.25, 0.5, 0.38, 0.38, 0.25, 0.12, 0.12, 0.12, 0.14])

# Baseline test errors
baseline_errors = np.array([0.25, 0.62, 0.75, 0.5, 0.38, 0.62, 0.12, 0.12, 0.38, 0.57])

# ann_errors, logistic_regression_errors, baseline_errors

def find_p_and_ci(error_array1, error_array2):
    '''Takes 2 error arrays and calculates the p-value and confidence interval
     for the difference in generalization error per split between the two models.'''
    

    # Initialize the difference in error array
    r = np.zeros(10)
    # Calculate the difference in error for each split
    
    for i in range(10):
        if i != 9:
            r[i] = 1/8 * (error_array1[i] - error_array2[i])
        else:
            r[i] = 1/7 * (error_array2[i] - error_array2[i])

    # Get the CI and the p-value
    alpha = 0.05
    rho = 1/10
    p, CI = correlated_ttest(r, rho, alpha=alpha)

    # Display the values
    print(f"The p-value is: {p}")
    print(f"The confidence interval is: {CI}")

    return p, CI

# Now for getting the 3 p-values and confidence intervals

print("ANN vs Logistic Regression")
find_p_and_ci(ann_errors, logistic_regression_errors)

print("ANN vs Baseline")
find_p_and_ci(ann_errors, baseline_errors)

print("Logistic Regression vs Baseline")
find_p_and_ci(logistic_regression_errors, baseline_errors)











