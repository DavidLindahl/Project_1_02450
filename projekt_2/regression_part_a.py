from data_loader import *
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
A = 100
K = len(y)
error_per_fold = np.zeros(K)
error_per_aplha = np.zeros(A)
i = 0
CV = model_selection.KFold(K, shuffle=True)
for alpha in np.linspace(0, 40, A):
    j = 0
    model = lm.Ridge(alpha=alpha)
    for train_index, test_index in CV.split(st_X, y):
        X_train = st_X[train_index]
        y_train = y[train_index]
        X_test = st_X[test_index]
        y_test = y[test_index]
        # Fit ordinary least squares regression model
        model.fit(X_train, y_train)

        # Predict y
        y_est = model.predict(X_test)
        error_per_fold[j] = mean_squared_error(y_est, y_test)
        j += 1
    error_per_aplha[i] = np.average(error_per_fold)
    i += 1

# Display scatter plot
plt.figure()
plt.plot(np.linspace(0, 40, A), error_per_aplha, ".")
plt.xlabel(f"alpha")
plt.ylabel(f"mse")
plt.savefig("plots/error_per_alpha.png")
plt.show()

index_of_lowest_error = np.where(error_per_aplha == min(error_per_aplha))[0]
best_alpha = np.linspace(0, 40, A)[index_of_lowest_error]
print(f"best alpha value is: {best_alpha}\nthe error at that alpha value is: {min(error_per_aplha)}")
model = lm.Ridge(alpha=best_alpha)
model.fit(st_X, y)
coefficients = model.coef_
print(coefficients, X.columns)