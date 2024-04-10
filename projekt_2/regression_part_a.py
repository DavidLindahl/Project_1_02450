from data_loader import *
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
A = 100
K = 10
K = 10
error_per_fold = np.zeros(K)
error_per_aplha = np.zeros(A)
i = 0
CV = model_selection.KFold(K, shuffle=False)
CV = model_selection.KFold(K, shuffle=False)
for alpha in np.linspace(0, 40, A):
    j = 0
    model = lm.Ridge(alpha=alpha)
    for train_index, test_index in CV.split(st_X_reg, y_reg):
        X_train = st_X_reg[train_index]
        y_train = y_reg[train_index]
        X_test = st_X_reg[test_index]
        y_test = y_reg[test_index]
        # Fit ordinary least squares regression model
        model.fit(X_train, y_train)

        # Predict y
        y_est = model.predict(X_test)
        error_per_fold[j] = mean_squared_error(y_test, y_est)
        j += 1
    error_per_aplha[i] = np.sum(error_per_fold)
    i += 1

# Display scatter plot
plt.figure()
plt.plot(np.linspace(0, 40, A), error_per_aplha, ".-")
plt.xlabel(f"alpha")
plt.ylabel(f"mse")
plt.savefig("plots/error_per_alpha.png")
plt.show()

index_of_lowest_error = np.where(error_per_aplha == min(error_per_aplha))[0]
best_alpha = np.linspace(0, 40, A)[index_of_lowest_error]
print(f"best alpha value is: {best_alpha}\nthe error at that alpha value is: {min(error_per_aplha)}")
model = lm.Ridge(alpha=best_alpha)
model.fit(st_X_reg, y_reg)
coefficients = model.coef_
print(coefficients, X.columns)