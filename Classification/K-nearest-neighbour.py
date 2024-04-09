# exercise 6.3.1

import importlib_resources
from matplotlib.pyplot import (
    colorbar,
    figure,
    imshow,
    plot,
    show,
    title,
    xlabel,
    xticks,
    ylabel,
    yticks,
    legend,
)
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import os
from dataloader import *
from sklearn.model_selection import train_test_split


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Plot the training data points (color-coded) and test data points.
figure(1)
styles = [".b", ".r", ".g", ".y"]
for c in range(C):
    class_mask = y_train == c
    plot(X_train[class_mask, 0], X_train[class_mask, 1], styles[c])

# K-nearest neighbors
K = 8

# Distance metric (corresponds to 2nd norm, euclidean distance).
# You can set dist=1 to obtain manhattan distance (cityblock distance).
dist = 2
metric = "euclidean"
metric_params = {}  # no parameters needed for minkowski

# You can set the metric argument to 'cosine' to determine the cosine distance
# metric = 'cosine'
# metric_params = {} # no parameters needed for cosine

# To use a mahalonobis distance, we need to input the covariance matrix, too:
# metric='mahalanobis'
# metric_params={'V': cov(X_train, rowvar=False)}

# Fit classifier and classify the test points
knclassifier = KNeighborsClassifier(
    n_neighbors=K, p=dist, metric=metric, metric_params=metric_params
)
knclassifier.fit(X_train, y_train)
y_est = knclassifier.predict(X_test)


# Plot the classfication results
styles = ["ob", "or", "og", "oy"]
for c in range(C):
    class_mask = y_est == c
    plot(X_test[class_mask, 0], X_test[class_mask, 1], styles[c], markersize=10)
    plot(X_test[class_mask, 0], X_test[class_mask, 1], "kx", markersize=8)
legend(classNames)

title("Data classification - KNN")

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_est)
accuracy = 100 * cm.diagonal().sum() / cm.sum()
error_rate = 100 - accuracy
figure(2)
imshow(cm, cmap="binary", interpolation="None")
colorbar()
xticks(range(C))
yticks(range(C))
xlabel("Predicted class")
ylabel("Actual class")
title(
    "Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)".format(accuracy, error_rate)
)

show()

print("Ran Exercise 6.3.1")
