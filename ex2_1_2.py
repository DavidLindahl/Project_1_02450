# exercise 2.1.2

# Imports the numpy and xlrd package, then runs the ex2_1_1 code
from dataloader import *
from matplotlib.pyplot import figure, legend, plot, show, title, xlabel, ylabel

# (requires data structures from ex. 2.1.1)


# Data attributes to be plotted
i = 5
j = 3

##
# Make a simple plot of the i'th attribute against the j'th attribute
# Notice that X is of matrix type (but it will also work with a numpy array)
# X = np.array(X) #Try to uncomment this line
plot(X[:, i], X[:, j], "o")
show()
'''This plot is for beginners "Gravity" plottet against "pH". 
To change what you want to plot against eachother, change i and j'''
# %%
# Make another more fancy plot that includes legend, class labels,
# attribute names, and a title.
f = figure()
title("Kidney Stone Prediction data")
C = len(attributeNames)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y == c
    plot(X[class_mask, i], X[class_mask, j], "o", alpha=0.3)

legend(attributeNames)
xlabel(attributeNames[i])
ylabel(attributeNames[j])

# Output result to screen
show()
print("Ran Exercise 2.1.2")

# %%
