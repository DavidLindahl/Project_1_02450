# Exercise 4.2.4
# requires data from exercise 4.1.1

'''This makes m amount of boxplots for c classes
Explained: This makes boxplots for each class you have.
Great for comparing classes and to get a quick summary
visualisation if there is any differnces / correlations.'''
from dataloader import *
from matplotlib.pyplot import boxplot, figure, show, subplot, title, xticks, ylim

figure(figsize=(14, 7))
for c in range(C):
    subplot(1, C, c + 1)
    class_mask = y == c  # binary mask to extract elements of class c
    # or: class_mask = nonzero(y==c)[0].tolist()[0] # indices of class c

    boxplot(X[class_mask, :])
    # title('Class: {0}'.format(classNames[c]))
    title("Class: " + classNames[c-1])
    xticks(
        range(1, len(attributeNames) + 1), [a[:7] for a in attributeNames], rotation=45
    )
    y_up = X.max() + (X.max() - X.min()) * 0.1
    y_down = X.min() - (X.max() - X.min()) * 0.1
    ylim(y_down, y_up)

show()

print("Ran Exercise 4.2.4")
