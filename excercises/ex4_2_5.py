# Exercise 4.2.5
'''This is the m x m plots.
Explained: This plots all of your attributes against eachother.
This can be used to get a visualisation of ALL attributes, whereafter
you can compare the plots and see, which most likely has a chance of 
succes with linear regression.'''
# requires data from exercise 4.2.1
from dataloader import *
from matplotlib.pyplot import (
    figure,
    legend,
    plot,
    show,
    subplot,
    xlabel,
    xticks,
    ylabel,
    yticks,
)

figure(figsize=(12, 10))
for m1 in range(M):
    for m2 in range(M):
        subplot(M, M, m1 * M + m2 + 1)
        for c in range(C):
            class_mask = y == c
            plot(np.array(X[class_mask, m2]), np.array(X[class_mask, m1]), ".")
            if m1 == M - 1:
                xlabel(attributeNames[m2])
            else:
                xticks([])
            if m2 == 0:
                ylabel(attributeNames[m1])
            else:
                yticks([])
            # ylim(0,X.max()*1.1)
            # xlim(0,X.max()*1.1)
legend(classNames)

show()

print("Ran Exercise 4.2.5")
