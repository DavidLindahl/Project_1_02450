# Exercise 4.2.3
'''This makes boxplots of the attributes.
The data is not normalized yet.'''
# requires data from exercise 4.2.1
from dataloader import *
from matplotlib.pyplot import boxplot, show, title, xticks, ylabel

boxplot(X)
xticks(range(1, len(attributeNames)+1), attributeNames)
ylabel("cm")
title("Kidney Stone Prediction based on Urine Analysis - Boxplots")
show()

print("Ran Exercise 4.2.3")
