# Exercise 4.2.3
'''This makes boxplots of the attributes.
The data is normalized'''
# requires data from exercise 4.2.1
from dataloader import *
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Set figure size
plt.figure(figsize=(12, 6))

# Set title
plt.title("Kidney Stone Prediction: Standardised Boxplots")

# Calculate z-scores
z_score_data = zscore(X, ddof=1)

# Define pastel colors
pastel_colors = ['#FFD1DC', '#FFA07A', '#FFD700', '#87CEEB', '#98FB98', '#FFB6C1', '#E6E6FA']

# Create boxplots
boxplot = plt.boxplot(z_score_data, patch_artist=True)

# Set boxplot properties
for patch, color in zip(boxplot['boxes'], pastel_colors):
    patch.set_facecolor(color)
    patch.set_linewidth(2)
    patch.set_linestyle('-')

# Set x-axis ticks and labels
plt.xticks(range(1, len(attributeNames) + 1), attributeNames, rotation=45)

# Add horizontal gridlines with a distance of 1 between lines
plt.grid(axis='y', linestyle='--', linewidth=0.5, color='gray')

# Save figure as .png
# plt.savefig('Standardized_Boxplot')

# Show plot
plt.show()

