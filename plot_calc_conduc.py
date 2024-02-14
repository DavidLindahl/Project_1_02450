from dataloader import *
import matplotlib.pyplot as plt

# Plot 1: Plotting calcium vs. cond:
calc = X[:,5]
conductivity = X[:,3]

# Pynt
plt.xlabel('Calcium // mg/ml')
plt.ylabel('Conductivity // #indsæt værdi')
plt.title('Conductivity vs. Calcium')

# Plot
plt.scatter(calc , conductivity)
plt.show()
