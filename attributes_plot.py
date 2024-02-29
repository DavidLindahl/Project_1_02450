import matplotlib.pyplot as plt
import seaborn as sns
from dataloader import *

train_data = data
plt.figure(figsize=(12,10))
sns.pairplot(train_data,hue="target")
plt.title("Looking for Insights in Data")
plt.legend("target")
plt.tight_layout()
plt.plot()
plt.show()