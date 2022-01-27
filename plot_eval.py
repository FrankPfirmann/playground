from unicodedata import decimal
import matplotlib.pyplot as plt
import numpy as np 

data = np.loadtxt("./eval-linear.csv", skiprows=1, delimiter=";")

threshold=data[:, 0]
accuracy=data[:, 1]
precision=data[:, 2]
recall=data[:, 3]
f1= 2 * (precision*recall) / (precision+recall)

plt.plot(threshold, accuracy, label="Accuracy")
plt.plot(threshold, precision, label="Precision")
plt.plot(threshold, recall, label="Recall")
plt.plot(threshold, f1, label="F1-Score")
plt.xlabel('Threshold')
plt.legend()
plt.show()