
from matplotlib import pyplot as plt
import numpy as np

xAxis = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
yAxis = [8.228058,4.382240,3.053931,2.417429,2.043835,2.708664,2.346369,2.056148,2.137956,1.969407,1.975135,2.859916,3.559919,3.126562,4.116651, 3.851177,4.450000,5.043340,4.659124,5.159991]

fig, ax = plt.subplots()
ax.plot(xAxis, yAxis)
plt.xlabel('Number of workers')
plt.ylabel('Execution Time')
plt.title('Execution time as a function of the number of workers')
ax.set_xticks(np.arange(0, 21, 1))
plt.show()