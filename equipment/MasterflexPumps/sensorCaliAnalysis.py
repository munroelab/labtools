%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.plot(density, sensorReading)

slope, intercept = np.polyfit(np.array(density), np.array(sensorReading),1)

print(1/slope)
print(1/intercept)


