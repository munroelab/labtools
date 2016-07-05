
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

data_name = pd.read_csv('exp160620222758.txt', skiprows=6, sep=' ', names =
        ['date', 'time', 'steps', 'z', 'V'])

slope = -0.00649151
intercept = 1.02819
data_name.rho = slope * data_name.V + intercept
plt.plot(data4.rho, data4.z)

slope, intercept = np.polyfit(np.array(data_name.rho.dropna()),
        np.array(data_name.z.dropna()),1)
print(1/slope)
print(1/intercept)
print(math.sqrt((1/slope)*980))

