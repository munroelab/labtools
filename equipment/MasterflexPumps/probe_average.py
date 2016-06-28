import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

data = pd.read_csv('cup1.txt', skiprows=6, sep=' ', names =
        ['date', 'time', 'V'])

data = data.V.dropna()

data_mean = np.mean(data)

print(data_mean)

