import numpy as np
import pumpDriver
from time import sleep

# User Parameters
densityFW = 0.9987    # kg/L
densitySW = 1.0592    # kg/L
depth = 0.5    # m
rate = 7    # L/min - estimated for 3hr fill
area = 2.5    # m^2
step = 0.01   # m

# Calibration Factors
cali1 = 0.0223
cali2 = 0.0258
cali_yint1 = 0.45
cali_yint2 = 0.42

# Linear Stratification Parameters
slope = -(depth/(densitySW - densityFW))
yint = (slope * (0 - densityFW) + depth)    # m

# Misc Calculations
volume = depth * area * 1000    # L
total_steps = (depth / step)

# Time Calculations
time_total = volume / rate    # minutes
time_step = time_total / total_steps    # minutes

class Pump1(object):

    def __init__(self, stateFW, rateFW):
        self.stateFW = stateFW
        self.rateFW = rateFW

    def get_stuff(self):
        return self.stateFW
        return self.rateFW

    def set_stuff(self, stateFW, rateFW):
        self.stateFW = stateFW
        self.rateFW = rateFW

class Pump2(object):

    def __init__(self, stateSW, rateSW):
        self.stateSW = stateSW
        self.rateSW = rateSW

    def get_stuff(self):
        return self.stateSW
        return self.rateSW

    def set_stuff(self, stateSW, rateSW):
        self.stateSW = stateSW
        self.rateSW = rateSW


# starts the communication
print("Initializing...")
pumpDriver.init()
sleep(2)

for z in np.arange(0, depth, step):

    try:
        density = (z - yint) / slope
        cSW = (density - densitySW) / (densityFW - densitySW)
        cFW = 1 - cSW
        rateFW = rate * cFW
        rateSW = rate * cSW
        rateFW = int((rateFW - cali_yint2) / cali2)
        rateSW = int((rateSW - cali_yint1) / cali1)

        pumpDriver.set_speed(1, rateSW)
        pumpDriver.set_state(1, 1)
        pumpDriver.set_speed(2, rateFW)
        pumpDriver.set_state(2, 1)

        time_for_step = ((area * z * 1000) / rate)
        time_remaining = (volume / rate) - time_for_step

        print "Fresh water rate: %d" % rateFW
        print "Salt water rate: %d" % rateSW
        print "Time remaining: %.1f min" %  time_remaining
        print

        sleep(time_step*60)
    except KeyboardInterrupt:
        break

print("Turn off")
# turn off pumps
pumpDriver.set_state(1, 0) # 0 for off, 1 for on
pumpDriver.set_state(2, 0)

# stops the communication
pumpDriver.close()

