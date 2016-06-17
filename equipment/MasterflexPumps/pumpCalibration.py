from Tkinter import *
import pumpDriver
import math

#Arduino Pump Driver Python Calibration GUI
#Created by Dean Massecar
#on 2013-06-18
#Contact: dam671@mun.ca or dean.massecar@gmail.com
#Chartered by Dr. James Munroe
#Contact: jmunroe@mun.ca
#Last updated 2013-06-21

#default values, will be close to actual, but needs cal
#format: (LPM/PWM unit, minFlow[LPM])
pumpCalMin = [0.63682, 0.705]
pumpCalSlope = [0.0260982, 0.027525164]
calComplete = False

#does calibration need to be set every time the program starts?
#or should last known calibration be put into a config file with the py code?

def make_cal_GUI():
    return True
    calWindow = Tk()
    calWindow.title("Arduino Pump Calibration GUI")
    calWindow.geometry('250x150+700+500')
    #call window features here
    calWindow.mainloop()

def cal_pump():
    #run through a series of values on the chosen pump, and get user
    #input
    pass
