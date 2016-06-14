#Python Arbitrary Flow Rate Generator
#Created by Dean Massecar
#on 2013-07-03
#Contact: dam671@mun.ca or dean.massecar@gmail.com
#Chartered by Dr. James Munroe
#Contact: jmunroe@mun.ca
#Last updated 2013-07-04

#This is purely for testing, generates random sin/cos/tan waves 
#for a given period


import math
import arbitraryFlowRate
import pumpCalibration

time1 = []
time2 = []
flow1 = []
flow2 = []

A1 = (pumpCalibration.pumpCalSlope[0] * 255 - pumpCalibration.pumpCalMin[0])/2
A2 = (pumpCalibration.pumpCalSlope[1] * 255 - pumpCalibration.pumpCalMin[1])/2
avg1 = (pumpCalibration.pumpCalSlope[0] * 255 + pumpCalibration.pumpCalMin[0])/2
avg2 = (pumpCalibration.pumpCalSlope[1] * 255 + pumpCalibration.pumpCalMin[1])/2

def generate_function():
    period = 180
    time = 0
    while time < period:
        #y=A*sin(2*pi*f*t), f = 1/T
        time1.append(time)
        time2.append(time)
        flow1.append(A1 * math.sin(2* math.pi / period * time) + avg1)
        flow2.append(A2 * math.cos(2* math.pi / period * time) + avg2)
        time += 1
    
def send_arb_flows():
    arbitraryFlowRate.set_lists(time1, time2, flow1, flow2)
    return True