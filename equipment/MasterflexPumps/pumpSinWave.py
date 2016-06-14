from tkinter import *
import pumpDriver
import math
import time
import pumpCalibration

#Arduino Pump Driver Sine Wave Test GUI
#Created by Dean Massecar
#on 2013-06-19
#Contact: dam671@mun.ca or dean.massecar@gmail.com
#Chartered by Dr. James Munroe
#Contact: jmunroe@mun.ca
#Last updated 2013-07-03

sinWin = Tk()
#sinWin = Toplevel()
sinWin.title("Sine Wave Setup")
sinWin.geometry("320x250+800+400")

#textvariables must be declared AFTER the tk() object is created
p1MinFlowSet = DoubleVar()
p1MaxFlowSet = DoubleVar()
p1PeriodSet = DoubleVar()
p2MinFlowSet = DoubleVar()
p2MaxFlowSet = DoubleVar()
p2PeriodSet = DoubleVar()
svMinFlow = DoubleVar()
svMaxFlow = DoubleVar()
svPump = IntVar()
svPeriod = DoubleVar()

#because of the looping nature of a tkinter window,
#global variables are needed for settings and states.
#some sort of variable assignment error? These should be within
#the scope of this file
minFlow = [0.0, 0.0]
maxFlow = [0.0, 0.0]
period = [0.01, 0.01]
sinTime = [0.000, 0.000]
startTime = [0.000, 0.000]
inProgress = [False, False]

def setup_GUI():
    #called immediately upon needing, simply for code readability
    pLabel = Label(sinWin, text = "Parameter(unit)", height = 2, justify=RIGHT)
    pLabel.grid(row=0, column=0)
    sLabel = Label(sinWin, text = "Value", height = 2, justify = CENTER)
    sLabel.grid(row=0, column=1)
    s1Label = Label(sinWin, text = "P1 Setting", height = 2, justify = CENTER)
    s1Label.grid(row=0, column=2)
    s2Label = Label(sinWin, text = "P2 Setting", height = 2, justify = CENTER)
    s2Label.grid(row=0, column=3)
    
    pMinFlow = Label(sinWin, text = "minFlow(LPM)", height = 2, justify=RIGHT)
    pMinFlow.grid(row=1, column=0)
    sMinFlow = Entry(sinWin, width = 5, justify = CENTER, textvariable = svMinFlow)
    sMinFlow.grid(row=1, column=1)
    p1MinFlow = Label(sinWin, width = 4, textvariable=p1MinFlowSet, justify=CENTER)
    p1MinFlow.grid(row=1, column=2)
    p2MinFlow = Label(sinWin, width = 4, textvariable=p2MinFlowSet, justify=CENTER)
    p2MinFlow.grid(row=1, column=3)
    
    pMaxFlow = Label(sinWin, text = "maxFlow(LPM)", height = 2, justify=RIGHT)
    pMaxFlow.grid(row=2, column=0)
    sMaxFlow = Entry(sinWin, width = 5, justify = CENTER, textvariable = svMaxFlow)
    sMaxFlow.grid(row=2, column=1)
    p1MaxFlow = Label(sinWin, textvariable=p1MaxFlowSet, height = 2, width = 4, justify=RIGHT)
    p1MaxFlow.grid(row=2, column=2)
    p2MaxFlow = Label(sinWin, textvariable=p2MaxFlowSet, height = 2, width = 4, justify=RIGHT)
    p2MaxFlow.grid(row=2, column=3)
    
    pPeriod = Label(sinWin, text = "period(s)", height = 2, justify=RIGHT)
    pPeriod.grid(row=3, column=0)
    sPeriod = Entry(sinWin, width = 5, justify = CENTER, textvariable = svPeriod)
    sPeriod.grid(row=3, column=1)
    p1Period = Label(sinWin, width = 5, textvariable = p1PeriodSet, justify = CENTER)
    p1Period.grid(row=3, column=2)
    p2Period = Label(sinWin, width = 5, textvariable = p2PeriodSet, justify = CENTER)
    p2Period.grid(row=3, column=3)
    
    pPump = Label(sinWin, text = "pump(#)", height = 2, justify=RIGHT)
    pPump.grid(row=4, column=0)
    sPump = Entry(sinWin, width = 5, justify = CENTER, textvariable = svPump)
    sPump.grid(row=4, column=1)
    p1PumpLabel = Label(sinWin, width = 5, text = "0", justify = CENTER)
    p1PumpLabel.grid(row=4, column=2)
    p2PumpLabel = Label(sinWin, width = 5, text = "1", justify = CENTER)
    p2PumpLabel.grid(row=4, column=3)
    
    bOkay = Button(sinWin, text = "OK",  width=3, height=2, command = set_values)
    bOkay.grid(row=5, column=0)
    bClear = Button(sinWin, text = "CLEAR", width=5, height=2, command = clear_values)
    bClear.grid(row=5, column=1)
    bStop = Button(sinWin, text = "STOP", width=4, height=2, command = stop_sine_wave)
    bStop.grid(row=5, column=2)
    
def set_values():
    #need to add constricting values in get values
    sinMinFlow = svMinFlow.get()
    sinMaxFlow = svMaxFlow.get()
    sinPeriod = svPeriod.get()
    sinPump = svPump.get()
    sinMinFlow = float(sinMinFlow)
    sinMaxFlow = float(sinMaxFlow)
    sinPeriod = float(sinPeriod)
    sinPump = int(sinPump)
    
    if sinPump != 0:
        #if not a valid pump number, don't continue this function
        print("no valid pump")
        return False
    elif sinPump != 1:
        print("no valid pump")
        return False
    
    pumpMinFlow = pumpCalibration.pumpCalMin[sinPump]
    pumpMaxFlow = (pumpCalibration.pumpCalSlope[sinPump] * 255) + pumpMinFlow
    print("Max: " + str(pumpMaxFlow) + ", Min: " + str(pumpMinFlow))
    
    if sinMinFlow < pumpMinFlow:
        sinMinFlow = pumpMinFlow
    elif sinMaxFlow > pumpMaxFlow:
        sinMaxFlow = pumpMaxFlow
        
    if sinMinFlow > sinMaxFlow:
        #ensures minimum is lower than maximum
        temp = sinMinFlow
        sinMinFlow = sinMaxFlow
        sinMaxFlow = temp
    
    minFlow[sinPump] = sinMinFlow
    maxFlow[sinPump] = sinMaxFlow
    period[sinPump] = sinPeriod
    inProgress[sinPump] = True
    startTime[sinPump] = time.time()
    update_values()
    #clear_values()
    
    #if sinPump == p1SinPump:
    #    p1minFlow = sinMinFlow
    #    p1maxFlow = sinMaxFlow
    #    p1period = sinPeriod
    #    p1inProgress = True
    #    p1lastRun = time.time()
    #elif sinPump == p2SinPump:
    #    p2minFlow = sinMinFlow
    #    p2maxFlow = sinMaxFlow
    #    p2period = sinPeriod
    #    p2inProgress = True
    #    p2lastRun = time.time()
    #else:
    #    return False
    
def run_sine_wave(pump):
    #runs a sin wave on pump, with minFlow and maxFlow, over period in
    #seconds. sin wave will continue to run on specified motor until stopped.
        
    sinAmplitude = (maxFlow[pump] - minFlow[pump]) / 2
    sinOmega = 2 * math.pi / period[pump]
    sinTime[pump] = (time.time() - startTime[pump])

    #there is an issue with time.time(): in 2038, epoch time will over value
    #and this will no longer work. That being said, 25 years support...
    
    #y(t)=Asin(2*pi*freq*t + phi), no phi, 2*pi*freq = sinOmega
    #flowRate(t) = A*sin(omega * t)
    if inProgress[pump] == True:
        newFlow = sinAmplitude * math.sin(sinOmega * sinTime[pump]) + (maxFlow[pump] + minFlow[pump]) / 2
        print(newFlow)
        actOK = pumpDriver.set_flow_rate(pump, newFlow)
        if actOK == True:
            #print("OK")
            return True
        else:
            print("NOT OK")
            return False
    else:
        #print("No settings")
        return False
    

def clear_values():
    #simply clears the text boxes
    svMinFlow.set(0.0)
    svMaxFlow.set(0.0)
    svPeriod.set(0.001)
    svPump.set(2)
    return True

def stop_sine_wave():
    minFlow[0] = 0.00
    maxFlow[0] = 0.0
    period[0] = 0.001
    sinTime[0] = 0.001
    inProgress[0] = False
    minFlow[1] = 0.00
    maxFlow[1] = 0.0
    period[1] = 0.001
    sinTime[1] = 0.001
    inProgress[1] = False
    update_values()
    return True
    
def update_sine_wave():
    run_sine_wave(0)
    run_sine_wave(1)
    sinWin.after(1000, update_sine_wave)
    return True
    
def update_values():
    p1MinFlowSet.set(minFlow[0])
    p1MaxFlowSet.set(maxFlow[0])
    p1PeriodSet.set(period[0])
    p2MinFlowSet.set(minFlow[1])
    p2MaxFlowSet.set(maxFlow[1])
    p2PeriodSet.set(period[1])
    
def close_window():
    stop_sine_wave()
    sinWin.destroy()
    #check values in window for all being complete

setup_GUI()
stop_sine_wave()
sinWin.protocol("WM_DELETE_WINDOW", close_window) 
