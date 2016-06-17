#Python Arbitrary Flow Rate GUI
#Created by Dean Massecar
#on 2013-06-26
#Contact: dam671@mun.ca or dean.massecar@gmail.com
#Chartered by Dr. James Munroe
#Contact: jmunroe@mun.ca
#Last updated 2013-07-10

#Stand alone window to control flow rates for an arbitrary function,
#generated in another function. Given that no motor control should
#be available while running the program, only the buttons will be available.

from Tkinter import *
import pumpDriver
import pumpCalibration
import math
import time

arbFlowWin = Tk()
arbFlowWin.title("Arbitrary Flow Rate")
arbFlowWin.geometry('300x110+800+150')

time_elapsed = DoubleVar()
rate_flow1 = DoubleVar()
rate_flow2 = DoubleVar()
perc_done = DoubleVar()
file_name = StringVar()

startTime = 0.0
timeElapsed = 0.0
funcInProgress = False
time1 = []
time2 = []
flow1 = []
flow2 = []
curFlow1 = 0.0
curFlow2 = 0.0
totalTime = 0.0

def longest_time_in_lists(list1, list2):
    l1 = len(list1)
    l1 = list1[l1-1]
    l2 = len(list2)
    l2 = list2[l2-1]
    if l1 > l2:
        return l1
    else:
        return l2

def set_lists(at1, at2, af1, af2):
    #sets locally accessable lists from passed lists
    #any calls of this function assumes that lengths
    #of flowx and timex are equal across x
    #as well, time is in seconds, flow is in LPM
    #time is also expected to be in increasing order
    global time1, time2, flow1, flow2, funcInProgress, totalTime
    
    time1 = at1
    time2 = at2
    flow1 = af1
    flow2 = af2
    totalTime = longest_time_in_lists(time1, time2)
    funcInProgress = False    

def set_flow_rate(pump, time_list, flow):
    global startTime
    curTime = time.time() - startTime
    curTime = math.trunc(curTime)
    length = len(time_list)
    index = 0
    if curTime > totalTime:
        print("Flow profile complete")
        stop_arb_flow()
        return True
    while time_list[index] < curTime:
        index += 1
    pumpDriver.set_flow_rate(pump, flow[index])
    
def update_values():
    time_elapsed.set(time.time() - startTime)
    perc_done.set((totalTime - (time.time() - startTime)) / totalTime * 100)
    rate_flow1.set(curFlow1)
    rate_flow2.set(curFlow2)
    
def update_flow_rates():
    global funcInProgress, time1, time2, flow1, flow2
    #checks and updates required flow rates
    if not funcInProgress:
        arbFlowWin.after(1000, update_flow_rates)
        return False
    else:
        set_flow_rate(0, time1, flow1)
        set_flow_rate(1, time2, flow2)
        update_values()
        arbFlowWin.after(1000, update_flow_rates)
        return True

def run_arb_flow():
    global funcInProgress, startTime
    funcInProgress = True
    startTime = time.time()
    print("Profile running")
    return True

def pause_arb_flow():
    global funcInProgress, startTime, timeElapsed
    if funcInProgress == False and startTime == 0.0:
        return False
    elif funcInProgress == False and startTime != 0.0:
        print("Profile resumed")
        funcInProgress = True
        startTime = time.time() - timeElapsed
        pumpDriver.set_state(0, True)
        pumpDriver.set_state(1, True)
        return True
    else:
        print("Profile paused")
        funcInProgress = False
        pumpDriver.set_state(0, False)
        pumpDriver.set_state(1, False)
        timeElapsed = time.time() - startTime
        return True

def stop_arb_flow():
    global startTime, funcInProgress, curFlow1, curFlow2
    startTime = 0.0
    funcInProgress = False
    curFlow1 = 0.0
    curFlow2 = 0.0
    pumpDriver.set_state(0, False)
    pumpDriver.set_state(1, False)
    pumpDriver.set_flow_rate(0, 0.0)
    pumpDriver.set_flow_rate(1, 0.0)
    return True

def parse_file(fileName = ''):
    pass
    if fileName == '':
        return False
    else:
        #format:
        #
        #1t1 1f1\n
        #2t1 2f1\n
        #xt1 xf1\n
        #*\n
        #1t2 1f2\n
        #2t2 2f2\n
        #xt2 xf2\n
        # 
        #star acting as a break point for
        #between flow profiles, where
        # xty xfy
        #are doubles representing time in
        #seconds and flow rate in LPM, respectively
        rf = open(fileName, 'r')
        onSet = 1
        list1_index = 0
        list2_index = 0
        ft1 = []
        ft2 = []
        ff1 = []
        ff2 = []
        
        for line in rf:
            rl = rf.readline()
            if rl != '' and rl != '\n':
                if onSet == 1:
                    list = rl.split(' ')
                    t1 = list[0]
                    f1 = list[1]
                    f1 = f1.rstrip()
                    ft1[list1_index] = float(t1)
                    ff1[list1_index] = float(f1)
                    list1_index += 1
                elif onSet == 2:
                    list = rl.split(' ')
                    t2 = list[0]
                    f2 = list[1]
                    f2 = f2.rstrip()
                    ft2[list2_index] = float(t2)
                    ff2[list2_index] = float(f2)
                    list2_index += 1
            elif rl == '*\n':
                #when * is reached, advance to
                #next pair of lists
                onSet = 2
        rf.close()
        set_lists(ft1, ft2, ff1, ff2)
        return True

def get_file():
    fileName = tkFileDialog.askopenfilename(defaultextension = '.csv', 
        parent = arbFlowWin, title = 'Select Flow Profile')
    parse_file(fileName)

def make_GUI():
    #creates the GUI elements, more to keep code segregated
    timeLabel = Label(arbFlowWin, text = 'Time (s)', justify = 'center')   
    timeLabel.grid(row = 0, column = 0)
    flow1Label = Label(arbFlowWin, text = 'Flow 1 (LPM)', justify = 'center')   
    flow1Label.grid(row = 0, column = 2)
    flow2Label = Label(arbFlowWin, text = 'Flow 2 (LPM)', justify = 'center')   
    flow2Label.grid(row = 0, column = 4)
    perDoneLabel = Label(arbFlowWin, text = '% Done', justify = 'center')
    perDoneLabel.grid(row = 0, column = 6)
    
    timeElapsed = Label(arbFlowWin, textvariable = time_elapsed, justify = 'center', width = 5)
    timeElapsed.grid(row = 1, column = 0)
    rateFlow1 = Label(arbFlowWin, textvariable = rate_flow1, justify = 'center', width = 4)
    rateFlow1.grid(row = 1, column = 2)
    rateFlow2 = Label(arbFlowWin, textvariable = rate_flow2, justify = 'center', width = 4)
    rateFlow2.grid(row = 1, column = 4)
    percDone = Label(arbFlowWin, textvariable = perc_done, justify = 'center', width = 4)
    percDone.grid(row = 1, column = 6)
    
    runButton = Button(arbFlowWin, text = 'Run', width = 4, justify = 'center', command = run_arb_flow)
    runButton.grid(row = 2, column = 0)
    pauseButton = Button(arbFlowWin, text = 'Pause/Un-Pause', width = 16, justify = 'center', command = pause_arb_flow)
    pauseButton.grid(row = 2, column = 2, columnspan = 4)
    stopButton = Button(arbFlowWin, text = 'Stop', width = 4, justify = 'center', command = stop_arb_flow)
    stopButton.grid(row = 2, column = 6)
    
    actFileLab = Label(arbFlowWin, text = 'Active: ', width = 6, justify = 'center')
    actFileLab.grid(row = 3, column = 0)
    actFileName = Label(arbFlowWin, textvariable = file_name, justify = 'left')
    actFileName.grid(row = 3, column = 1, columnspan = 5)
    openFileBut = Button(arbFlowWin, text = 'OPEN', width = 4, justify = 'center', command = get_file)
    openFileBut.grid(row = 3, column = 6)
    
def close_window():
    stop_arb_flow()
    arbFlowWin.destroy()
    
make_GUI()
arbFlowWin.protocol("WM_DELETE_WINDOW", close_window) 
arbFlowWin.after(1000, update_flow_rates)
#arbFlowWin.mainloop()
