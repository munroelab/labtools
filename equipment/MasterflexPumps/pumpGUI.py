from tkinter import *
import pumpDriver
import sys
import pumpCalibration
import pumpSinWave
import arbitraryFlowRate
import createFlowPattern

#Arduino Pump Driver Python Test GUI
#Created by Dean Massecar
#on 2013-06-10
#Contact: dam671@mun.ca or dean.massecar@gmail.com
#Chartered by Dr. James Munroe
#Contact: jmunroe@mun.ca
#Last updated 2013-07-04

app = Tk()
app.title("Arduino Pump Controller GUI")
app.geometry('550x300+1025+145')
    
pump1Status = DoubleVar()
pump1Speed = 0
pump1CurrentSpeed = DoubleVar()
pump1CurrentStatus = StringVar()
pump2Status = DoubleVar()
pump2Speed = 0
pump2CurrentSpeed = DoubleVar()
pump2CurrentStatus = StringVar()
pump1StateRes = StringVar()
pump1SpeedRes = StringVar()
pump2StateRes = StringVar()
pump2SpeedRes = StringVar()
serialConStatus = StringVar()

def init():
    result = pumpDriver.init()
    if result == False:
        pumpDriver.close()
        app.destroy()
        sys.exit("Error: No Connection Established")
    #sets all display values to zero
    #Arduino guarenteed to be at zero, because new
    #serial connection causes reset, and Arduino
    #sets state and speed to zero on setup
    pump1CurrentStatus.set(0)
    pump2CurrentStatus.set(0)
    pump1CurrentSpeed.set(0)
    pump2CurrentSpeed.set(0)
    pump1StateRes.set("OFF")
    pump2StateRes.set("OFF")
    pump1SpeedRes.set(0)
    pump2SpeedRes.set(0)
    pumpUpdater()
    createFlowPattern.generate_function()
    createFlowPattern.send_arb_flows()
    check_connection()

def reinit():
    #check port, if open, close, then init and update status
    #if closed, init and update status
    conOpen()
    check_connection()
    
def close():
    pumpSinWave.stop_sine_wave()
    pumpDriver.close()
    check_connection()

def conOpen():
    pumpDriver.init()

def pump1Updater():
    if pumpDriver.get_state(0) == True:
        pump1StateRes.set("ON")
    else:
        pump1StateRes.set("OFF")
    currentSpeed = pumpDriver.get_speed(0)
    pump1SpeedRes.set(currentSpeed)

def pump2Updater():
    if pumpDriver.get_state(1) == True:
        pump2StateRes.set("ON")
    else:
        pump2StateRes.set("OFF")
    currentSpeed = pumpDriver.get_speed(1)
    pump2SpeedRes.set(currentSpeed)

def pumpUpdater():
    pump1Updater()
    pump2Updater()
    
def toggle_status_pump1():
    pump = 0
    if pump1Status.get() == 1:
        actOK = pumpDriver.set_state(pump, True)
        if actOK == True:
            pump1CurrentStatus.set(1)
            pumpUpdater()
        else:
            print("Error: Could not complete action\n")
    elif pump1Status.get() == 0:
        actOK = pumpDriver.set_state(pump, False)
        if actOK == True:
            pump1CurrentStatus.set(0)
            pumpUpdater()
        else:
            print("Error: Could not complete action\n")
    else:
        print("Error: Checkbox action not recognized\n")

def toggle_status_pump2():
    pump = 1
    if pump2Status.get() == 1:
        actOK = pumpDriver.set_state(pump, True)
        if actOK == True:
            pump2CurrentStatus.set(1)
            pumpUpdater()
        else:
            print("Error: Could not complete action\n")
    elif pump2Status.get() == 0:
        actOK = pumpDriver.set_state(pump, False)
        if actOK == True:
            pump2CurrentStatus.set(0)
            pumpUpdater()
        else:
            print("Error: Could not complete action\n")
    else:
        print("Error: Checkbox action not recognized\n")
        
def shutdown():
    close()
    app.destroy()

def set_speed_pump1():
    #the commented out entries are for PWM, the new entries for LPM
    pump = 0
    pumpSpeed = pump1CurrentSpeed.get()
    #if entry slot is not blank, update values
    if pump1SpeedEntry.get()!='':
        pumpSpeed = pump1SpeedEntry.get()
    pumpSpeed = float(pumpSpeed)
    if pumpSpeed > pumpCalibration.pumpCalSlope[pump] * 255:
        pumpSpeed = pumpCalibration.pumpCalSlope[pump] * 255
    elif pumpSpeed < pumpCalibration.pumpCalMin[pump]:
        pumpSpeed = pumpCalibration.pumpCalMin[pump]
    actOK = pumpDriver.set_flow_rate(pump, pumpSpeed)
    if actOK == True:
        pump1CurrentSpeed.set(pumpSpeed)
        pumpUpdater()
    else:
        print("Error: Could not complete action\n")

def set_speed_pump2():
    #the commented out entries are for PWM, the new entries for LPM
    pump = 1
    pumpSpeed = pump2CurrentSpeed.get()
    #if entry slot is not blank, update values
    if pump2SpeedEntry.get()!='':
        pumpSpeed = pump2SpeedEntry.get()
    pumpSpeed = float(pumpSpeed)
    if pumpSpeed > pumpCalibration.pumpCalSlope[pump] * 255:
        pumpSpeed = pumpCalibration.pumpCalSlope[pump] * 255
    elif pumpSpeed < pumpCalibration.pumpCalMin[pump]:
        pumpSpeed = pumpCalibration.pumpCalMin[pump]
    actOK = pumpDriver.set_flow_rate(pump, pumpSpeed)
    if actOK == True:
        pump2CurrentSpeed.set(pumpSpeed)
        pumpUpdater()
    else:
        messagebox.showinfo(message="Error: Could not complete action\n")

def check_connection():
    if pumpDriver.ser.isOpen() == True:
        serialConStatus.set("Con")
        return True
    else:
        serialConStatus.set("No Con")
        return False

#setup labels for top row
pumpNumLabel = Label(app, text="PUMP", justify = 'left')
pumpNumLabel.grid(row=0, column=0)
currentStatusLabel = Label(app, text="STATUS", justify = 'left')
currentStatusLabel.grid(row=0, column=1)
onOffLabel = Label(app, text="ON/OFF", justify = 'left')
onOffLabel.grid(row=0, column=2)
currentSpeedLabel = Label(app, text="FLOW", justify = 'left')
currentSpeedLabel.grid(row=0, column=4)
inputSpeedLabel = Label(app, text="INPUT", justify = 'left')
inputSpeedLabel.grid(row=0, column=5)
  
#setup pump 1 inputs
pump1Label = Label(app, text="PUMP 1: ", height = 3)
pump1Label.grid(row=1, column=0)
pump1StatusCheck = Label(app, textvariable=pump1CurrentStatus, height = 3, width = 2)
pump1StatusCheck.grid(row=1, column=1)
pump1Check = Checkbutton(app, variable = pump1Status, command = toggle_status_pump1, text = "ON")
pump1Check.grid(row=1, column=2)
pump1SpeedLabel = Label(app, text="SPEED: ", height = 3)
pump1SpeedLabel.grid(row=1, column=3)
pump1CurrentSpeedLabel = Label(app, textvariable=pump1CurrentSpeed, height = 3, width = 3, justify = 'left')
pump1CurrentSpeedLabel.grid(row=1, column = 4)
pump1SpeedEntry = Entry(app, width = 6, justify = 'center')
pump1SpeedEntry.grid(row=1, column=5)
pump1SpeedEntryButton = Button(app, text="OK", width = 3, command=set_speed_pump1)
pump1SpeedEntryButton.grid(row=1, column=6)

#setup pump 2 inputs
pump2Label = Label(app, text="PUMP 2: ", height = 3)
pump2Label.grid(row=2, column=0)
pump2StatusCheck = Label(app, textvariable=pump2CurrentStatus, height = 3, width = 2)
pump2StatusCheck.grid(row=2, column=1)
pump2Check = Checkbutton(app, variable = pump2Status, command = toggle_status_pump2, text = "ON")
pump2Check.grid(row=2, column=2)
pump2SpeedLabel = Label(app, text="SPEED: ", height = 3)
pump2SpeedLabel.grid(row=2, column=3)
pump2CurrentSpeedLabel = Label(app, textvariable=pump2CurrentSpeed, height = 3, width = 3, justify = 'left')
pump2CurrentSpeedLabel.grid(row=2, column = 4)
pump2SpeedEntry = Entry(app, width = 6, justify = 'center')
pump2SpeedEntry.grid(row=2, column=5)
pump2SpeedEntryButton = Button(app, text="OK", width = 3, command=set_speed_pump2)
pump2SpeedEntryButton.grid(row=2, column=6)

#setup pump 1 query results
pump1StateResult = Label(app, text="Pump 1 State: ", height = 3)
pump1StateResult.grid(row=3,column=0)
pump1StateOutput = Label(app, textvariable = pump1StateRes, height = 3, width = 4)
pump1StateOutput.grid(row=3, column=1)
pump1SpeedResult = Label(app, text="Speed: ", height = 3)
pump1SpeedResult.grid(row=3, column=2)
pump1SpeedOutput = Label(app, textvariable = pump1SpeedRes, height = 3, width = 4)
pump1SpeedOutput.grid(row=3, column=3)

#setup pump 2 query results
pump2StateResult = Label(app, text="Pump 2 State: ", height = 3)
pump2StateResult.grid(row=4,column=0)
pump2StateOutput = Label(app, textvariable = pump2StateRes, height = 3, width = 4)
pump2StateOutput.grid(row=4, column=1)
pump2SpeedResult = Label(app, text="Speed: ", height = 3)
pump2SpeedResult.grid(row=4, column=2)
pump2SpeedOutput = Label(app, textvariable = pump2SpeedRes, height = 3, width = 4)
pump2SpeedOutput.grid(row=4, column=3)

#setup status line with init button, and sin button
serialConLabel = Label(app, text="Serial: ", height = 3)
serialConLabel.grid(row=5, column=0)
serialConRes = Label(app, textvariable = serialConStatus, height = 3, width = 5)
serialConRes.grid(row=5, column=1)
serialConInit = Button(app, text="init()", width = 3, height=3, command=reinit)
serialConInit.grid(row=5, column=2)
serialConClose = Button(app, text="close()", width = 6, height = 3, command=close)
serialConClose.grid(row=5, column=3)
#runSinWave = Button(app, text="sin wave", width = 7, height = 3, command=open_sine_GUI)
#runSinWave.grid(row=5, column=4)
#runArbFlow = Button(app, text="arb flow", width = 7, height = 3, command=set_lists)
#runArbFlow.grid(row = 5, column = 5)
calBut = Button(app, text="calibrate", width = 8, height = 3)
calBut.grid(row=5, column=6)

init()

#event handlers
app.protocol("WM_DELETE_WINDOW", shutdown)

pumpSinWave.sinWin.after(1000, pumpSinWave.update_sine_wave)
arbitraryFlowRate.arbFlowWin.after(1000, arbitraryFlowRate.update_flow_rates)
app.mainloop()
