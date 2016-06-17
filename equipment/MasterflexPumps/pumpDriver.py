"""
Arduino Pump Driver Python API
Created by Dean Massecar
on 2013-06-10
Contact: dam671@mun.ca or dean.massecar@gmail.com
Chartered by Dr. James Munroe
Contact: jmunroe@mun.ca
Last updated 2013-06-21
"""

import serial
import glob
import time
import math
import pumpCalibration

ser = serial.Serial()

pumpCal = [1, 0]
TESTING = False


def init(devicePath=None, testing=False):
    if testing=True:
        # fake the presence of real pumps

        TESTING = True
        return

    if devicePath is None:
        #minor problem: while Mac uses this:
        devicePaths = glob.glob("/dev/ttyACM0")
        #Linux uses this:
        #devicePaths = glob.glob("/dev/tty*")
        if len(devicePaths) > 0:
                devicePath = devicePaths[0]
        else:
                print("Arduino Not Found/Connected")
                return False
    ser.baudrate = 57600
    ser.port = devicePath
    ser.timeout = 1.25
    ser.open()
    while ser.isOpen()!=True:
        time.sleep(0.5)
        print("WAITING")
    print("DEVICE FOUND: " + devicePath)
    print(ser)

    time.sleep(1)
    ser.flushInput()
    ser.flushOutput()

    #for Arduino compatibility, byte strings must be used
    #this section tests the connection to the Arduino by asking for
    #the initial state of pump one, which should be received as '0\n'
    write('GT1\n')
    response = ser.read(2)
    #print(response)
    if response == b'0\n':
        #always works correctly on first startup, but on subsequent connection
        #establishment, always throws a false...
        print('Connected.')
        return True
    else:
        print(response)
        print('No connection.')
        return False

def write(s):
    #'GT%d%d'%(pump, 123)
    s = s + '\n'
    #bs = bytes(s, 'UTF-8')
    bs = bytes(s)
    ser.write(bs)
    return True

def read_input():
    #reads the serial stream and pulls the first int
    #from the incoming string
    #used for the GET functions to pass the value onwards
    valuePassed = ser.readline()
    valuePassed = valuePassed.decode("UTF-8")
    valuePassed = valuePassed[:valuePassed.find('\n')]
    typeCastValue = int(valuePassed)
    return typeCastValue

def good_input():
    #used on the SET functions, where it's expecting
    #'OK\n' representing a good set, 'ERROR\n'
    #meaning a bad input, and nothing has changed
    valuePassed = ser.readline()
    valuePassed = valuePassed.decode("UTF-8")
    if valuePassed[:valuePassed.find('\n')] == 'OK':
        return True
    elif valuePassed[:valuePassed.find('\n')] == 'ERROR':
        return False
    else:
        return False

def set_speed(pump, setting):
    if TESTING:
        return (True)

    pump = pump + 1 #Python to Arduino conversion of pump notation
    if setting > 255:
        setting = 255
    elif setting < 0:
        setting = 0
    write('SP' + str(pump) + str(setting))
    if good_input() == True:
        return True
    else:
        return False

def set_state(pump, boolState):
    if TESTING:
        return (True)

    pump = pump + 1 #Python to Arduino conversion of pump notation
    if boolState == True: #converts from boolean to an int for the Arduino
        state = 1
    else:
        state = 0
    write('ST' + str(pump) + str(state))
    if good_input() == True:
        return True
    else:
        return False

def get_speed(pump):
    if TESTING:
        return (True)

    pump = pump + 1 #Python to Arduino conversion of pump notation
    write('GP' + str(pump))
    pumpSpeed = read_input()
    return pumpSpeed

def get_state(pump):
    if TESTING:
        return (True)

    pump = pump + 1 #Python to Arduino conversion of pump notation
    write('GT' + str(pump))
    pumpState = read_input()
    if pumpState == 1:
        return(True)
    else:
        return(False)

def set_flow_rate(pump, flowRate):
    #allows the GUI to call a decimal flowrate, and the
    #function will then pass along the appropriate PWM value
    #to the set speed function
    pumpCal[0] = pumpCalibration.pumpCalSlope[pump]
    pumpCal[1] = pumpCalibration.pumpCalMin[pump]
    valuePWM = (flowRate - pumpCal[1]) / pumpCal[0]
    valuePWM = math.trunc(valuePWM)
    #should probably round this
    actOK = set_speed(pump, valuePWM)
    return actOK

def close():
    set_state(0, False)
    set_state(1, False)
    set_speed(0, 0)
    set_speed(1, 0)
    ser.close()
