#!/usr/bin/env python
"""
code to communication with lab jack and MSCTI

Control for VXM-1

JogUp
JogDown
TraverseUp
TraverseDown
"""

steps_per_mm = 200

import argparse
import serial
import sys
import glob
import io
import os
import time
import datetime
import u3
from labdb import LabDB

def load_strat_data(filename, strat_id):
    f = open(filename, 'r')

    db = LabDB()
    for line in f.readlines():
        data = line.split(',')
        
        sql = """INSERT INTO stratification_data (strat_id, z, Tvolt, Cvolt)
                 VALUES (%d, %f, %f, %f)""" % (strat_id, z, Tvolt, Cvolt)

        print line
        db.execute(sql)

    db.commit()
    db.close()

def get_strat_id():
    db = LabDB()

    sql = "INSERT INTO stratification (strat_id) VALUES (NULL)"
    db.execute(sql)
    sql = "SELECT LAST_INSERT_ID()"
    rows = db.execute(sql)
    strat_id, = rows[0]

    db.commit()
    db.close()

    return strat_id

def set_strat_metadata(strat_id, path):
    db = LabDB()

    sql = """UPDATE stratification
             SET path = '%s'
             WHERE strat_id = %d""" % (path, strat_id)

    db.execute(sql)
    db.commit()

    db.close()

def main():

    parser = argparse.ArgumentParser(description="Move the probe and measure the stratification")

    parser.add_argument("dz", type=float, help = "distance to increment probe")
    args = parser.parse_args()

    print "Probe will move by", args.dz, "cm"

    # Set up LabJack
    d = u3.U3()
    #print d.configU3()
    AIN0_REGISTER = 0

    # determine tty name for VXM-1
    usb_serial_devices = glob.glob('/dev/tty.usbserial*')
    if len(usb_serial_devices) == 0:
        print "error: no usb serial device found"
    elif len(usb_serial_devices) > 1:
        print "warning: multiple usb serial devices found. Using first one."

    devname = usb_serial_devices[0]
    #print "Found: %s" % devname

    ser = serial.Serial(devname, timeout=0)

    def readLine(ser):
        str = ""
        while True:
            ch = ser.read()
            if(ch == '\r' or ch == ''):
                break
            str += ch
        return str


    print "Initializing VXM..."

    # reset VXM-1
    ser.write("KCG")

    def Increment(z, speed = 2000):

        z = int(z * steps_per_mm * 10)
        print "Incrementing..."
        program_template = """
    PM-0,
    S%d,
    I%d,
    """
        program = program_template % (speed, z)

        ser.write(program)
        ser.write("PM0,R\n")
        # block and wait for completion?

    valid_position = False
    while not valid_position:
        ser.write("X")
        reading = readLine(ser).strip()

        try:
            x = int(reading)
            position = float(reading) / steps_per_mm / 10.0
            valid_position = True
            break
        except:
            pass
        time.sleep(1.0)

    print "X=", position, x

    time.sleep(1)

    strat_id = get_strat_id()
    strat_path= '/Volumes/HD3/strat_data/%d' % strat_id
    os.mkdir(strat_path)

    Increment(args.dz)

    now = datetime.datetime.now()
    filename = "stratification_%s.csv" % now.strftime("%y%m%d%H%M%S")
    strat_path = os.path.join(strat_path, filename)

    f = open(strat_path, 'w')
    f.write("timestamp (isoformat), X (steps), z (cm), Conductivity(V)\n")

    in_motion = True
    while in_motion:
        ser.write("X")
        reading = readLine(ser).strip()
        if reading == "^":
            in_motion = False
            reading = readLine(ser).strip()

        try:
            x = int(reading)
            position = float(reading) / steps_per_mm / 10.0
        except:
            print reading
            print "Position read error"

        sample = d.readRegister(AIN0_REGISTER)
        now = datetime.datetime.now()
        print now, x, position, sample
        f.write("%s, %s, %s, %s\n" % (now, x, position, sample))

        time.sleep(0.1)

    f.close()
    print "Motion Completed"
    ser.write("Q")

    set_strat_metadata(strat_id, path=strat_path)

    print "traverse data is stored in %s" % strat_path
    print
    print "strat_id =", strat_id

    # import this data into the lab db
    # open the csv file


if __name__ == "__main__":
    main()
