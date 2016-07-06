#!/usr/bin/env python
"""
This script drives the vertical travserse and the conducitivity probe
sampled via the LabJack U3.

Command line usage:
    profile -50
will measure a stratification going 50 cm

Only produces stratification_*.csv file if the probe is 
moving downward!

DOUBLE check limit switches are in place. The program can not
tell where the base of the tank is located.

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

def set_strat_metadata(strat_id, path, calib_id):
    """
    Updata metadata in stratication table
    """
    db = LabDB()

    sql = """UPDATE stratification
             SET path = '%s'
             WHERE strat_id = %d""" % (path, strat_id)
    db.execute(sql)
    
    if calib_id is None:
        """look up newest calibration curve"""
        sql = """SELECT calib_id FROM stratification
                ORDER BY calib_id DESC"""
        rows = db.execute(sql)
        calib_id = rows[0][0]

    sql = """UPDATE stratification
             SET calib_id = %d
             WHERE strat_id = %d""" % (calib_id, strat_id)

    db.execute(sql)


    db.commit()

    db.close()

def main():

    parser = argparse.ArgumentParser(description="Move the probe and measure the stratification")

    parser.add_argument("dz", type=float, help = "distance (cm) to increment probe. Positive dz is UP.")
    parser.add_argument("--calib_id", dest='calib_id',
                         help = 'calib_id to use for this stratificaiton profile')
    args = parser.parse_args()

    print "Probe will move by", args.dz, "cm"

    # only save the profile data if the probe is moving downwards
    if (args.dz < 0):
        save_data = True
    else:
        save_data = False

    if not save_data:
        print "Since probe is moving upwards, no record data is being made"

    
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
# changing speed from 2000 to 200 -- 06/12/2013
    def Increment(z, speed = 400):
        """
        Move the vertical traverse by a distance z. (z < 0 is downwards)
        """

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

    if save_data:
        strat_id = get_strat_id()
        strat_path= '/Volumes/HD3/strat_data/%d' % strat_id
        os.mkdir(strat_path)

    Increment(args.dz)

    if save_data:
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

        sample = d.getAIN(0)
        now = datetime.datetime.now()
        print now, x, position, sample
        if save_data:
            f.write("%s, %s, %s, %s\n" % (now, x, position, sample))

        time.sleep(0.1)

    print "Motion Completed"
    ser.write("Q")

    if save_data:
        f.close()
        set_strat_metadata(strat_id, path=strat_path, calib_id=args.calib_id)

        print "traverse data is stored in %s" % strat_path
        print
        print "strat_id =", strat_id
    else:
        print "Data not recorded (z < 0)"


if __name__ == "__main__":
    main()
