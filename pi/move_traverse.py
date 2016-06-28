#!/usr/bin/env python
"""
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

def main():

    parser = argparse.ArgumentParser(description="Move the probe and measure the stratification")

    parser.add_argument("dz", type=float, help = "distance (cm) to increment probe. Positive dz is UP.")
    args = parser.parse_args()

    print "Probe will move by", args.dz, "cm"

    # determine tty name for VXM-1
    usb_serial_devices = glob.glob('/dev/ttyUSB*')
    if len(usb_serial_devices) == 0:
        print "error: no usb serial device found"
    elif len(usb_serial_devices) > 1:
        print "warning: multiple usb serial devices found. Using first one."

    devname = usb_serial_devices[0]
    print "Found: %s" % devname

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

    time.sleep(5)

    Increment(args.dz)

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
            print "Position read error"

        #sample = d.getAIN(0)
        sample = 0.0
        now = datetime.datetime.now()
        print now, x, position, sample

        time.sleep(0.1)

    print "Motion Completed"
    ser.write("Q")

if __name__ == "__main__":
    main()
