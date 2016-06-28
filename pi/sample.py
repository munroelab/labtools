#!/usr/bin/env python
"""
This program is used to get a specific voltage measure for a single density cup.
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

def main():

    cup_number = raw_input("Enter cup number: ")
    cup_density = raw_input("Enter its density: ")
 
    # Set up LabJack
    d = u3.U3()
    # print d.configU3()
    AIN0_REGISTER = 0
    print "Cup no.: ", cup_number, "\n", "Density: ", cup_density
    
    while True:
        sample = d.getAIN(0)
        now = datetime.datetime.now()
        print now, sample

        time.sleep(0.1)


if __name__ == "__main__":
    main()
