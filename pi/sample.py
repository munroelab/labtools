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
    f = open("cup.txt", "w")
    cup_number = raw_input("Enter cup number: ")
    cup_density = raw_input("Enter its density: ")
    #print "Cup no.: ", cup_number, "\n", "Density: ", cup_density
    f.write("Cup_number: ")
    f.write(cup_number) 
    f.write("\nCup_density: ")
    f.write(cup_density)
    f.write("\n")

    # Set up LabJack
    d = u3.U3()
    # print d.configU3()
    AIN0_REGISTER = 0
    
    while True:
        sample = str(d.getAIN(0))
        now = str(datetime.datetime.now())
        print sample
        f.write(now)
        f.write(" ")
        f.write(sample)
        f.write("\n")
        time.sleep(0.1)


if __name__ == "__main__":
    main()

f.close()
