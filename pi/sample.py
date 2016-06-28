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

    cup_number = 1
    density = None
 
    # Set up LabJack
    d = u3.U3()
    print d.configU3()
    AIN0_REGISTER = 0


    while True:
        sample = d.getAIN(0)
        now = datetime.datetime.now()
        print now, sample

        time.sleep(0.1)


if __name__ == "__main__":
    main()
