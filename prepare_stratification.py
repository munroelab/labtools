#!/bin/env python
"""
This program aids in the creation of density stratified tanks using the Double
Bucket method (Oster 1966?)

It is integrated with the Munroe Lab labdb but will also function as a stand
alone tool.

# Scenario: To prepare a linear stratification with N = 1.00 in the Big Tank.
The double buckets contain and unknown certain amount of water / salt water.
(could be empty)

We consider a tank of dimensions W x L to be filled to a depth of H.  
check: H < tank_depth

Objects (like the wave maker) placed inside the tank will change things.  But,
if we assume that they are small in surface area, we can ignore them in an
initial approximation.  

The program guides the experimenter on how much salt
and water to add to the double buckets. It also documents what has been done
automatically in the database.

"""

