# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import time
from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
from pymavlink import mavutil # Needed for command message definitions


def arm_and_takeoff(aTargetAltitude):
    #Arms vehicle and fly to aTargetAltitude.
    print("Basic pre-arm checks")
    # Don't try to arm until autopilot is ready
    while not vehicle.is_armable:
        print("Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arming motors")
    time.sleep(1)
    # Copter should arm in GUIDED mode
    vehicle.mode    = VehicleMode("GUIDED")
    print("Mode: %s" % vehicle.mode.name)
    vehicle.armed   = True

    
    while (not vehicle.mode.name=="GUIDED"  ):
        print("Getting ready to take off ...")
        vehicle.mode    = VehicleMode("GUIDED")
        time.sleep(1)
    # Confirm vehicle armed before attempting to take off
    while not vehicle.armed:
        print("Waiting for arming...")
        time.sleep(1)   
   
    print("Taking off!")
     # Take off to target altitude
    vehicle.simple_takeoff(aTargetAltitude)
    while True:
        print("Altitude: %s" % vehicle.location.global_relative_frame.alt)
        print("Velocity: %s" % vehicle.velocity)
        # Wait until the vehicle reaches a safe height before processing the goto (otherwise the command
        #  after Vehicle.simple_takeoff will execute immediately).
        #Break and return from function just below target altitude.
        if vehicle.location.global_relative_frame.alt>=aTargetAltitude*0.95:
        #if wait_for_alt(alt = 1, epsilon=0.3, rel=True, timeout=None)
            print("Reached target altitude")
            break
        time.sleep(1)

#Connect to the vehicle 
vehicle = connect('/dev/ttyS0', baud=921600)
"""
#Connect SITL
 run this on cmd 
 cd AppData\Local\Programs\Python\Python36-32\Scripts
 dronekit-sitl copter
"""
#vehicle = connect('tcp:127.0.0.1:5762', wait_ready=True)
# Get some vehicle attributes (state)
print("Get some vehicle attribute values:")
print("GPS: %s" % vehicle.gps_0)
print("Local Location: %s" % vehicle.location.local_frame)
print("Global Location (relative altitude): %s" % vehicle.location.global_relative_frame)
print("Battery: %s" % vehicle.battery)
print("Attitude: %s" % vehicle.attitude)
print("Heading: %s" % vehicle.heading)
print("Velocity: %s" % vehicle.velocity)
print("Last Heartbeat: %s" % vehicle.last_heartbeat)
print("Is Armable?: %s" % vehicle.is_armable)
print("System status: %s" % vehicle.system_status.state)
print("Mode: %s" % vehicle.mode.name)    # settable

#cmds = vehicle.commands
#cmds.download()
#cmds.wait_ready()

# This should return none since home location did not defined or set

print (" Home Location: %s" % vehicle.home_location)
# Now set home location

vehicle.home_location=vehicle.location.global_frame
print (" New Home Location: %s" % vehicle.home_location)

time.sleep(3)
arm_and_takeoff(1)
time.sleep(3)
print("Landing Initiated...")
vehicle.mode = VehicleMode("LAND")

vehicle.armed   = False

print("Vehicle Closing...")
time.sleep(2)
#vehicle.close()
