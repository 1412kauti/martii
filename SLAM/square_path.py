import cv2
import time
import sys
import math
import pybullet as p
from qibullet import *

#Loading simulation environment, with pepper and the obstacle
simulation_manager = SimulationManager()

client_id = simulation_manager.launchSimulation(gui=True, auto_step=True)
urdf_path = "grocery_aisle.urdf"
p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)

pepper = simulation_manager.spawnPepper(client_id,translation=[-4, -4, 0],quaternion=[0, 0, 0, 1],spawn_ground_plane=True)



#moving robot in one square path and offsetting by moving it to (0.5,0.5) for next path coverage


x=0.35

distance = 8
ang_vel = 1
y = 0
while distance>= 1:
    angle = 90
    lin_time = distance/x
    ang_time = math.radians(angle)/ang_vel

    pepper.move(x,y,0)
    time.sleep(lin_time)
    pepper.move(0,0,0)

    pepper.move(0,0,ang_vel)
    time.sleep(ang_time)
    pepper.move(0,0,0)

    pepper.move(x,y,0)
    time.sleep(lin_time)
    pepper.move(0,0,0)

    pepper.move(0,0,ang_vel)
    time.sleep(ang_time)
    pepper.move(0,0,0)

    pepper.move(x,y,0)
    time.sleep(lin_time)
    pepper.move(0,0,0)

    pepper.move(0,0,ang_vel)
    time.sleep(ang_time)
    pepper.move(0,0,0)

    pepper.move(x,y,0)
    time.sleep(lin_time)
    pepper.move(0,0,0)

    pepper.move(0,0,ang_vel)
    time.sleep(ang_time)
    pepper.move(0,0,0)
    
    #offset to start second loop 
    angle = 45
    ang_time = math.radians(angle)/ang_vel
    pepper.move(0,0,ang_vel)
    time.sleep(ang_time)
    pepper.move(0,0,0)

    distance_offset = math.sqrt(2)*0.5
    lin_time = distance_offset/x
    pepper.move(x,y,0)
    time.sleep(lin_time)
    pepper.move(0,0,0)

    angle = 45
    ang_time = math.radians(angle)/ang_vel
    pepper.move(0,0,-ang_vel)
    time.sleep(ang_time)
    pepper.move(0,0,0)

    distance -=1



