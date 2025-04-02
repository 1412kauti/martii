import cv2
import time
import math
import threading
import pybullet as p
from qibullet import *

#Startup simulation
simulation_manager = SimulationManager()
client_id = simulation_manager.launchSimulation(gui=True, auto_step=True)

#Import Pepper and Obstacle
pepper = simulation_manager.spawnPepper(client_id, translation = [-4,-1,0],quaternion=[0, 0, 0, 1], spawn_ground_plane=True)
obstacle_path = "/home/devatreya/pepper/grocery_aisle.urdf"


obstacle=p.loadURDF(obstacle_path, basePosition=[0,0,0], useFixedBase=True)


lock = threading.Lock()
#saving laser values as dictionary to call later
curr_laser_value = {'Front': None, 'Right': None, 'Left': None}

#Function to 
def getPepperLasers(pepper, lock):
    pepper.subscribeLaser()
    pepper.showLaser(True)
    
    while True:
        with lock:
            curr_laser_value['Front'] = pepper.getFrontLaserValue()[-1]
            curr_laser_value['Right'] = pepper.getRightLaserValue()[-1]
            curr_laser_value['Left'] = pepper.getLeftLaserValue()[-1]
            print(f"Front laser = {curr_laser_value['Front']}, Right laser = {curr_laser_value['Right']}, Left laser = {curr_laser_value['Left']}")
        time.sleep(0.1)


#Function to check if obstacle is present
def ObstacleDetection():
    with lock:
        Front_laser = curr_laser_value['Front']#use front laser value to check with threshold
        
    print(f"Front laser: {Front_laser}")
    
    if float(Front_laser) < 2 :

        return True
    return False

#Function to check if path is clear
def ObstacleClear():
    with lock:
        Front_laser = curr_laser_value['Front'] #use front laser value to check with threshold
        
    print(f"Front laser: {Front_laser}") 
    
    if float(Front_laser) > 2 :

        return True
    return False

#Start thread for Laser 
Laser_thread = threading.Thread(target = getPepperLasers, args = (pepper,lock),daemon =True)
Laser_thread.start()

#Initial movement 
time.sleep(3)
pepper.move(0.35,0,0)
time.sleep(1/0.35)
pepper.move (0,0,0)

#First movement and obstacle check 
pepper.move(0.35,0,0)
time.sleep(1/0.35)
pepper.move (0,0,0)

if ObstacleDetection():
    pepper.move(0, 0, 0)
    print("Obstacle detected! Strafe Right")
    pepper.move(0, -0.35 ,0 )
    if ObstacleClear():
        pepper.move(0,0,0)
        time.sleep(1)    
    pepper.move(0,-0.35,0)
    time.sleep(6)

#Second movement and obstacle check 
pepper.move(0.35,0,0)
time.sleep(1/0.35)
pepper.move (0,0,0)

if ObstacleDetection():
    pepper.move(0, 0, 0)
    print("Obstacle detected! Strafe Right")
    pepper.move(0, -0.35 ,0 )
    if ObstacleClear():
        pepper.move(0,0,0)
        time.sleep(1)    
    pepper.move(0,-0.35,0)
    time.sleep(6)

#Third movement and obstacle check 
pepper.move(0.35,0,0)
time.sleep(1/0.35)
pepper.move (0,0,0)

if ObstacleDetection():
    pepper.move(0, 0, 0)
    print("Obstacle detected! Strafe Right")
    pepper.move(0, -0.35 ,0 )
    if ObstacleClear():
        pepper.move(0,0,0)
        time.sleep(1)    
    pepper.move(0,-0.35,0)
    time.sleep(6)



