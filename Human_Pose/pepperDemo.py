#Script that combines human landmarking with simulation environemt using Pepper's camera
#and a humanlike URDF to integrate the two systems
#Daniel Pawlak combining simulation members code
import sys
import pybullet as p
from qibullet import *
import cv2
import time
import mediapipe as mp
import numpy as np
from collections import deque

#Sim and pepper setup (code received from SLAM members w/ loaded human like urdf)
simulation_manager = SimulationManager()
client_id = simulation_manager.launchSimulation(gui=True)
p.setGravity(0,0,0) #Gravity off so its easier to manipulate sim human
pepper = simulation_manager.spawnPepper(client_id, translation=[-3.0, 0, 0], quaternion=[0, 0, 0, 1], spawn_ground_plane=True)
p.loadURDF("/home/pawlak/Desktop/perception/human2.urdf", basePosition=[0, 0, 1], useFixedBase=True)
i_see_you = pepper.subscribeCamera(PepperVirtual.ID_CAMERA_TOP, resolution=Camera.K_QVGA, fps=15.0) #Given pepper camera vision

#Mediapipe setup
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDrawing = mp.solutions.drawing_utils
waveHistory = deque(maxlen=10) #Used to track movement of wrist, higher number means more confirmation needed before flagging waving status

#Function that tracks development of waving using positions of right wrist and shoulder
def detectWave(landmarks):
    rightWrist = landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value]
    rightShoulder = landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value]
    #Checks if wrist is above shoulder then adds to history tracker
    if rightWrist.y < rightShoulder.y:
        waveHistory.append(rightWrist.x)
        #If history is longer equal to chosen parameter then check for waving motion (e.g left to right or vice versa)
        if len(waveHistory) == waveHistory.maxlen:
            diffs = np.diff(waveHistory)
            signChanges = np.count_nonzero(np.diff(np.sign(diffs)))
            #Sign chances gives the amount of waves needed (can be changed for strictness for more 'waves' before flagged)
            if signChanges >= 2:
                return True
    else:
    	#If wave is no longer developing then clear the memory
        waveHistory.clear()
    return False

#Function for checking if person is laying down by comaparing angles between upper chest (between shoulders) and middle of hips
def isLyingDown(landmarks):
    try:
    	#Finds positions on frame of landmarks then finds the midpoint by averaging them
        leftShoulder = np.array([landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y])
        rightShoulder = np.array([landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].y])
        leftHip = np.array([landmarks[mpPose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mpPose.PoseLandmark.LEFT_HIP.value].y])
        rightHip = np.array([landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].y])
        midShoulder = (leftShoulder + rightShoulder) / 2
        midHip = (leftHip + rightHip) / 2
        #Finds vector from positions then the resulting angle 
        torsoVector = midHip - midShoulder
        angle = np.degrees(np.arctan2(torsoVector[0], torsoVector[1]))
        return abs(angle) > 60 # Threshold; If more than 60 then person is closer to being horizontal than vertical
    except:
        return False

#Main loop of program; Gets image from pepper then applies landmarking if possible then the checks to label screen
try:
    while True:
        img = pepper.getCameraFrame(i_see_you)
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRgb)

        if results.pose_landmarks:
            mpDrawing.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            if isLyingDown(results.pose_landmarks.landmark):
                cv2.putText(img, "Lying on Floor", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            elif detectWave(results.pose_landmarks.landmark):
                cv2.putText(img, "Waving", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
            else:
                cv2.putText(img, "Standing", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        else:
            cv2.putText(img, "No person detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        displayFrame = cv2.resize(img, (1280, 720)) #Resize for bigger cv2 window if wanted
        cv2.imshow("Pepper Pose Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Closing")

finally:
    pepper.unsubscribeCamera(i_see_you)
    cv2.destroyAllWindows()
    if sys.version_info[0] >= 3:
        input("Press a key to end the simulation")
    else:
        raw_input("Press a key to end the simulation")
    simulation_manager.stopSimulation(client_id)

