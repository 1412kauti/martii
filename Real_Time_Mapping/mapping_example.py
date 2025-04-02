import pybullet as p
from qibullet import *
import time
import math
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def spawn_env():
    simulation_manager = SimulationManager()
    client_id = simulation_manager.launchSimulation(gui=True)
    pepper = simulation_manager.spawnPepper(client_id, translation=[0,0,0], quaternion=[0, 0, 0, 1], spawn_ground_plane=True)
    return simulation_manager,client_id,pepper

def robot_speed():
    def_lin_vel = 0.35 # in (m/s)
    def_ang_vel = 1.0 # in (rad/s)
    return def_lin_vel, def_ang_vel

def map_path_config():
    map_size = 8.0 # in Meters(m) 
    angle_turn = 90 # in Degrees
    return map_size,angle_turn

def time_linear(distance,lin_vel):
    time_taken = distance/lin_vel
    return time_linear

def time_angular(angle,ang_vel):
    time_taken = (math.radians(angle)/ang_vel)
    return time_taken

def update_positions(positions,pepper):
    while True:
        pos, _ = p.getBasePositionAndOrientation(pepper.robot_model)
        positions.append(pos[:2])
        time.sleep(0.1)

def move_robot(pepper,x,distance,y,ang_vel):
    while True:
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
def main():
    simulation_manager,client_id,pepper = spawn_env()
    lin_vel , ang_vel = robot_speed()
    distance , angle = map_path_config()
    positions = []
    position_thread = threading.Thread(target=update_positions, args=(positions,pepper) ,daemon=True)
    position_thread.start()

    fig, ax = plt.subplots()
    sc, = ax.plot([], [], 'bo-', markersize=10)  # Blue dot for movement
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Pepper Map")
    ax.grid(True)

    def animate(frame):
        """ Update plot with new positions """
        if positions:
            x_data, y_data = zip(*positions)
            sc.set_data(x_data, y_data)
            ax.set_xlim(min(x_data) - 1, max(x_data) + 1)
            ax.set_ylim(min(y_data) - 1, max(y_data) + 1)

    ani = animation.FuncAnimation(fig, animate, interval=200)

    move_thread = threading.Thread(target=move_robot,args=(pepper,lin_vel,distance,0,ang_vel), daemon=True)
    move_thread.start()

    plt.show()

if __name__=='__main__':
    main()
