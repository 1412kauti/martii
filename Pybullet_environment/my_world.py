import sys
import pybullet as p
from qibullet import SimulationManager

simulation_manager = SimulationManager()

client_id = simulation_manager.launchSimulation(gui=True)

pepper = simulation_manager.spawnPepper(client_id,translation=[0, -3, 0],quaternion=[0, 0, 0, 1],spawn_ground_plane=True)

urdf_path = "/home/devatreya/grocery_aisle.urdf"
p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)


if sys.version_info[0] >= 3:
	input("Press a key to end the simulation")
else:
	raw_input("Press a key to end the simulation")
    
#stop simulatio
simulation_manager.stopSimulation(client_id)