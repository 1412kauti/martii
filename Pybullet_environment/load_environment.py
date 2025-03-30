from qibullet import SimulationManager
import pybullet as p

# Start simulation
simulation_manager = SimulationManager()
client_id = simulation_manager.launchSimulation(gui=True)

# Spawn Pepper at one end of the aisle (e.g., 2.5 meters behind the center)
pepper = simulation_manager.spawnPepper(client_id, translation=[0, -2.5, 0])

# Path to your URDF aisle
urdf_path = "/home/devatreya/pepper/grocery_aisle.urdf"

# Load the aisle in the center
p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
