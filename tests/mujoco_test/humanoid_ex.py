import numpy as np
import mujoco
import mujoco_viewer
import mediapy as media
import matplotlib.pyplot as plt

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

model = mujoco.MjModel.from_xml_path('/home/robo/ext_ctrl/mujoco_test/models/humanoid.xml')
data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)

# simulate and render
for _ in range(10000):
    if viewer.is_alive:
        # mujoco use generalized coordinates (Lagrangian)
        mujoco.mj_step(model, data) # same as mj_forward, but this integrates in time
        viewer.render()
    else:
        break

# close
viewer.close()

# print information about the model and its data
print("Total number of DOFs in the model: ", model.nv)
print("Generalized positions: ", data.qpos)
print("Generalized velocities: ", data.qvel)