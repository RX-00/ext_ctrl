import numpy as np
import mujoco
import mujoco_viewer
import mediapy as media
import matplotlib.pyplot as plt

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

model = mujoco.MjModel.from_xml_path('/home/robo/ext_ctrl/tests/mujoco_test/models/cartpole_nominal.xml')
data = mujoco.MjData(model)

times = []
duration = 1000 # seconds of simulation time

# mujoco viewer
viewer = mujoco_viewer.MujocoViewer(model, data)

#mujoco.mj_resetData(model, data)
#data.ctrl = 0 # passive system

# simulate and render
while data.time < duration:
  if viewer.is_alive:
    mujoco.mj_step(model, data)

    viewer.render()
  else:
    break

# close viewer render
viewer.close()