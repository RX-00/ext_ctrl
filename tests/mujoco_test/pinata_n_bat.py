import numpy as np
import mujoco
import mujoco_viewer
import mediapy as media
import matplotlib.pyplot as plt

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

model = mujoco.MjModel.from_xml_path('/home/robo/ext_ctrl/tests/mujoco_test/models/pinata_n_bat.xml')
data = mujoco.MjData(model)

times = []
sensor_data = []
# time step of the graphs, not the simulation itself
dt = 0.005 # the smaller the timestep, the more accurate the energy conservation
duration = 10 # seconds of simulation time

# mujoco viewer
viewer = mujoco_viewer.MujocoViewer(model, data)

# constant actuator control signal
mujoco.mj_resetData(model, data)
data.ctrl = 20

# simulate and render
while data.time < duration:
  if viewer.is_alive:
    mujoco.mj_step(model, data)

    # saving data
    times.append(data.time)
    sensor_data.append(data.sensor('accelerometer').data.copy())

    viewer.render()
  else:
    break

# close viewer render
viewer.close()

# plot data
ax = plt.gca()

ax.plot(np.asarray(times), np.asarray(sensor_data), label='timestep = {:2.2g}ms'.format(1000*dt))

# finalize plot
ax.set_title('Accelerometer values')
ax.set_ylabel('meter/second^2')
ax.set_xlabel('second')
ax.legend(frameon=True, loc='lower right');
plt.tight_layout()

plt.show()