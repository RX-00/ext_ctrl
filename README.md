# Extended Controller
Repo for work on extending state-space control methods with reinforcement learning methods.

## Systems for testing
- cartpole Balance (LQR): 
    - inverted pendulum on a cart from gym 0.14.0
    - cartpole simulation in mujoco \
- cartpole Swing-Up: 
    - inverted pendulum on a cart from gym 0.14.0
    - cartpole simulation in mujoco \
- slip: spring loaded inverted pendulum in mujoco

## Project Organization
The root directory is split up into different system environments being that of the Cart-Pole and SLIP. Inside each will be directories of 2D equations of motion simulations, a mujoco simulation, and an implementation of the extended controller framework (ext_ctrl). Inside these directories will be different controllers ranging from state-space to RL learned policies. The extended controller directory is where the main research is done.

## Custom Envs
The custom Gymnasium environments used can be found and installed from here: [https://github.com/RX-00/ext_ctrl_envs]

## Running
NOTE: You might need to re-pip install the custom mujoco environments to make the non-nominal cartpole case to work. Currently not sure why this is the case.