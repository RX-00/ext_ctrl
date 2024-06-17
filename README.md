# Extended Controller
Repo for work on extending state-space control methods with reinforcement learning methods.
> We present a method to combine the advantages of the separate approaches for control, that being Reinforcement Learning (RL) and optimal control, into a single controller. Optimal control allows for precise movement and behavior but is restricted to a specific modelled system. RL is capable of being more generalizable to different environments but it is difficult to train a policy to perform specific desired motions. In this work we gather trajectories from an optimal linear quadratic regulator (LQR) controller for which an RL policy is trained on. We refer the resulting policy as an extended controller and evaluate its performance on the environment the LQR trajectories were taken from along with a novel environment that the extended controller has never experienced. The extended controller is able to exhibit the advantage of RL's generalizability while also capturing optimal controllers' smooth and precise movement.

TODO: clean up the English to be more elegant on the abstract

- Break down on trajectory optimization vs endpoint optimization (optimal control)
- What does the policy learn specifically and what it's supposed to do?
- How to evaluate generalizability
  - 3 nominal cases -> give it a fourth incline to evaluate [How different  should each environment be?]
  - maybe even have model variations?
  - transfer learning maybe for the more radical environmental changes?

- Ask big picture questions, how could this work be extended/re-purposed? Like something different from a cartpole problem
  - How would the ideas/approaches to formulation of reward functions be applied to other systems?

- Generalize the LQR setup -> work on the math of the whole thing -> where does the reward function and stuff come from?
  - general case n by n for formulation
  - how to approach explaining the problem?
  - then focus to the specific case (or go directly depending on time) 


## Systems for testing
- cartpole Balance (LQR):
    - inverted pendulum on a cart from gym 0.14.0
    - cartpole simulation in [mujoco]
- cartpole Swing-Up and Balance (LQR + Energy Shaping): 
    - inverted pendulum on a cart from gym 0.14.0
    - cartpole simulation in [mujoco]

## Project Organization
The root directory is split up into different system environments being that of the carpole tasks. Inside each will be directories of 2D equations of motion simulations, a mujoco simulation, and an implementation of the extended controller framework (ext_ctrl). Inside these directories will be different controllers ranging from state-space to RL learned policies. The extended controller directory is where the main research is done.

## Custom Envs
The custom Gymnasium environments used can be found and installed from here: [https://github.com/RX-00/ext_ctrl_envs]

## Running
NOTE: You might need to re-pip install the custom mujoco environments to make the non-nominal cartpole case to work. Currently not sure why this is the case.
\
To collect trajectories, navigate to the "/ext_ctrl/traj" folder within either task folder "cartpole_balance" or "cartpole_swingup". You can then run the following command:
```
python cartpole_LQR_trajs.py
```
To train a policy based on a state-space controller, navigate to the "/ext_ctrl" folder within either task folder "cartpole_balance" or "cartpole_swingup". You can then run the following command:
```
python main.py
```
Within the main function in this script "main.py" you should be able to quickly edit the function to evaluate the trained policy by uncommenting "test()"

## References
The xml model for the cartpole is based off of the "dm_control" package from Google Deepmind mentioned here: [dm_publication]
```
@article{tunyasuvunakool2020,
         title = {dm_control: Software and tasks for continuous control},
         journal = {Software Impacts},
         volume = {6},
         pages = {100022},
         year = {2020},
         issn = {2665-9638},
         doi = {https://doi.org/10.1016/j.simpa.2020.100022},
         url = {https://www.sciencedirect.com/science/article/pii/S2665963820300099},
         author = {Saran Tunyasuvunakool and Alistair Muldal and Yotam Doron and
                   Siqi Liu and Steven Bohez and Josh Merel and Tom Erez and
                   Timothy Lillicrap and Nicolas Heess and Yuval Tassa},
}
```



[mujoco]: https://mujoco.org/
[dm_publication]: https://doi.org/10.1016/j.simpa.2020.100022
