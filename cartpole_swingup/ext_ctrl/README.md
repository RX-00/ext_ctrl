# Extended Control Framework

## Trajectory Collecting
/traj
|   cartpole_LQR_trajs.py
|   /trajs

## PPO RL | Nominal Policy
/ppo_nominal_policy
|   ppo.py
|   train_policy.py
|   test_policy.py

## Transfer Learning | Extended Nonnominal Policy
/tf_nonnom_policy
|   TODO

## Usage
1. Collect trajectories from the state-space controller in the nominal environment.
2. The a policy based on the collected trajectories to achieve similar behavior.
3. Test this learned policy in the nominal environment.

# TODO:
- Finish PPO RL (Usage steps 2 and 3)
- Write up framework for transfer learning section