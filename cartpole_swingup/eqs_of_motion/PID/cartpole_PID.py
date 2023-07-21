import numpy as np
import gym
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

env = gym.make('CartPole-v1')
desired_state = np.array([0, 0, 0, 0])
desired_mask = np.array([0, 0, 1, 0])

P, I, D = 0.1, 0.01, 0.5

for i_episode in range(1):
    state = env.reset()
    integral = 0
    derivative = 0
    prev_error = 0
    x_positions = [state[0]]
    theta_positions = [state[2]]

    for t in range(500):
        env.render()
        error = state - desired_state

        integral += error
        derivative = error - prev_error
        prev_error = error

        pid = np.dot(P * error + I * integral + D * derivative, desired_mask)
        action = sigmoid(pid)
        action = np.round(action).astype(np.int32)

        state, reward, done, info = env.step(action)
        x_positions.append(state[0])
        theta_positions.append(state[2])

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()


fig, axs = plt.subplots(2, 1, constrained_layout=True)
axs[0].plot(x_positions)
axs[0].set_title('Cart position')
axs[0].set_xlabel('Time step')
axs[0].set_ylabel('Position')

fig.suptitle('Cartpole PID Controller', fontsize=16)

axs[1].plot(theta_positions)
axs[1].set_xlabel('Time step')
axs[1].set_title('Pendulum angular position')
axs[1].set_ylabel('Radians')

plt.show()
