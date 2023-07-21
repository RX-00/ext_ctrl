# ----------- Libraries -----------
import numpy as np

from collections import deque

import matplotlib.pyplot as plt
#%matplotlib inline # uncomment for Jupyter notebook

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Gym
import gym


#------------- Enviroment -----------
env_id = "CartPole-v1"
# Create the env
env = gym.make(env_id)

# Get the state space and action space
#s_size = env.observation_space.shape[0]
s_size = 4
a_size = env.action_space.n

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#------------ Policy --------------
class Policy(nn.Module):
  def __init__(self, s_size, a_size, h_size):
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      super(Policy, self).__init__()
      # Create two fully connected layers
      self.fc1 = nn.Linear(s_size, h_size)
      self.fc2 = nn.Linear(h_size, a_size)


  def forward(self, x):
      # Define the forward pass
      # state goes to fc1 then we apply ReLU activation function
      x = F.relu(self.fc1(x))
      # fc1 outputs goes to fc2
      x = self.fc2(x)
      # We output the softmax
      return F.softmax(x, dim=1)
  
  def act(self, state):
      """
      Given a state, take action
      """
      state = torch.from_numpy(state).float().unsqueeze(0).to(device)
      probs = self.forward(state).cpu()
      m = Categorical(probs)
      action = m.sample()
      return action.item(), m.log_prob(action)

#--------------- Reinforce --------------
def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
  # Help us to calculate the score during the training
  scores_deque = deque(maxlen=100)
  scores = []
  # Line 3 of pseudocode
  for i_episode in range(1, n_training_episodes+1):
      saved_log_probs = []
      rewards = []
      state =  env.reset()
      #env.render() # if you want to display the env (will make training take a lot longer)
      # Line 4 of pseudocode
      for t in range(max_t):
          action, log_prob = policy.act(state)
          saved_log_probs.append(log_prob)
          state, reward, done, _ = env.step(action)
          # penalizing if you drift the cart too much
          if state[0] > 0.2 or state[0] < -0.2:
              reward = -100
              done = True
          rewards.append(reward)
          if done:
              break 
      scores_deque.append(sum(rewards))
      scores.append(sum(rewards))
      
      # Line 6 of pseudocode: calculate the return
      returns = deque(maxlen=max_t) 
      n_steps = len(rewards) 
      # Compute the discounted returns at each timestep,
      # as the sum of the gamma-discounted return at time t (G_t) + the reward at time t
      
      # In O(N) time, where N is the number of time steps
      # (this definition of the discounted return G_t follows the definition of this quantity 
      # shown at page 44 of Sutton&Barto 2017 2nd draft)
      # G_t = r_(t+1) + r_(t+2) + ...
      
      # Given this formulation, the returns at each timestep t can be computed 
      # by re-using the computed future returns G_(t+1) to compute the current return G_t
      # G_t = r_(t+1) + gamma*G_(t+1)
      # G_(t-1) = r_t + gamma* G_t
      # (this follows a dynamic programming approach, with which we memorize solutions in order 
      # to avoid computing them multiple times)
      
      # This is correct since the above is equivalent to (see also page 46 of Sutton&Barto 2017 2nd draft)
      # G_(t-1) = r_t + gamma*r_(t+1) + gamma*gamma*r_(t+2) + ...
      
      
      ## Given the above, we calculate the returns at timestep t as: 
      #               gamma[t] * return[t] + reward[t]
      #
      ## We compute this starting from the last timestep to the first, in order
      ## to employ the formula presented above and avoid redundant computations that would be needed 
      ## if we were to do it from first to last.
      
      ## Hence, the queue "returns" will hold the returns in chronological order, from t=0 to t=n_steps
      ## thanks to the appendleft() function which allows to append to the position 0 in constant time O(1)
      ## a normal python list would instead require O(N) to do this.
      for t in range(n_steps)[::-1]:
          disc_return_t = (returns[0] if len(returns)>0 else 0)
          returns.appendleft(  gamma*disc_return_t + rewards[t]  )
     
      ## standardization of the returns is employed to make training more stable
      eps = np.finfo(np.float32).eps.item()
      
      ## eps is the smallest representable float, which is 
      # added to the standard deviation of the returns to avoid numerical instabilities
      returns = torch.tensor(returns)
      returns = (returns - returns.mean()) / (returns.std() + eps)
      
      # Line 7:
      policy_loss = []
      for log_prob, disc_return in zip(saved_log_probs, returns):
          policy_loss.append(-log_prob * disc_return)
      policy_loss = torch.cat(policy_loss).sum()
      
      # Line 8: PyTorch prefers gradient descent 
      optimizer.zero_grad()
      policy_loss.backward()
      optimizer.step()
      
      if i_episode % print_every == 0:
          print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
      
  return scores

# ---------- Training Hyperparameters ----------
cartpole_hyperparameters = {
  "h_size": 16,
  "n_training_episodes": 5000,
  "n_evaluation_episodes": 100, # batch size
  "max_t": 1000,
  "gamma": 1.0,
  "lr": 1e-2, #learning rate
  "env_id": env_id,
  "state_space": s_size,
  "action_space": a_size,
}


def train():
    # ---------- Policy and optimizer ----------
    # Create policy and place it to the device
    cartpole_policy = Policy(cartpole_hyperparameters["state_space"], cartpole_hyperparameters["action_space"], cartpole_hyperparameters["h_size"]).to(device)
    cartpole_optimizer = optim.Adam(cartpole_policy.parameters(), lr=cartpole_hyperparameters["lr"])

    # --------- Training -----------
    scores = reinforce(cartpole_policy,
                       cartpole_optimizer,
                       cartpole_hyperparameters["n_training_episodes"], 
                       cartpole_hyperparameters["max_t"],
                       cartpole_hyperparameters["gamma"], 
                       100)

    torch.save(cartpole_policy.state_dict(), "/home/robo/Documents/cartpole/DQN/q-fxn.pth")

def test():
    env = gym.make('CartPole-v1')
    model = Policy(cartpole_hyperparameters["state_space"], cartpole_hyperparameters["action_space"], cartpole_hyperparameters["h_size"]).to(device)
    model.load_state_dict(torch.load("q-fxn.pth", map_location=device))
    state = env.reset()
    print("Size of reset env dimension: ", state)
    done = False
    i = 0
    x_positions = [state[0]]
    theta_positions = [state[2]]
    while not done:
        env.render()
        """
        Given a state, take action
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_values = model(state_tensor)
        action = np.argmax(action_values.numpy())
        next_state, reward, done, _ = env.step(action)
        state = next_state
        x_positions.append(state[0])
        theta_positions.append(state[2])
        i += 1
        if done:
            print("Done with score: {}".format(i))
            break
    
    env.close()

    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].plot(x_positions)
    axs[0].set_title('Cart position')
    axs[0].set_xlabel('Time step')
    axs[0].set_ylabel('Position')

    fig.suptitle('Cartpole DQN Controller', fontsize=16)

    axs[1].plot(theta_positions)
    axs[1].set_xlabel('Time step')
    axs[1].set_title('Pendulum angular position')
    axs[1].set_ylabel('Radians')

    plt.show()


if __name__ == "__main__":
    #train()
    test()