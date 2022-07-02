import gym
import os
import math
import random
import glob
import imageio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import PIL.ImageDraw as ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.animation
import io
from pyqtgraph.Qt import QtWidgets
import sys
import base64
import pyqtgraph as pg


device = torch.device("cuda")

print(torch.cuda.is_available())

env = gym.make('CartPole-v0').unwrapped


def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    drawer.text((im.size[0]/20,im.size[1]/18), f'Episode: {episode_num+1}', fill=text_color)

    return im


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, 128)
        self.fc2 = nn.Linear(128, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

def get_observation():
  return torch.tensor(env.state).float().reshape(1,4)

BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
learningRate = 0.01

def set_batch(size):
    BATCH_SIZE = size

def set_learningRate(rate):
    learningRate = rate

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(4, n_actions).to(device)
target_net = DQN(4, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr = learningRate)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state.reshape(1,-1)).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []
loss_history = []
def optimize_model():
    global loss_history
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    "print(state_batch.shape)"
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # Add current loss to the loss_history
    loss_history.append(loss)
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

num_episodes = 2000
frames = []
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    state = get_observation()
    for t in count():
        # Select and perform an action
        action = select_action(state) #this does not work with observation instead of pictures?

        if(i_episode % 100 == 0):
            frame = env.render(mode='rgb_array')
            frames.append(_label_with_episode_number(frame, i_episode))

        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        
        if not done:
          next_state = get_observation()
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network) and save loss in loss_history
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            break
    # Update the target neFtwork, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')

env.close()

# Plot loss
# printing the list using loop
loss_history = [float(i) for i in loss_history]
loss_history = [min(10,i) for i in loss_history]
x = np.arange(len(loss_history))
y = loss_history
# Initialize the plot
plt = pg.plot()
# Set the plot values
line = plt.plot(x, y)

#combines renderings into gif 
imageio.mimwrite(os.path.join('./videos/', 'random_agent.gif'), frames, fps=60)
print("done") 

#starts Qt so we can look at plot
if __name__ == '__main__':
    # Start Qt event loop unless running in interactive mode
    if sys.flags.interactive != 1:
        QtWidgets.QApplication.instance().exec()
