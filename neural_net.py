from operator import truediv
from re import I
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
import sys
from tqdm import tqdm
import pandas as pd
path = os.path.abspath(os.getcwd())
ui_path = path + '/ui'
sys.path.insert(1, ui_path)
from render_graphics import render, gif

if(torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

env = gym.make('CartPole-v0').unwrapped

#remove this later, for labeling the frames with episode number
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
        self.fc1 = nn.Linear(inputs, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64, outputs)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

def get_observation():
  return torch.tensor(env.state).float().reshape(1,4)

BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 2500
TARGET_UPDATE = 1000
learningRate = 0.001


# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(4, n_actions).to(device)
target_net = DQN(4, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=learningRate)

def set_batch(size):
    BATCH_SIZE = size

def set_learningRate(rate):
    optimizer = optim.RMSprop(policy_net.parameters(), lr=rate)

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
    #print('loss is', loss)
    # Add current loss to the loss_history
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
loss_history = []
memory = ReplayMemory(65536)
num_episodes = 3000
frames = [] #we store the frames in here 
best_episode=np.NINF #initialize best_episode length
new_best = False     #variable for new best episode length 
a = 0                #we use this just to compare the best_episode to this
for i_episode in tqdm(range(num_episodes)):
    # Initialize the environment and state
    env.reset()
    state = get_observation()
    loss_history.append(0)

    float_states = []
    for t in count():

        float_states.append([env.state[0],env.state[2]]) #for rendering 
        # Select and perform an action
        action = select_action(state)
    
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
        if len(memory) < BATCH_SIZE:
          pass
        else:
          loss_history[-1] += optimize_model()
            
        if done:
            episode_durations.append(t + 1)
            float_states.append([123,123])
            break

    #resets new_best 
    new_best = False
    # Update the target neFtwork, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    #this figures out if the last episode is the new best episode    
    """best_episode = max(best_episode, int(episode_durations[-1]))
    #a is set to the new best episode
    if(a!=best_episode):
        new_best = True
        a = best_episode

        #renders frames
        for i in float_states:
            if(i[0]==123 and i[1] == 123):
                for j in range(30):
                    frame=render(float_states[-2][0],float_states[-2][1],True)
                    frame = _label_with_episode_number(frame,episode_num=i_episode)
                    frames.append(frame)
            else:
                frame = render(i[0],i[1]) #renders frame
                frame = _label_with_episode_number(frame,episode_num=i_episode)
                frames.append(frame) #appends frame to frame list
             #labels frame (relocate this to rendering function later)"""
           
           
    
print('Complete')
env.close()
# Plot loss
# printing the list using loop
loss_history = pd.Series([float(x) for x in loss_history]).rolling(20).mean().tolist()
x = np.arange(len(loss_history))
y = loss_history
x2 = np.arange(len(episode_durations))
y2 = episode_durations
plt.plot(x, y)
plt.show()        #plot 1 = loss
plt.plot(x2, y2)
plt.show()        #plot 2 = episode durations
# Initialize the plot
#combines renderings into gif
#imageio.mimwrite(os.path.join('./videos/', 'render.gif'), frames, fps=60)
#os.system("ffmpeg -y -i ./videos/render.gif -movflags faststart -pix_fmt yuv420p -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" video.mp4")
#print("done")


