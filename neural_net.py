from operator import truediv
from re import I
import os
import math
import random
import glob
from venv import create
import imageio
from threading import Thread
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import io
import sys
from tqdm import tqdm
import requests
import pandas as pd
from threading import Thread
path = os.path.abspath(os.getcwd())
ui_path = path + '/ui'
sys.path.insert(1, ui_path)
from render_graphics import render
from physics import step, State

"""if(torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")"""
device = torch.device("cuda")

#env = gym.make('CartPole-v0').unwrapped



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



class DQN1(nn.Module):
    def __init__(self, inputs, nodes1, outputs):
        super(DQN1, self).__init__()
        self.fc1 = nn.Linear(inputs, nodes1)
        self.fc2 = nn.Linear(nodes1, outputs)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class DQN2(nn.Module):
    def __init__(self, inputs, nodes1, nodes2, outputs):
        super(DQN2, self).__init__()
        self.fc1 = nn.Linear(inputs, nodes1)
        self.fc2 = nn.Linear(nodes1, nodes2)
        self.fc3 = nn.Linear(nodes2, outputs)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class DQN3(nn.Module):
    def __init__(self, inputs, nodes1, nodes2, nodes3, outputs):
        super(DQN3, self).__init__()
        self.fc1 = nn.Linear(inputs, nodes1)
        self.fc2 = nn.Linear(nodes1, nodes2)
        self.fc3 = nn.Linear(nodes2, nodes3)
        self.fc4 = nn.Linear(nodes3, outputs)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x



"""def get_observation():
  return torch.tensor(env.state).float().reshape(1,4)"""

GAMMA = 0.8
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 100




#optimizer = optim.RMSprop(policy_net.parameters(), lr=learningRate)

def set_batch(size):
    global BATCH_SIZE
    BATCH_SIZE = size

def set_targetUpdate(x):
    global TARGET_UPDATE
    TARGET_UPDATE = x

def set_nrEpisodes(x):
    global num_episodes
    num_episodes = x

def set_learningRate(rate):
    global optimizer
    optimizer = optim.RMSprop(policy_net.parameters(), lr=rate)

def set_layers(x):
    # 1-3 layer
    global numberOfLayers
    numberOfLayers = x

def set_neurons1(x):
    global neurons1
    neurons1 = x

def set_neurons2(x):
    global neurons2
    neurons2 = x

def set_neurons3(x):
    global neurons3
    neurons3 = x

def set_policy_net():
    global n_actions
    n_actions = 2
    global policy_net
    global target_net
    if numberOfLayers == 1:
        policy_net = DQN1(4, neurons1, n_actions).to(device)
        target_net = DQN1(4, neurons1, n_actions).to(device)
    if numberOfLayers == 2:
        policy_net = DQN2(4, neurons1, neurons2, n_actions).to(device)
        target_net = DQN2(4, neurons1, neurons2, n_actions).to(device)
    if numberOfLayers == 3:
        policy_net = DQN3(4, neurons1, neurons2, neurons3, n_actions).to(device)
        target_net = DQN3(4, neurons1, neurons2, neurons3, n_actions).to(device)
    else:
        policy_net = DQN1(4, 128, n_actions).to(device)
        target_net = DQN1(4, 128, n_actions).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()


# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
# Get number of actions from gym action space



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

global loss_history, episode_durations
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


def createVideo(frames):
    imageio.mimwrite(os.path.join('./static/', 'render.gif'), frames, fps=30)
    os.system("ffmpeg -y -i ./static/render.gif -movflags faststart -pix_fmt yuv420p -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" ./static/video.mp4")

global frames, frame_count, a, best_episode, memory, new_best, counter, best_frames
memory = ReplayMemory(65536)
def start_learning():
    print(BATCH_SIZE)
    global num_episodes
    frame_count = 0
    a = 0                #we use this just to compare the best_episode to this
    best_episode=np.NINF #initialize best_episode length
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    loss_history = []
    counter =0
    best_frames = []
    new_best = False     #variable for new best episode length
    
    global episode
    episode = 0
    if(os.path.exists("model.pt")):
        print("reading!")
        checkpoint = torch.load("model.pt")
        policy_net.load_state_dict(checkpoint['policy_state_dict'])
        target_net.load_state_dict(checkpoint['target_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        episode = checkpoint['i_episode']
        num_episodes = num_episodes - episode
        best_episode = checkpoint['best_episode']
        best_frames = checkpoint['best_frames']
        loss_history = checkpoint['loss_history']
        global episode_durations
        episode_durations = checkpoint['episode_durations']
        print(num_episodes)
        policy_net.eval()
        with open('static/flag', 'w') as f:
                    f.write('0')
                    f.close
    for i_episode in tqdm(range(num_episodes)):
        frames = []
        counter += 1
        with open('static/dfile.txt', 'w') as f:
            f.write(str(100 * (counter / num_episodes))+','+str(i_episode+episode))

        # Initialize the environment and state
        # create random state
        randState = State(0.0, 0.0, 0.0, 0.0)
        loss_history.append(0)
        float_states = []
        state = torch.tensor([randState.coord, randState.speed, randState.angle, randState.ang_speed]).float().reshape(1,4)
        

        if(frame_count>900):
                break
        for t in count():
            frame_count += 1
            if(t > 900):
                break
            float_states.append([randState.coord,randState.angle]) #for rendering 
            # Select and perform an action
            action = select_action(state)
    
            #_, reward, done, _ = env.step(action.item())
            done = step(randState, action)
            if(not done):
                reward = 1.0
            else:
                reward = 0.0

            reward = torch.tensor([reward], device=device)

            if not done:
                next_state = torch.tensor([randState.coord, randState.speed, randState.angle, randState.ang_speed]).float().reshape(1,4)
            else:
                next_state = None
            # Store the transition in memory
            memory.push(state, action, next_state, reward)
            # Move to the next staSte
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
        frame_count = 0
        y = pd.Series([float(x) for x in loss_history]).rolling(20).mean().tolist()
        y2 = pd.Series(episode_durations).rolling(20).mean()
        line=str(y[i_episode])
        with open('static/loss.txt', 'w') as f:
            f.write(line+','+str(i_episode+episode))
        line=str(y2[i_episode])
        with open('static/episodes.txt', 'w') as f:
            f.write(line+','+str(i_episode+episode)) 
        # resets new_best 
        new_best = False
        # Update the target neFtwork, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        #this figures out if the last episode is the new best episode    
        best_episode = max(best_episode, int(episode_durations[-1]))
        #a is set to the new best episode


        if(i_episode >= 99 and i_episode%100 == 0):
            for i in float_states:
                if(i[0]==123 and i[1] == 123):
                    for j in range(30):
                        frame=render(float_states[-2][0],float_states[-2][1],True,i_episode+episode, int(episode_durations[-1]))
                        frames.append(frame)
                else:
                    frame = render(i[0],i[1],False,i_episode+episode, int(episode_durations[-1])) #renders frame
                    frames.append(frame) #appends frame to frame list
            thread = Thread(target=createVideo(frames))
            thread.start()
            thread.join()
            
            

        if(a!=best_episode):
            new_best = True
            a = best_episode
            best_frames = []
            print(a)
            #renders frames
            for i in float_states:
                if(i[0]==123 and i[1] == 123):
                    for j in range(30):
                        frame=render(float_states[-2][0],float_states[-2][1],True,i_episode+episode, best_episode)
                        best_frames.append(frame)
                else:
                    frame = render(i[0],i[1],False,i_episode+episode, best_episode) #renders frame
                    best_frames.append(frame) #appends frame to frame list
        with open('static/flag', 'r') as f:
            if '1' in f.read():
                torch.save({
                'i_episode': i_episode+episode,
                'policy_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'target_state_dict': target_net.state_dict(),
                'loss_history': loss_history,
                'best_episode': best_episode,
                'best_frames': best_frames,
                'episode_durations': episode_durations}, "model.pt")
                print("true")
                f.close()
                with open('static/flag', 'w') as f:
                    f.write('0')
                    f.close
                break
    print('Complete')
    thread = Thread(target=createVideo(frames))
    thread.start()
    thread.join()
