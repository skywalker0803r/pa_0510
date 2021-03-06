from torch import nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
import warnings
warnings.simplefilter('ignore')
import torchviz
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import joblib
import os

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_step = time_step
        self.num_sensor = num_sensor
        self.h_size = 64
        
        self.output_layer = nn.Sequential(nn.Linear(self.h_size,self.h_size),
                                          nn.ReLU(),
                                          nn.Linear(self.h_size,1),
                                         )
        
        self.stream_layer = nn.Sequential(nn.Linear(self.h_size,self.h_size),
                                          nn.ReLU(),
                                          nn.Linear(self.h_size,1),
                                         )
        
        self.conv_layer = nn.Sequential(nn.Conv1d(self.num_sensor,self.h_size-1,kernel_size = 36),
                                        nn.ReLU(),
                                       )
        
    def forward(self,state,action):
        batch_size = state.shape[0]
        
        action = self.conv_layer(action.permute(0,2,1))
        action = action.reshape(batch_size,-1)
        
        combine = torch.cat((state,action),dim=-1)
        
        output = self.output_layer(combine)
        stream = self.stream_layer(combine)
        
        return F.sigmoid(output),F.sigmoid(stream)

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_step = 36
        self.num_sensor = 13
        self.flat_size = self.time_step*self.num_sensor
        self.fc = nn.Sequential(nn.Linear(2,128),nn.ReLU(),nn.Linear(128,self.flat_size))
        
    def forward(self,state,request):
        action = self.fc(torch.cat((state,request),dim=1))
        action = action.view(-1,self.time_step,self.num_sensor)
        return F.sigmoid(action)

class PA_ROBOT:
    def __init__(self):
        self.mm_output = data['mm_output']
        self.mm_stream = data['mm_stream']
        self.mm_state = data['mm_state']
        self.mm_action = data['mm_action']
        self.action_col = data['action_col']
        self.tag_map = tag_map
        self.actor = actor
        self.critic = critic
    
    def get_advice(self,state,request):
        # sacle inpus
        request = self.mm_output.transform([[request]])
        state = self.mm_state.transform([[state]])
        
        # tensor input
        request = torch.FloatTensor([request]).cuda().reshape(-1,1)
        state = torch.FloatTensor([state]).cuda().reshape(-1,1)
        
        # actor forward
        action = self.actor(state,request)
        
        # critic forward
        output,stream = self.critic(state,action)
        output = output.detach().cpu().numpy()
        stream = stream.detach().cpu().numpy()
        output = self.mm_output.inverse_transform(output)
        stream = self.mm_stream.inverse_transform(stream)
        
        action = action.detach().cpu().numpy()
        action = np.array([self.mm_action.inverse_transform(i) for i in action]).squeeze(0)
        advice = pd.DataFrame(index = self.action_col)
        advice['chinese'] = advice.index.map(self.tag_map) 
        advice['mean'] = action.mean(axis=0)
        advice['max'] = action.max(axis=0)
        advice['min'] = action.min(axis=0)
        
        return advice,output,stream