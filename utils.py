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
        self.h_size = 128
        
        self.output_layer = nn.Sequential(nn.Linear(self.h_size,self.h_size),
                                          nn.ReLU(),
                                          nn.Dropout(0.2),
                                          nn.Linear(self.h_size,1),
                                         )
        
        self.stream_layer = nn.Sequential(nn.Linear(self.h_size,self.h_size),
                                          nn.ReLU(),
                                          nn.Linear(self.h_size,1),
                                         )
        
        self.conv_layer = nn.Sequential(nn.Conv1d(self.num_sensor,self.h_size-1,kernel_size = 36),
                                        nn.ReLU(),
                                       )
        
        self.fc_layer = nn.Sequential(nn.Linear(self.h_size+1,self.h_size),
                                          nn.ReLU(),
                                          nn.Linear(self.h_size,self.h_size),
                                         )
        
    def forward(self,state,action):
        batch_size = state.shape[0]
        
        # action
        action = self.conv_layer(action.permute(0,2,1)).reshape(batch_size,-1)
        
        # combine state action
        combine = torch.cat((state,action),dim=-1)
        
        # fc forward
        combine = self.fc_layer(combine)
        
        # get output and stream
        output = self.output_layer(combine)
        stream = self.stream_layer(combine)
        
        return F.sigmoid(output),F.sigmoid(stream)

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_step = data['action'].shape[1]
        self.num_sensor = data['action'].shape[2]
        self.flat_size = self.time_step*self.num_sensor
        self.fc = nn.Sequential(nn.Linear(3,128),nn.ReLU(),nn.Linear(128,self.flat_size))
        
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
        self.lasso_w = lasso_w

    def get_predict(self,s,a):
        
        feed = a.iloc[0,0]
        action = pd.DataFrame(index=[*range(36)],columns=self.action_col)
        for i in range(36):
            action.iloc[i,:] = a.values
        action = action.values
        
        state = self.mm_state.transform([s])
        action = self.mm_action.transform(action)
        action = np.expand_dims(action,axis=0)

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)

        # critic forward but not predict stream
        output,_ = self.critic(state.cuda(),action.cuda())

        # lasso predict stream
        batch_size = action.shape[0]
        A = torch.cat((action.reshape(batch_size,-1),state),dim=-1)
        stream = (A@self.lasso_w).reshape(-1,1)

        # inverse transform
        output = output.detach().cpu().numpy()
        output = self.mm_output.inverse_transform(output)[0][0]
        stream = stream.detach().cpu().numpy()
        stream = self.mm_stream.inverse_transform(stream)[0][0]

        return output,stream,feed/output,feed/stream
    
    def get_advice(self,state,request):
        
        # sacle input
        request = self.mm_output.transform([[request]])
        state = self.mm_state.transform([state])
        
        # tensor format input
        request = torch.FloatTensor([request]).reshape(-1,1)
        state = torch.FloatTensor(state)
        
        # actor forward
        action = self.actor(state,request)
        
        # critic forward but not predict stream
        output,_ = self.critic(state.cuda(),action.cuda())
        
        # lasso predict stream
        batch_size = action.shape[0]
        A = torch.cat((action.reshape(batch_size,-1),state),dim=-1)
        stream = (A@self.lasso_w).reshape(-1,1)
        
        # inverse transform
        output = output.detach().cpu().numpy()
        output = self.mm_output.inverse_transform(output)
        stream = stream.detach().cpu().numpy()
        stream = self.mm_stream.inverse_transform(stream)
        action = action.detach().cpu().numpy()
        action = np.array([self.mm_action.inverse_transform(i) for i in action]).squeeze(0)
        
        # create advice DataFrame
        advice = pd.DataFrame(index = self.action_col)
        advice['chinese'] = advice.index.map(self.tag_map) 
        advice['mean'] = action.mean(axis=0)
        advice['max'] = action.max(axis=0)
        advice['min'] = action.min(axis=0)
        
        # feed
        feed = advice.loc['MLPAP_FQ-0619.PV','mean']
        return advice,output,stream,feed/output[0][0],feed/stream[0][0]