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

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_step = 36
        self.num_sensor = 13
        self.flat_size = self.time_step*self.num_sensor
        
        # 觸媒使用時間和設定值 -> 操作條件
        self.fc = nn.Sequential(nn.Linear(2,128),nn.ReLU(),nn.Linear(128,self.flat_size))
        
    def forward(self,d,s):
        action = self.fc(torch.cat((d,s),dim=1))
        action = action.view(-1,self.time_step,self.num_sensor)
        return F.sigmoid(action)

class PA_ROBOT(object):
    def __init__(self):
        self.mm_v = data['mm_v']
        self.mm_d = data['mm_d']
        self.mm_a = data['mm_a']
        self.a_col = data['a_col']
        self.tag_map = tag_map
        self.actor = actor
        self.critic = critic
    
    def get_advice(self,s,d):
        s = self.mm_v.transform([[s]])
        d = self.mm_d.transform([[d]])
        d = torch.FloatTensor([d]).cuda().reshape(-1,1)
        s = torch.FloatTensor([s]).cuda().reshape(-1,1)
        
        a = self.actor(d,s)
        
        v = self.critic(d,a).detach().cpu().numpy()
        v = self.mm_v.inverse_transform(v).squeeze()
        
        a = a.detach().cpu().numpy()
        a = np.array([self.mm_a.inverse_transform(i) for i in a]).squeeze(0)
        advice = pd.DataFrame(index = self.a_col)
        advice['chinese'] = advice.index.map(self.tag_map) 
        advice['mean'] = a.mean(axis=0)
        advice['max'] = a.max(axis=0)
        advice['min'] = a.min(axis=0)
        
        return advice,v

class panet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_step = time_step
        self.num_sensor = num_sensor
        self.flat_size = time_step*num_sensor
        
        self.w = nn.Sequential(nn.Linear(self.flat_size,128),# 操作條件 -> 權重
                               nn.ReLU(),
                               nn.Linear(128,time_step))
        
        self.b = nn.Sequential(nn.Linear(self.flat_size,128),# 操作條件 -> 偏移
                               nn.ReLU(),
                               nn.Linear(128,1))
        
        self.p = nn.Linear(1,1,bias=False) # 入料/出料 比例
        
    def forward(self,d,a):
        feed,factor = self.fetch(a) #把入料和其他因子分開
        W = F.softmax(self.w(a.view(-1,self.flat_size))**2,dim=1) #全部因子丟進去算權重
        b = self.b(a.view(-1,self.flat_size)) #全部因子丟進去算偏移
        WX = torch.sum(feed*W,dim=1).view(-1,1) #入料跟權重做內積
        output = self.p(WX) + b #類似線性回歸
        return F.sigmoid(output) #限縮到正常範圍內
        
    def fetch(self,a):
        batch_size = a.shape[0]
        feed = a[:,:,0]
        factor = a[:,:,1:].reshape(batch_size,-1)
        return feed,factor