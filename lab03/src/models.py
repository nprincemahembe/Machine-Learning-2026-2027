import torch
from torch import nn
class MLP(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,10))
 def forward(self,x):return self.net(x)
class CNN(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(32*7*7,10))
 def forward(self,x):return self.net(x)
class Attention(nn.Module):
 def __init__(self):
  super().__init__();self.proj=nn.Linear(49,64);self.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(64,4,128,batch_first=True),2);self.head=nn.Linear(64,10)
 def forward(self,x):
  x=x.unfold(2,7,7).unfold(3,7,7).contiguous().view(x.shape[0],16,49)
  return self.head(self.enc(self.proj(x)).mean(1))
