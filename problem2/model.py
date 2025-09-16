import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2)
    )

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = conv_block(1, 32)   # 128->64
        self.c2 = conv_block(32, 64)  # 64->32
        self.c3 = conv_block(64, 128) # 32->16
        self.c4 = conv_block(128, 256)# 16->8
    def forward(self, x):
        x1 = self.c1(x)   # [B,32,64,64]
        x2 = self.c2(x1)  # [B,64,32,32]
        x3 = self.c3(x2)  # [B,128,16,16]
        x4 = self.c4(x3)  # [B,256,8,8]
        return x1, x2, x3, x4

class HeatmapNet(nn.Module):
    def __init__(self, num_keypoints=5):
        super().__init__()
        self.enc = Encoder()
        self.de4 = nn.Sequential(nn.ConvTranspose2d(256,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)) # 8->16
        self.de3 = nn.Sequential(nn.ConvTranspose2d(256,64,4,2,1),  nn.BatchNorm2d(64),  nn.ReLU(inplace=True)) # 16->32
        self.de2 = nn.Sequential(nn.ConvTranspose2d(128,32,4,2,1),  nn.BatchNorm2d(32),  nn.ReLU(inplace=True)) # 32->64
        self.out = nn.Conv2d(32, num_keypoints, 1)
    def forward(self, x):
        x1,x2,x3,x4 = self.enc(x)
        u4 = self.de4(x4)                    # [B,128,16,16]
        u3 = self.de3(torch.cat([u4, x3],1)) # [B,64,32,32]
        u2 = self.de2(torch.cat([u3, x2],1)) # [B,32,64,64]
        return self.out(u2)                  # [B,K,64,64]

class RegressionNet(nn.Module):
    def __init__(self, num_keypoints=5):
        super().__init__()
        self.enc = Encoder()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                       # [B,256,1,1]
            nn.Flatten(),
            nn.Linear(256,128), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(128,64),  nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(64, num_keypoints*2),
            nn.Sigmoid()                                   # 输出 [0,1]
        )
    def forward(self, x):
        _,_,_,x4 = self.enc(x)
        return self.fc(x4)                                 # [B,10]