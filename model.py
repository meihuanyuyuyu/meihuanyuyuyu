from typing import ForwardRef
from torch.functional import Tensor
import torch.nn as nn
from torch.nn import init
from torch.nn.functional import dropout, sigmoid, upsample
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d, ConvTranspose2d
import torch
from torch.nn import init
from torch.nn import functional as F

from metric import precision

class Down(nn.Module):
    r'下采样模块，输入(n,c,h,w)的特征图，依次输出下采样得到的特征图和用来跳跃连接特征图，in_channel:输入下采样模块的通道数，out_channel:输出模块的通道数'
    def __init__(self,
                in_channel:int,
                out_channel:int) -> None:
        super().__init__()
        ################### 初始化一些卷积参数 ###################################
        self.in_channel =in_channel
        self.out_channel = out_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel,out_channels=self.out_channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(self.out_channel,self.out_channel,3,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )
        self.maxpool = nn.MaxPool2d(2)

    def forward(self,x):
        ################### 前向传播 ############################################
        x1=x= self.conv1(x) 
        return self.maxpool(x),x1 # 返回下采样后的张量x和跳跃连接张量x1

class UpSample(nn.Module):
    r'上采样模块，输入两个变量，一个是顺基础块正向传播的特征图和跳跃连接过来的特征图，依次输出下采样得到的特征图和用来跳跃连接特征图，in_c:输入下采样模块的通道数，out_c:输出模块的通道数'
    def __init__(self,
                in_c,
                out_c):
        super().__init__()
        ################### 初始化一些卷积参数 ###################################
        self.in_c = in_c
        self.out_c = out_c
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.in_c,self.out_c,3,1,1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
            nn.Conv2d(self.out_c,self.out_c,3,1,1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )
        self.up = nn.ConvTranspose2d(self.out_c,int(self.out_c/2),2,2)
    def forward(self,x,x1):
        x = torch.cat([x,x1],dim=1) #通道数进行拼接
        x = self.conv2(x)
        return self.up(x)

class mid_bridge(nn.Module):
    def __init__(self,
                in_c,
                out_c) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(self.in_c,self.out_c,3,1,1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(self.out_c,self.out_c,3,1,1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        self.up = nn.ConvTranspose2d(self.out_c,int(self.out_c/2),2,2)
    
    def forward(self,x):
        x = self.conv(x)
        return self.up(x)

class unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = Down(3,64)
        self.down2 = Down(64,128)
        self.down3 = Down(128,256)
        self.down4 = Down(256,512)
        self.mid = mid_bridge(512,1024)
        self.up1 = UpSample(1024,512)
        self.up2 = UpSample(512,256)
        self.up3 = UpSample(256,128)
        self.final = nn.Sequential(
            nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64,2,1)
        )
    def forward(self,x:Tensor):
        x,x1 = self.down1(x)
        x,x2 = self.down2(x)
        x,x3 =self.down3(x)
        x,x4 = self.down4(x)
        x = self.mid(x)
        x = self.up1(x,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = torch.cat([x,x1],dim=1)
        return self.final(x)

def init_weight(m):
    if  isinstance(m,nn.Conv2d):
        init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='relu')
    if isinstance(m,nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='relu')

class attention_block(nn.Module):
    r'注意力门控，输入跳跃连结通道数，上采样模块通道数，对这两个特征图进行相加，卷积后sigmod输出一个乘以注意力系数（0-1）大小的特征图。'
    def __init__(self,f_in,f_g,f_media):
        super().__init__()
        ################### 初始化一些卷积参数 ###################################
        self.input_channel = f_in # 跳跃连结输入的通道数
        self.gating_channel = f_g # 上一层解码层输出的通道数
        self.inter_channel = f_media # 相加后的通道数
        self.w = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=self.inter_channel,out_channels=1,kernel_size=1,stride=1),
            nn.Sigmoid()
        )# 将相加的特征图输出为1通道的注意力系数
        self.W_g = nn.ConvTranspose2d(self.gating_channel,self.inter_channel,2,2)
        self.W_x = nn.Conv2d(self.input_channel,self.inter_channel,1,1)

    def forward(self,x,g):
        ################### 注意力跳跃连结前向传播 ###################################
        input_size = x.size()
        batch_size = input_size[0]  
        assert batch_size == g.size(0)
        w_x = self.W_x(x)
        w_g = self.W_g(g)
        w_media = w_x +w_g
        a = self.w(w_media)
        return a*x

class attention_up(UpSample):
    def forward(self,x,x1):
        x = torch.cat([x,x1],dim=1)
        g = self.conv2(x)
        return g,self.up(g)

class attention_u_net(unet):
    def __init__(self):
        super(attention_u_net,self).__init__()
        self.mid = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(256,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
            )
        self.ConvT =nn.ConvTranspose2d(512,256,2,2)
        self.ag3 = attention_block(256,512,256)
        self.ag2 = attention_block(128,256,128)
        self.aup1 = attention_up(512,256)
        self.aup2 =attention_up(256,128)

    def forward(self,x):
        x,x1 = self.down1(x)
        x,x2 = self.down2(x)
        x,x3 = self.down3(x)
        g3=x = self.mid(x)
        x = self.ConvT(x)
        ag3 =self.ag3(x3,g3)
        g2,x = self.aup1(x,ag3)
        ag2 =self.ag2(x2,g2)
        # 最后一层做拼接：
        x = self.up3(x,ag2)
        x = torch.cat([x,x1],dim=1)
        return self.final(x)
        
class residual_down_block(Down):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__(in_channel, out_channel)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        self.residual_conv =nn.Sequential(
            nn.Conv2d(out_channel,out_channel,3,1,1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel,out_channel,3,1,1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    
    def forward(self,x):
        x1=x = self.conv(x)
        x = self.residual_conv(x)
        x = x1+x
        return self.maxpool(x),x

class residual_up_block(UpSample):
    def __init__(self, in_c, out_c):
        super().__init__(in_c, out_c)
        self.up_conv = nn.Sequential(
            nn.Conv2d(out_c,out_c,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c,out_c,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
    
    def forward(self,x,skip_connect):
        x = torch.cat((x,skip_connect),dim=1)
        x = x1 = self.conv(x)
        x = self.up_conv(x)
        x =x1+x
        return self.up(x)

class residual_mid_bridge(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(in_c,out_c,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        self.mid_conv=nn.Sequential(
            nn.Conv2d(out_c,out_c,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c,out_c,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        ) 
        self.up = nn.ConvTranspose2d(out_c,int(out_c/2),2,2)

    def forward(self,x):
        x= x1  =self.conv(x)
        x = self.mid_conv(x)
        x = x+x1
        return self.up(x) 

class unet_basic(nn.Module):
    def __init__(self,block:nn.Module,layer_num,*args) -> None:
        super().__init__()
        self.block = block
        self.layer_num =layer_num
        self.contract = self.contract_blocks(*args)
        self.expand = self.expand_blocks(*args)
        self.mid = self.mid_blocks(*args)
    
    def contract_blocks(self,*args):
        blocks = []
        for i in range(self.layer_num):
            block = self.block(args[i],args[i+1])
            blocks.append(block)
        return nn.Sequential(
            *blocks
        )
    
    def mid_blocks(self,*args):
        block = self.block(args[self.layer_num],args[self.layer_num+1])
        return block
    
    def expand_blocks(self,*args):
        blocks = []
        for i in range(self.layer_num+1,2*self.layer_num+1):
            blocks.append(self.block(args[self.layer_num+1],args[2*self.layer_num]))
        return nn.Sequential(*blocks)
    
    def forward(self,x):
        pass
        
class Mysequential(nn.Module):
    def __init__(self,*arg):
        super().__init__()
        self.length = len(arg)
        for index,item in enumerate(arg):
            self.add_module(str(index),item)
        
    def forward(self,*arg):
        for index in range(self.length):
            arg = self._modules[str(index)](*arg)
        return arg[0] if len(arg)==1 else tuple(arg)
            

class residual_unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = residual_down_block(3,64)
        self.down2  =residual_down_block(64,128)
        self.down3  =residual_down_block(128,256)
        self.down4  =residual_down_block(256,512)
        self.mid  = residual_mid_bridge(512,1024)
        self.up1 = residual_up_block(1024,512)
        self.up2 = residual_up_block(512,256)
        self.up3 = residual_up_block(256,128)
        self.final = nn.Sequential(
            nn.Conv2d(128,64,3,1,1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,2,1,1)
        )

    def forward(self,x):
        x,x1 = self.down1(x)
        x,x2 = self.down2(x)
        x,x3 = self.down3(x)
        x,x4  = self.down4(x)
        x = self.mid(x)
        x  = self.up1(x,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = torch.cat([x,x1],dim=1)
        return  self.final(x)


class My_down_block(residual_down_block):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__(in_channel, out_channel)
    
    def forward(self,x):
        x1=x = self.conv(x)
        x = self.residual_conv(x)
        x = x1+x
        return self.maxpool(x),x

class My_mid_block(residual_mid_bridge):
    def __init__(self, in_c, out_c):
        super().__init__(in_c, out_c)
    
    def forward(self, x):
        x = x1 = self.conv(x)
        x = self.mid_conv(x)
        x = x+x1
        return self.up(x),x

