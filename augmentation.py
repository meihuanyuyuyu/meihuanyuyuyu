from ctypes import sizeof
import random
from torch.functional import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import torch
from torch.nn.functional import grid_sample
from PIL import Image
from torchvision.transforms.transforms import RandomCrop



class Image_transform(object):
    def __init__(self) -> None:
        pass
    def __call__(self):
        pass

class Random_crop(Image_transform):
    '''
    随机裁剪。
    初始化输入裁剪大小.eg,[512,512]:list,
    调用类函数：传入变换的图片列表.[img,label,...]:list,图像类型可以为PIL或Tensor
    返回：裁剪完的图像列表.[img,label,...]:list,图像类型为PIL或Tensor
    
    '''
    def __init__(self,size:list) -> None:
        super().__init__()
        self.size = size
        self.width = size[1]
        self.height = size[0] 
    
    def __call__(self,arg:list):
        param = T.RandomCrop.get_params(arg[0],self.size)
        res = []
        for i in arg:
            res.append(TF.crop(i,*param))
        return res

class Random_flip(Image_transform):
    '''
    随机翻转，0.5概率水平翻转，0.5概率垂直翻转
    初始化输入裁剪大小.eg,[512,512]:list,
    调用类函数：传入变换的图片列表.[img,label,...]:list,图像类型可以为PIL或Tensor
    返回：翻转完的图像列表.[img,label,...]:list,图像类型为PIL或Tensor

    '''
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self,arg:list):
        randomseed1 = random.random()
        randomseed2 = random.random()
        res = []
        for i in arg:
            if randomseed1<0.5:
                i = TF.vflip(i)
            if randomseed1<0.5:
                i= TF.hflip(i)
            res.append(i)
        return res

class Resize(Image_transform):
    '''
    放缩
    初始化输入大小.eg,[512,512]:list,
    调用类函数：传入变换的图片列表.[img,label,...]:list,图像类型可以为PIL或Tensor
    返回：放缩完的图像列表.[img,label,...]:list,图像类型为PIL或Tensor   
    
    '''
    def __init__(self,size:list) -> None:
        super().__init__()
        self.size = size
    
    def __call__(self,arg:list):
        res = []
        for i in arg:
            res.append(TF.resize(i,self.size))
        return res

class Randomcrop_resize(Image_transform):
    '''
    随机裁剪，放缩
    初始化输入大小.eg,[512,512]:list,[1000,1000]
    调用类函数：传入变换的图片列表.[img,label,...]:list,图像类型可以为PIL或Tensor
    返回：放缩完的图像列表.[img,label,...]:list,图像类型为PIL或Tensor   
    '''
    def __init__(self,size:list,scale:list) -> None:
        super().__init__()
        self.size = size
        self.scale = scale
        
    def __call__(self,arg):
        res = []
        param = T.RandomCrop.get_params(arg[0],self.size)
        for i in arg:
            i = TF.crop(i,*param)
            i = TF.resize(i,self.scale)
            res.append(i)
        return res

class Randomrotation(Image_transform):
    '''
    随机旋转
    初始化输入翻转大小.eg,[0,359]，
    调用类函数：传入变换的图片列表.[img,label,...]:list,图像类型可以为PIL或Tensor
    返回：翻转完的图像列表.[img,label,...]:list,图像类型为PIL或Tensor   
    
    '''    
    def __init__(self,range) -> None:
        super().__init__()
        self.range = range
    
    def __call__(self,arg:list):
        res = []
        angel = T.RandomRotation.get_params(self.range)
        for i in arg:
            res.append(TF.rotate(i,angel))
        return res

class Elastic_deformation(Image_transform):
    def __init__(self,sigma:int=10,grid_size:int =100,label_index:int=1,theta:float=0.5) -> None:
        super().__init__()
        self.sigma = sigma
        self.grid_size = grid_size
        self.label_index = label_index
        self.theta = theta

    def __call__(self,arg:list)->list:
        if isinstance(arg[0],Image.Image):
            size = arg[0].size
        else:
            size = arg[0].size()
        shift = torch.randn((2,int((size[-2]-1)/self.grid_size)+1,int((size[-1]-1)/self.grid_size)+1))*self.sigma
        shift = TF.resize(shift,(size[-2],size[-1])).permute(1,2,0)
        x = torch.linspace(-1,1,size[-1])
        y = torch.linspace(-1,1,size[-2])
        y,x = torch.meshgrid(y,x)
        y,x = y.unsqueeze(2),x.unsqueeze(2)
        xy = torch.cat((x,y),dim=2)
        shift[...,0] = shift[...,0]/size[-1]
        shift[...,1] = shift[...,0]/size[-2]
        xy = xy +shift
        res =[]
        for index,i in enumerate(arg):
            if index ==self.label_index:
                i = TF.to_tensor(i)
                i =grid_sample(i.unsqueeze(0),xy.unsqueeze(0),'bicubic')
                i = (i>self.theta)*1.0
                res.append(i.squeeze(0))
            else:
                i = TF.to_tensor(i)
                i =grid_sample(i.unsqueeze(0),xy.unsqueeze(0),'bicubic')
                res.append(i.squeeze(0))
        return res
'''
class _Elastic_deformation(Image_transform):
    def __init__(self,sigma,grid_size,alpha,label_index:int=1) -> None:
        super().__init__()
        self.sigma = sigma
        self.grid_size = grid_size
        self.alpha = alpha
        self.label_index = label_index
    
    def __call__(self,arg):
        if isinstance(arg[0],Image.Image):
            size = arg[0].size
        else:
            size = arg[0].size()
        shift = torch.randn((2,int((size[-2]-1)/self.grid_size)+1,int((size[-1]-1)/self.grid_size)+1))*self.sigma
        shift = TF.resize(shift,(size[-2],size[-1])).permute(1,2,0)
        x = torch.linspace(-1,1,size[-1])
        y = torch.linspace(-1,1,size[-2])
        y,x = torch.meshgrid(y,x)
        y,x = y.unsqueeze(2),x.unsqueeze(2)
        xy = torch.cat((x,y),dim=2)
        shift[...,0] = shift[...,0]/size[-1]
        shift[...,1] = shift[...,0]/size[-2]
        xy = xy +shift
        res =[]
        for index,i in enumerate(arg):
            if index ==self.label_index:
                i = TF.to_tensor(i)
                i =grid_sample(i.unsqueeze(0),xy.unsqueeze(0),'bicubic')
                i = i.squeeze(0)
                i = filters.gaussian_filter(i,self.alpha)
                i = (torch.from_numpy(i)>0.5)*1.0
                res.append(i)
            else:
                i = TF.to_tensor(i)
                i =grid_sample(i.unsqueeze(0),xy.unsqueeze(0),'bicubic')
                i = i.squeeze(0)
                i = filters.gaussian_filter(i,self.alpha)
                res.append(torch.from_numpy(i))
        return res'''
class Normalize_(Image_transform):
    def __init__(self,mean:list,divation:list) -> None:
        super().__init__()
        self.mean = mean
        self.divation = divation
    def __call__(self,arg:list):
        assert isinstance(arg[0],Tensor)
        arg[0] = TF.normalize(arg[0],self.mean,self.divation)
        return arg














label = Image.open('TCGA-18-5592-01Z-00-DX1.tif')
img = Image.open('TCGA-18-5592-01Z-00-DX1 copy.tif')
arg = [img,label]
e = Elastic_deformation(20,50,1)
rc = Random_crop([512,512])
#e = _Elastic_deformation(10,50,1,1)
# res = e(arg)
res =rc(arg)
for i,pic in enumerate(res):
    pic.save(f'{i}e.png')
