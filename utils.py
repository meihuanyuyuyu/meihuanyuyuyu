from typing import Sequence, Tuple
import torch
import torch.optim as optim
import torch.nn as nn
from torch.functional import Tensor, tensordot
from torch.utils.data import Dataset,DataLoader
from PIL import Image,ImageDraw
import os
from torchvision.transforms.transforms import ColorJitter
from torchvision.utils import save_image
import random
import torchvision.transforms.functional as F
import torchvision.io as io
import torch.nn.functional as nnf
import numpy as np
import xml.dom.minidom
import metric
import json
from scipy.ndimage import label,distance_transform_edt


class Mydataset(Dataset):
    def __init__(self,root,mode='train',transforms:list =None) -> None:
        super().__init__()
        if mode =='train':
            self.path = os.path.join(root,'2018 Training Data')
            self.images_path = os.path.join(self.path,'Tissue Images')
            self.labels_path = os.path.join(self.path,'Annotations')
        if mode == 'test':
            self.path = os.path.join(root,'MoNuSegTestData')
            self.images_path = os.path.join(self.path,'imgs')
            self.labels_path = os.path.join(self.path,'labels')
        self.images = os.listdir(self.images_path)
        self.labels = os.listdir(self.labels_path)
        self.transforms = transforms
    
    def __len__(self):
        return len(self.images)

    def transform(self,img,label):
        for i in self.transforms:
            img,label = i(img,label)
        return img,label    
   
    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        img = Image.open(os.path.join(self.images_path,img))
        img = F.to_tensor(img)
        label = Image.open(os.path.join(self.labels_path,label))
        label = F.to_tensor(label)
        return self.transform(img,label)
    
class Elastictransform():
    def __init__(self,sigma=5,grid=150) -> None:
        self.grid_size = grid
        self.sigma = sigma
    
    def __call__(self,img:Tensor,label:Tensor) -> Tensor:
        size = img.size()
        shift = torch.randn(2,int((size[1] + self.grid_size - 2) / (self.grid_size - 1)),int((size[2] + self.grid_size - 2) / (self.grid_size - 1))) * self.sigma 
        shift=F.resize(shift,(size[1],size[2])).permute(1,2,0)
        x = torch.linspace(-1,1,size[1])
        y = torch.linspace(-1,1,size[2])
        y,x =torch.meshgrid(x,y)
        y =y.unsqueeze(2)
        x =x.unsqueeze(2)
        xy = torch.cat((x,y),dim=2)
        shift[:,:,0]=shift[:,:,0]*2/size[1]
        shift[:,:,1]=shift[:,:,1]*2/size[2]
        xy = xy +shift
        target = nnf.grid_sample(img.unsqueeze(0),xy.unsqueeze(0),mode='bicubic')
        label = nnf.grid_sample(label.unsqueeze(0),xy.unsqueeze(0),mode='bicubic')

        label =(label>0.5)*1.0
        return target.squeeze(0),label.squeeze(0)

def polygon2mask_default(w:int,h:int,polygons:list) -> Image.Image:
    '''

    '''
    binary_mask=Image.new('L', (w, h), 0)
    for polygon in polygons:
        ImageDraw.Draw(binary_mask).polygon(polygon, outline=255, fill=255)  
    return binary_mask 

def xml_to_binary_mask(w,h,filename: str,polygon2mask=polygon2mask_default) -> Image.Image:
    xml_file   = filename
    xDoc       = xml.dom.minidom.parse(xml_file).documentElement
    Regions    = xDoc.getElementsByTagName('Region')
    xy         = []
    for i, Region in enumerate(Regions):
        verticies = Region.getElementsByTagName('Vertex')
        xy.append(np.zeros((len(verticies), 2)))
        for j, vertex in enumerate(verticies):
            xy[i][j][0], xy[i][j][1] = float(vertex.getAttribute('X')), float(vertex.getAttribute('Y'))
    polygons=[]
    for zz in xy:
        polygon = []
        for k in range(len(zz)):
            polygon.append((zz[k][0], zz[k][1]))
        polygons.append(polygon)
    return polygon2mask(w,h,polygons)

class Myrandomflip(object):
    def __call__(self, img:Tensor,label:Tensor) -> Tensor:
        if random.random()>0.5:
            img,label = F.F_t.hflip(img),F.F_t.hflip(label)
        if random.random()>0.5:
            img,label =F.F_t.vflip(img),F.F_t.vflip(label)
        return img,label

class Myrandomcrop(object):
    def __init__(self,size:tuple) -> None:
        super().__init__()
        self.size = size
        self.height = size[0]
        self.width = size[1]

    def __call__(self, img:Tensor,label:Tensor):
        raw_size =img.size()
        boundry =raw_size[-1] -self.size[1]
        x = random.randint(0,boundry)
        y = random.randint(0,boundry)
        img = F.F_t.crop(img,x,y,self.height,self.width)
        label = F.F_t.crop(label,x,y,self.height,self.width)
        return img,label

class Mynormalize(object):
    def __call__(self,*args) -> Tensor:
        if len(args)==1:
            img =args[0][0]
            label =args[0][1]
        else:
            img =args[0]
            label = args[1]
        return F.normalize(img,[0.5],[0.5]),(label>0.5)*1.0

class Myrandomrotation(object):
    def __call__(self,img:Tensor,label:Tensor):
        angel = random.uniform(0.1,4.1)
        return F.rotate(img,angel),F.rotate(label,angel)

class Myrandomscale(object):
    def __init__(self,size:list) -> None:
        super().__init__()
        self.size = size
    def __call__(self, img:Tensor,label:Tensor) :
        return F.resize(img,self.size),F.resize(label,self.size)

def visualization_black_as_target(predict:Tensor,label:Tensor,name)->None:
    size = predict.size()
    result = torch.zeros(size[0],3,size[1],size[2])
    result[:,0,:,:]= 1 -label
    result[:,1,:,:]=1- predict
    save_image(result,os.path.join('vision_result',name))

def visualization_white_as_target(predict:Tensor,label:Tensor,input:Tensor,name):
    size = predict.size()
    result = torch.zeros(size[0],3,size[1],size[2])
    result[:,0,:,:]= label
    result[:,1,:,:]= predict
    save_image(input,os.path.join('my-unet/vision_result','img_'+name))
    save_image(result,os.path.join('my-unet/vision_result',name))

def train(model:nn.Module,sampled_data:DataLoader,criterion:nn.Module,optimizer:optim.Optimizer,device:str='cuda'):
    losses =[]
    f1s =[]
    model.train()
    for data in sampled_data:
        imgs,labels = data
        labels =labels.squeeze(1)
        outputs = model(imgs.to(device))
        optimizer.zero_grad()
        # white = labels.sum()/(labels.size(0)*labels.size(1)*labels.size(2))
        #white = white.item()
        #black = 1-white
        #torch.Tensor([white,black]).to(device)
        # loss = criterion(outputs,labels.to(device=device,dtype=torch.long),torch.Tensor([white,black]).to(device))
        loss = criterion(outputs,labels.to(device,dtype=torch.long))
        binarymap = torch.argmax(outputs,dim=1)
        loss.backward()
        optimizer.step()
        f1 = metric.F1score_white_as_target(binarymap,labels.to(device))
        f1s.append(f1)
        losses.append(loss.item())
        return np.array(losses).mean(),np.array(f1s).mean()

def val(i,model:nn.Module,sampled_data:DataLoader,device:str='cuda'):
    f1s = []
    model.eval()
    for data in sampled_data:
        imgs,labels = data
        labels =labels.squeeze(1)
        outputs =model(imgs.to(device))
        binarymap = torch.argmax(outputs,dim=1)
        f1score = metric.F1score_white_as_target(binarymap,labels.to(device))
        f1s.append(f1score)
        visualization_white_as_target(binarymap,labels,imgs,f'{i}val.png')
    return np.array(f1s).mean()

class EM_mydataset(Dataset):
    def __init__(self,root:str,transforms:list) -> None:
        super().__init__()
        self.img_path = os.path.join(root,'train/image')
        self.label_path =os.path.join(root,'train/label')
        self.labels =os.listdir(self.label_path)
        self.imgs =os.listdir(self.img_path)
        self.transforms = transforms

    def __len__(self):
        return len(os.listdir('./EMdata/train/image'))

    def __getitem__(self, index):
        label = self.labels[index]
        img = self.imgs[index]
        label = Image.open(os.path.join(self.label_path,label))
        img = Image.open(os.path.join(self.img_path,img))
        img,label =F.to_tensor(img),F.to_tensor(label)
        return self.transform(img,label)

    def transform(self,img:Tensor,label:Tensor):
        if len(self.transforms)!=0:
            for i in self.transforms:
                img,label = i(img,label)
        return img,label
            
class split_set(Dataset):
    def __init__(self,data:Dataset,indices:Sequence[int],transforms:list=[]) -> None:
        super().__init__()
        self.data = data
        self.indices = indices
        self.transforms = transforms
    
    def __getitem__(self, index):
        img,label = self.data[self.indices[index]]
        return self.transform(img,label)
    
    def transform(self,img:Tensor,label:Tensor):
        for i in self.transforms:
            img,label = i(img,label)
        return img,label
    
    def __len__(self):
        return len(self.indices)

def kfold_crossval(data:Dataset,k:int=4,num_fold:int=0,train_t=[],val_t=[]):
    '''
    返回两个划分好的训练和验证集，数据类型为Dataset类
    '''
    assert (num_fold<k)
    length =len(data)
    k_size = int(length/k)
    test_arr=[num_fold*k_size+i for i in range(k_size)]
    return split_set(data,[x for x in range(length) if x not in test_arr],train_t),split_set(data,test_arr,val_t)

def save_result_json(fp:str,i,loss,val_f1,f1s):
    result = {'i': i, 'loss':loss,'f1':f1s, 'val_f1':val_f1}
    with open(os.path.join(fp,f'_{i}_train_score.json'),'w') as f:
        json.dump(result,f)

class Mousegment_2018_dataset(Dataset):
    def __init__(self,root:str,mode:str='train',transform:list=[]) -> None:
        super().__init__()
        if mode == 'train':
            self.imgs_path = os.path.join(root,'2018 Training Data/Tissue Images')
            self.labels_path = os.path.join(root,'2018 Training Data/Annotations')
            self.imgs = os.listdir(self.imgs_path)
            self.labels = os.listdir(self.labels_path)
        if mode == 'val':
            self.imgs_path = os.path.join(root,'MoNuSegTestData/imgs')
            self.labels_path = os.path.join(root,'MoNuSegTestData/labels')
            self.imgs = os.listdir(self.imgs_path)
            self.labels = os.listdir(self.labels_path)
        self.transforms =transform
    
    def __len__(self):
        return len(self.imgs)

    def transform(self,img,label):
        if len(self.transforms)==0:
            return img,label
        else:
            for t in self.transforms:
                img,label = t(img,label)
            return img,label
    
    def __getitem__(self, index):
        img = os.path.join(self.imgs_path,self.imgs[index])
        label = os.path.join(self.labels_path,self.labels[index])
        img=Image.open(img)
        label = Image.open(label)
        img,label = F.to_tensor(img),F.to_tensor(label)
        return self.transform(img,label)

class weighted_map(object):
    '''
    权重图
    '''
    def __init__(self,sigma=5,w_0=10) -> None:
        super().__init__()
        self.sigma = sigma
        self.w_0 = w_0

    def __call__(self,label_path:str)->Tensor:
        l = np.array(Image.open(label_path),dtype=np.int64)
        print(l.shape)
        l = torch.tensor(l,dtype=torch.int64)
        l = l.clip(0,1)
        cells,num_cells = label(l)
        cells = torch.tensor(cells,dtype=torch.int64)
        cells = nnf.one_hot(cells,num_classes=num_cells+1)
        dists = torch.zeros_like(cells[...,1:])
        for i in range(1,num_cells+1):
            dist = cells[...,i].numpy()
            dists[...,i-1] = torch.tensor(distance_transform_edt(1-dist))
        dists,_ = dists.sort(dim=-1)
        d1 = dists[...,0]
        d2 = dists[...,1]
        w_c = l.sum().item()/(l.size(-2)*l.size(-1))
        return w_c+ self.w_0*torch.exp(-(d1+d2)**2/(2*self.sigma**2))*(l == 0)

'''a = os.listdir('EMdata/train/label')
for j,i in enumerate(a):
    file_path = os.path.join('EMdata/train/label',i)
    wm = weighted_map()
    weight_map = wm(file_path)
    torch.save(weight_map,f'EMdata/train/weight map/{j}.pt')'''

class My_center_crop(object):
    def __init__(self,size:list) -> None:
        super().__init__()
        self.size = size

    def __call__(self,*args):
        imgs =[]
        for i in args:
            imgs.append(F.center_crop(i,self.size))
        return imgs[0] if len(args)==1 else imgs

class My_colorjitter(ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    def forward(self,img,label):
        return super().forward(img),label

class Dice_loss():
    def __init__(self,sigma:torch.long =1e-8,reduction:str='mean') -> None:
        self.reduction =reduction
        self.sigma = sigma

    def _dice_loss(self,softmax:Tensor,target:Tensor):
        target = nnf.one_hot(target,num_classes=2)
        target = target.permute(0,3,1,2)
        if self.reduction=='mean': # batch_size总的dice系数
            dice = 2*(softmax*target).sum()/(softmax[:,1].square().sum()+target.square().sum())+self.sigma
            dice = dice
            return 1-dice

    def __call__(self,predict:Tensor,target:Tensor):
        numerator = predict.exp()
        denominator = numerator.sum(dim=1,keepdim=True)
        softmax = numerator/denominator
        return self._dice_loss(softmax,target)

class Dice_loss_with_logist():
    def __init__(self) -> None:
        pass

    def _dice_loss_with_logist(self,predict:Tensor,target:Tensor):
        ce= nnf.cross_entropy(predict,target)
        target = nnf.one_hot(target,num_classes=2)
        target = target.permute(0,3,1,2)
        target = target[:,1]
        softmax = nnf.softmax(predict,dim=1)
        dice = 2*(softmax[:,1]*target).sum()/(softmax[:,1].sum()+target.sum())
        dice =1 -dice
        return ce/2+dice

    def __call__(self,predcit:Tensor,target:Tensor):
        return self._dice_loss_with_logist(predcit,target)
        



