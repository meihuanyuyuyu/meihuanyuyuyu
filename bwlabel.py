from numpy import absolute, number
import torch
from PIL import Image
import os
from torch.functional import Tensor
from torchvision.transforms import transforms
import torch.nn.functional as F
def load_gt(path:str)->Tensor:
    if not os.path.exists(path):
        os.mkdir(path)
    convert = transforms.PILToTensor()
    gts = os.listdir(path)
    for i,filename in enumerate(gts):
        bw_gts = []
        filepath = os.path.join(path,filename)
        bw_gts.append(Image.open(filepath))
        bw_gts[i]=convert(bw_gts[i])
        bw_gts[i]=(bw_gts[i]==255)*1
    return bw_gts


def bwlabel(bw:Tensor):
    n = 1
    bw =bw.squeeze(0)
    height = bw.size(-2)
    width = bw.size(-1)
    flag = torch.zeros_like(bw)
    stack = []
    for i in range(height):
        for j in range(width):
            if (flag[i][j]==0).item() and (bw[i][j]==1).item():
                stack.append((i,j))
                while len(stack):
                    r,c = stack.pop()
                    bw[r][c] = n
                    flag[r][c] = 1
                    if r+1<height and (flag[r+1][c]==0).item() and (bw[r+1][c]==1).item():
                        stack.append((r+1,c))
                    if r-1>=0 and (flag[r-1][c]==0).item() and (bw[r-1][c]==1).item():
                        stack.append((r-1,c))
                    if c+1<width and (flag[r][c+1]==0).item() and (bw[r][c+1]==1).item():
                        stack.append((r,c+1))
                    if c-1>=0 and (flag[r][c-1]==0).item() and (bw[r][c-1]==1).item():
                        stack.append((r,c-1))
                n = n+1
    bw = bw.unsqueeze(0)
    return bw,n


bw_gts = load_gt('./data')
bw = bw_gts[0]
labeled_bw,num = bwlabel(bw)
print(num)
a = F.one_hot(bw,num_classes=num).permute((0,3,1,2))

