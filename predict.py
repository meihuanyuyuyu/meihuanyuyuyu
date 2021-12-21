import torch 
from model import *
from utils import *
import json


def compare_data(model:nn.Module,dataset_path:str):
    pre_data = EM_mydataset(dataset_path,[Myrandomscale([2000,2000]),Myrandomcrop([512,512]),Mynormalize()])
    pre_sampled = DataLoader(pre_data,2,True,num_workers=1)
    pcs = []
    rcs = []
    f1s =[]
    for data in pre_sampled:
        img,label = data
        label = label.squeeze(1)
        predict = model(img.to('cuda'))
        binarymap = torch.argmax(predict,dim=1)
        pc = precision(binarymap.to('cpu'),label)
        rc = recall(binarymap.to('cpu'),label)
        f1 = F1score(binarymap.to('cpu'),label)
        pcs.append(pc)
        rcs.append(rc)
        f1s.append(f1)
        visualization(binarymap,label,f'predict.png')
    return np.array(pcs).mean(),np.array(rcs).mean(),np.array(f1s).mean()

'''
将保存模型参数加载并保存评估结果

'''
j = 2
performance =[]
au = attention_u_net()
au.load_state_dict(torch.load(f'model_parameters/au-net_model/{j}_EM_model'))
au.to(device='cuda')
au.eval()
for i in range(5):
    pc,rc,f1 = compare_data(au,'EMdata')
    performance.append({'precision':pc,'recall':rc,'F1 score':f1}) 
    
with open(f'attention_u-net_{j}_EM_different_scale_performance','w+') as f:
    json.dump(performance,f)





