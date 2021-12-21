import torch
from torch.optim.optimizer import Optimizer
from utils import *
from tqdm import tqdm
from torch.utils.data import DataLoader, dataloader, dataset
from model import attention_u_net, init_weight, unet,residual_unet
import warnings
import torch.nn.functional as F
from config import *
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')

def split_train_test(dataset:Dataset,train_t,val_t,i:int=0,kfold__mode:bool=False):
    if kfold__mode:
        data = dataset(root_path,[])
        t,v= kfold_crossval(data,k,i,train_t,val_t)
        return DataLoader(t,batch_size=batch_size,shuffle=True,num_workers=1),DataLoader(v,batch_size=batch_size,shuffle=True,num_workers=1)
    else:
        train_set = dataset(root_path,'train',train_t)
        val_set = dataset(root_path,'val',val_t)
        return DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=1),DataLoader(val_set,batch_size=batch_size,shuffle=True,num_workers=1)

writer = SummaryWriter()

def main(i,
        train_t:list,
        val_t:list,
        model:nn.Module,
        e:int,
        dataset:Dataset,
        result_root:str):
    train_sampled,val_sampled = split_train_test(dataset,train_t,val_t)
    net =model().to(device)
    net.apply(init_weight)
    opt = torch.optim.SGD(net.parameters(),lr=9e-3,momentum=0.9)
    criterion = Dice_loss_with_logist()
    if net.__class__.__name__== 'attention_u_net':
        model_dir = os.path.join(result_root,f'model_parameters/attention unet/{i}_EM_model.pt')
        json_dir = os.path.join(result_root,f'training_json/attention unet')
    if net.__class__.__name__ =='unet':
        model_dir = os.path.join(result_root,f'model_parameters/unet/{i}_EM_model.pt')
        json_dir = os.path.join(result_root,f'training_json/unet')
    if net.__class__.__name__ =='residual_unet':
        model_dir = os.path.join(result_root,f'model_parameters/residual unet/{i}_EM_model.pt')
        json_dir = os.path.join(result_root,f'training_json/residual unet')
    val_f1s = []
    losses = []
    f1s = []
    bar = tqdm(range(e))
    for epoch in bar:
        loss,f1 = train(net,train_sampled,criterion,opt,device=device)
        writer.add_scalar('Loss/train', loss, epoch)
        bar.set_description(f'training loss:{loss},f1 score:{f1}')
        val_f1 = val(i,net,val_sampled,device=device)
        bar.set_description(f'val_f1 {val_f1}')
        losses.append(loss)
        val_f1s.append(val_f1)
        f1s.append(f1)
        writer.add_scalars('variation of two metric', {'f1':f1,'val_f1':val_f1}, epoch)
        if epoch % 100 ==0:
            torch.save(net.state_dict(),model_dir)
    writer.close()
    save_result_json(json_dir,i,losses,val_f1s,f1s)
    
    


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(9,[Myrandomcrop([512,512]),My_colorjitter(0.7,0.7,0.7),Elastictransform(10,100),Myrandomrotation(),Myrandomflip(),Mynormalize()],
    [My_center_crop([512,512]),Mynormalize()],attention_u_net,1000,Mousegment_2018_dataset,'my-unet/result/2020seg_result')
