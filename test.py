#from share import *
import time
from models.model import create_model, load_state_dict

import sys
if './' not in sys.path:
	sys.path.append('./')
	
from omegaconf import OmegaConf
import argparse

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from ldm.util import instantiate_from_config
from models.util import load_state_dict
from models.logger import ImageLogger
from PIL import Image
#data的数据集
import os
from data.vitonhdblur import VitonHDDataset
from data.arg_parser import eval_parse_args
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import numpy as np
from torchvision.utils import save_image
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
#指定GPU 

#data 的读取
#data
args = eval_parse_args()
if args.category:
        category = [args.category]
else:
        category = ['dresses', 'upper_body', 'lower_body']
tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

if args.dataset == "dresscode":
        dataset = DressCodeDataset(
            dataroot_path=args.dataset_path,
            phase='test',
            order=args.test_order,
            radius=5,
            sketch_threshold_range=(20, 20),
            tokenizer=tokenizer,
            category=category,
            size=(512, 384)
        )
elif args.dataset == "vitonhd":
        dataset = VitonHDDataset(
            dataroot_path=args.dataset_path,
            phase='test',
            order=args.test_order,
            sketch_threshold_range=(20, 20),
            radius=5,
            tokenizer=tokenizer,
            size=(512, 384),
        )
else:
        raise NotImplementedError
def tesnsor2img(tensor):
    tensor = tensor.squeeze(0)
    grid = (tensor + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.numpy()
    grid = (grid * 255).astype(np.uint8)
    return grid

def save_local(save_dir, batch, images,image_number):
    
    root = os.path.join(save_dir, "pair2")                        #记得设置Unpaired 还是 paired
    filename_result = "{}-{}.png".format(
       batch['image_name'], batch['cloth_name'])
    #path = os.path.join(root, filename_grid)
    #os.makedirs(os.path.split(path)[0], exist_ok=True)
    #Image.fromarray(grid).save(path)
    path = os.path.join(root, filename_result)
    results = tesnsor2img(images["samples"])
    Image.fromarray(results).save(path)
#测试图片
def test_img(pl_module, batch,batch_size):
    if True :
        
        pl_module.eval()

        with torch.no_grad():
            torch.cuda.empty_cache()
            start_time = time.time()
            #images = pl_module.test_images(batch,N=batch_size,n_row=batch_size,ddim_steps=200)
            images = pl_module.test_images(batch,N=batch_size,n_row=batch_size,ddim_steps=200)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"程序运行时间: {execution_time} 秒")
            

        for k in images:
            N = images[k].shape[0]
            images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                
                images[k] = torch.clamp(images[k], -1.0, 1.0)

        for i in range(batch_size):
            save_local(
                save_dir,
                batch,
                images,
                i)
#首先是将数据转移到cuda
def Tocuda(batch):
    for key,value in batch.items():
        if key != 'cloth_name' and key != 'image_name' and key!='txt':
            print(key)
            batch[key] = value.to(device)
    
    return batch
# Configs
resume_path = 'checkpoint.ckpt' #权重地址
batch_size = 1
sd_locked = True
only_mid_control = False
save_dir = './testresult'
if __name__ == "__main__":
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    #model = create_model('./configs/cldm_duoinputv21.yaml').cuda()
    model = create_model('./configs/cldm_v21fusion.yaml').cuda()
    model.load_state_dict(load_state_dict(resume_path, location="cuda"))
    model.only_mid_control = only_mid_control

    # Misc
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=False)

    for i,batch in enumerate(dataloader):
        print(batch['txt'])
        #batch = Tocuda(batch)
        test_img(model,batch,batch_size)
        print(i,end='\n')