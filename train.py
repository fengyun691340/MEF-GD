#from share import *
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
#data的数据集
import os
from data.vitonhd2 import VitonHDDataset #将原来的数据集改为自己新建的数据集
#from data.dresscode import DressCodeDataset
from data.arg_parser import eval_parse_args
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import numpy as np
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageOps
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
            phase='train',
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
            phase='train',
            order=args.test_order,
            sketch_threshold_range=(20, 20),
            radius=5,
            tokenizer=tokenizer,
            size=(512, 384),
        )
else:
        raise NotImplementedError
# Configs
resume_path = './ckpt/init_fusion.ckpt'
batch_size = 2
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./configs/cldm_v21fusion.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control
checkpoint_callback = ModelCheckpoint(  
    every_n_epochs=10,  # 每5个epochs保存一次模型  
    save_last=True,  # 总是保存最后一个epoch的模型  
    save_top_k=-1,  # 保存所有检查点，-1表示不限制数量  
    dirpath='./checkpoints_fusion/',  # 检查点保存的路径  
    filename='{epoch}-{step}-{val_loss:.2f}',  # 可以自定义文件名格式  
)  

# Misc
#dataset = MyDataset()
#Data
dataloader = DataLoader(dataset, num_workers=12, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus="0,1",accelerator="ddp", precision=32,accumulate_grad_batches=4, callbacks=[logger, checkpoint_callback],max_epochs=1)


# Train!
trainer.fit(model, dataloader)