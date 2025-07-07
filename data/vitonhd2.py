# File havily based on https://github.com/aimagelab/dress-code/blob/main/data/dataset.py
import json
import os
import pathlib
import random
import sys
from typing import Tuple

PROJECT_ROOT = pathlib.Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageOps
from torchvision.ops import masks_to_boxes
#from data.posemap import get_coco_body25_mapping
#from data.posemap import kpoint_to_heatmap

class VitonHDDataset(data.Dataset):
    def __init__(
            self,
            dataroot_path: str,
            phase: str,
            tokenizer,
            radius=5,
            caption_folder='vitonhd.json',
            sketch_threshold_range: Tuple[int, int] = (20, 127),
            order: str = 'paired',
            outputlist: Tuple[str] = ('c_name', 'im_name', 'image', 'im_cloth', 'shape','pose_map','im_densepose','mask',
                                      'parse_array', 'im_mask', 'inpaint_mask', 'parse_mask_total','im_pose','parser_mask',
                                      'im_sketch', 'captions', 'original_captions'),
            size: Tuple[int, int] = (512, 384),
    ):

        super(VitonHDDataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.caption_folder = caption_folder
        self.sketch_threshold_range = sketch_threshold_range
        self.category = ('upper_body')
        self.outputlist = outputlist
        self.height = size[0]
        self.width = size[1]
        self.radius = radius
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.order = order

        im_names = []
        c_names = []
        dataroot_names = []

        possible_outputs = ['c_name', 'im_name', 'image', 'im_cloth', 'shape','pose_map','im_densepose','mask',
                                      'parse_array', 'im_mask', 'inpaint_mask', 'parse_mask_total','im_pose','parser_mask',
                                      'im_sketch', 'captions', 'original_captions']

        assert all(x in possible_outputs for x in outputlist)

        # Load Captions
        with open(os.path.join(self.dataroot, self.caption_folder)) as f:
            # self.captions_dict = json.load(f)['items']
            self.captions_dict = json.load(f)
        self.captions_dict = {k: v for k, v in self.captions_dict.items() if len(v) >= 3}

        dataroot = self.dataroot
        if phase == 'train':
            filename = os.path.join(dataroot, f"{phase}_pairs.txt")
        else:
            filename = os.path.join(dataroot, f"{phase}_pairs.txt")

        with open(filename, 'r') as f:
            data_len = len(f.readlines())

        with open(filename, 'r') as f:
            for line in f.readlines():
                if phase == 'train':
                    im_name, _ = line.strip().split()
                    c_name = im_name
                else:
                    if order == 'paired':
                        im_name, _ = line.strip().split()
                        c_name = im_name
                    else:
                        im_name, c_name = line.strip().split()

                im_names.append(im_name)
                c_names.append(c_name)
                dataroot_names.append(dataroot)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
    def __getitem__(self, index):
        """
        For each index return the corresponding sample in the dataset
        :param index: data index
        :type index: int
        :return: dict containing dataset samples
        :rtype: dict
        """
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        dataroot = self.dataroot_names[index]

        sketch_threshold = random.randint(self.sketch_threshold_range[0], self.sketch_threshold_range[1])

        if "captions" in self.outputlist or "original_captions" in self.outputlist:
            captions = self.captions_dict[c_name.split('_')[0]]
            # take a random caption if there are multiple
            #if self.phase == 'train':
            #    random.shuffle(captions)
            #captions = ", ".join(captions)
            max_text_length=77
            captions = captions[:max_text_length]
            original_captions = captions

        if "image" in self.outputlist:
            # Person image
            # image = Image.open(os.path.join(dataroot, 'images', im_name))
            image = Image.open(os.path.join(dataroot, self.phase, 'image', im_name))
            image = image.resize((self.width, self.height))
            #image = np.array(image)
            image = self.transform(image)  # [-1,1]

        if "im_sketch" in self.outputlist:
            # Person image
            # im_sketch = Image.open(os.path.join(dataroot, 'im_sketch', c_name.replace(".jpg", ".png")))
            if self.order == 'unpaired':
                im_sketch = Image.open(
                    os.path.join(dataroot, self.phase, 'im_sketch_unpaired',
                                 os.path.splitext(im_name)[0] + '_' + c_name.replace(".jpg", ".png")))
            elif self.order == 'paired':
                im_sketch = Image.open(os.path.join(dataroot, self.phase, 'im_sketch', im_name.replace(".jpg", ".png")))
            else:
                raise ValueError(
                    f"Order should be either paired or unpaired"
                )

            im_sketch = im_sketch.resize((self.width, self.height))
            im_sketch = ImageOps.invert(im_sketch)
            # threshold grayscale pil image
            im_sketch = im_sketch.point(lambda p: 255 if p > sketch_threshold else 0)
            im_sketch = np.array(im_sketch)
            im_sketch= im_sketch[:, :, np.newaxis]
            # im_sketch = im_sketch.convert("RGB")原来就忽略了
            im_sketch = transforms.functional.to_tensor(im_sketch)  # [-1,1]
            im_sketch = 1 - im_sketch
        if "im_pose" in self.outputlist:
            pose_name = im_name.replace('.jpg', '_rendered.png')
            im_pose = Image.open(os.path.join(dataroot, self.phase, 'openpose_img', pose_name))
            im_pose = im_pose.resize((self.width, self.height))
            im_pose = self.transform(im_pose)  # [-1,1]
            #im_pose = np.array(im_pose)
        if "im_densepose" in self.outputlist:
            densepose_name=im_name
            im_densepose=Image.open(os.path.join(dataroot, self.phase, 'image-densepose', densepose_name))
            #print(im_densepose.size())
            im_densepose=im_densepose.resize((self.width, self.height))
            im_densepose = self.transform(im_densepose)  # [-1,1]
            #im_densepose= np.array(im_densepose)
            #im_densepose= im_densepose[:, :, np.newaxis]
        if "im_mask" in self.outputlist:
            immask_name = im_name
            im_mask = Image.open(os.path.join(dataroot, self.phase, 'agnostic-v3.2', immask_name))
            im_mask = im_mask.resize((self.width, self.height))
            #正则化
            im_mask = self.transform(im_mask)  # [-1,1]
            #im_mask= transforms.ToTensor()(im_mask)
            #im_mask =transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(im_mask)
        if "mask" in self.outputlist:
            mask_name = im_name.replace('.jpg', '.png')
            mask = Image.open(os.path.join(dataroot, self.phase, 'mask', mask_name))
            mask = mask.resize((self.width, self.height))
            #正则化
            mask = transforms.ToTensor()(mask)
            mask = 1-mask
            #im_mask = self.transform(im_mask)  # [-1,1]
            
        result = {}
        #result[im_densepose]=im_densepose
        #source = np.concatenate((im_mask,mask, im_densepose, im_pose,), axis=2)
        source = torch.cat((im_mask,im_sketch,im_pose,im_densepose), dim=0)
        image=image.permute(1, 2, 0)
        im_sketch=im_sketch.permute(1, 2, 0)
        source=source.permute(1, 2, 0)

        image.numpy()
        source.numpy()
        #controlnet
        #return dict(jpg=image, txt=original_captions, hint=source,cloth_name=c_name,image_name=im_name,densepose=im_densepose,pose=im_pose,immask=im_mask,sketch=im_sketch,mask=mask)
        #controlnet+adpapter
        global_conditions = []
        return dict(jpg=image, txt=original_captions, local_conditions=source,im_sketch=im_sketch,img_mask=im_mask,cloth_name=c_name,image_name=im_name,global_conditions=global_conditions)
        
        #test
        #return dict(jpg=image, txt=original_captions, cloth_name=c_name,image_name=im_name,densepose=im_densepose,pose=im_pose,immask=im_mask,parse=im_parse,sketch=im_sketch)    


    def __len__(self):
        return len(self.c_names)