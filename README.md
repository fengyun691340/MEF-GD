# MEF-GD
MEF-GD: Multimodal Enhancement and Fusion Network for Garment Designer


https://github.com/user-attachments/assets/357f241f-5bbb-4239-a067-e1a53bf7ade0



https://github.com/user-attachments/assets/60b15e92-b1d1-477d-86ad-6c631592b712

## First create a new conda environment

    conda env create -f environment.yaml
    conda activate MEFGD

## Test
    sh test.sh 
## Prepare weights
You should first download the pretrained weights of ["v2-1_512-ema-pruned.ckpt"](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/tree/main) and put it to `./ckpt/` folder. Then, you can get the initial weights for training by:

    python utils/prepare_weights.py  ckpt/v2-1_512-ema-pruned.ckpt configs/cldm_v21fusion.yaml ckpt/init_fusion.ckpt
## Train
    sh train.sh
