import torch
from PIL import Image
from typing import List
import json
import random 
from diffusers import StableDiffusionPipeline
import warnings
import argparse
import ImageReward as RM
import numpy as np 

def main():
    parser = argparse.ArgumentParser(
                    prog='ImageSelect',
                    description='Arguments for running ImageSelect',)

    parser.add_argument('--num_seeds',type=int,default=10)
    parser.add_argument('--prompt',type=str,default='cash on a stone floor')
    parser.add_argument('--sd14',action='store_true')
    args = parser.parse_args()
    reward_model = model = RM.load("ImageReward-v1.0")
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if args.sd14:
        version = 'CompVis/stable-diffusion-v1-4'
    else:
        version = 'stabilityai/stable-diffusion-2-1-base'
    stable = StableDiffusionPipeline.from_pretrained(version,torch_dtype=torch.float16).to(device)
    random.seed(42)
    seeds = [i for i in range(100)]
    img_paths = []
    all_images = []
    for count in range(args.num_seeds):
        seed = seeds[count]
        g = torch.Generator('cuda').manual_seed(seed)
        prompt = args.prompt
                
        image = stable(prompt=prompt,generator=g).images[0]
        all_images.append(image)
        path = './all_images/'+str(count)+'.png'
        image.save(path)
        img_paths.append(path)
    
    ranking,rewards = model.inference_rank(prompt,img_paths)
    best_index = np.array(ranking).argmin()
    final_savepath = './best_images/'+prompt+'.png'
    all_images[best_index].save(final_savepath)

if __name__ == '__main__':
    main()