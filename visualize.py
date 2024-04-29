
import sys
import os
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR100, MNIST
from torchvision import transforms
from tqdm import trange
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from pytorch_pretrained_vit import ViT
import json
import PIL
from tqdm import tqdm, trange
from collections import Counter
from transformers import BertTokenizer
import ruamel_yaml as yaml
from scipy.optimize import linear_sum_assignment
import math

import clip
sys.path.append('/scratch/qingqu_root/qingqu1/siyich/multimodal-gap/utils')
sys.path.append('/scratch/qingqu_root/qingqu1/siyich/multimodal-gap')
from util import load_config_file
from simple_tokenizer import SimpleTokenizer
from model.model import CLIP

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


# step_list = [1,5,10,50,100,200,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
# step_list = [1,5,10,50,100,200,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000]
# step_list = [300,400,500,600,700,800,900]
# step_list = [1,5,10,50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]
# save_folder = '/scratch/qingqu_root/qingqu1/siyich/multimodal-gap/3_shrink_nw_train_checkpoints_1_5e-4_1e-1'

save_folder = '/scratch/qingqu_root/qingqu1/siyich/multimodal-gap/3_shrink_tiny_nw_train_checkpoints_1e-1_5e-4_1e-1'
step_list = [1,5,10,50,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]


for step in step_list:
    input_path = os.path.join(save_folder, f'shrink_tiny_1e-1-5e-6_{step}.npy')
    all_img_features, all_text_features = np.load(input_path)
    image_features = torch.from_numpy(all_img_features)
    text_features = torch.from_numpy(all_text_features)
    features_2d = np.concatenate([all_img_features, all_text_features], 0)


    plt.figure(figsize=(5, 5))
    plt.scatter(features_2d[:-len(all_img_features), 0], features_2d[:-len(all_img_features), 1], c='red')
    plt.scatter(features_2d[-len(all_img_features):, 0], features_2d[-len(all_img_features):, 1], c='blue')
    # connect the dots
    for i in range(len(all_img_features)):
        plt.plot([features_2d[i, 0], features_2d[len(all_img_features)+i, 0]], [features_2d[i, 1], features_2d[len(all_img_features)+i, 1]], c='black', alpha=0.1)
    save_path = os.path.join(save_folder, f"2d_{step}.png")
    plt.savefig(save_path) 


    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='3d')
    r = 1.0
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_surface(r*x, r*y, r*z, cmap=plt.cm.YlGnBu_r, alpha=0.2)
    ax.scatter(features_2d[:-len(all_img_features), 0], features_2d[:-len(all_img_features), 1], features_2d[:-len(all_img_features), 2], c='red')
    ax.scatter(features_2d[-len(all_img_features):, 0], features_2d[-len(all_img_features):, 1], features_2d[-len(all_img_features):, 2], c='blue')
    for i in range(len(all_img_features)):
        ax.plot([features_2d[i, 0], features_2d[len(all_img_features)+i, 0]], [features_2d[i, 1], features_2d[len(all_img_features)+i, 1]], [features_2d[i, 2], features_2d[len(all_img_features)+i, 2]], c='black', alpha=0.1)
    save_path = os.path.join(save_folder, f"3d_{step}.png")
    plt.savefig(save_path) 