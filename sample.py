import os
import json
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import numpy  as np

from main import DiffusionModel, samples_fn
from config import get_config

if __name__ == "__main__":

    '''parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.")
    parser.add_argument("--model_dir", type=str, default='', help="Path to model for loading.")
    parser.add_argument("--sample_dir", type=str, default='samples', help="Path to save generated samples.")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of generated samples in evaluation.")
    parser.add_argument("--n_samples", type=int, default=32, help="Number of generated samples in evaluation.")

    args = parser.parse_args()'''

    '''class Args():
        train = True
        config = "config/diffusion_celeba.json"
        ckpt_dir = "exp/celeba/"
        ckpt_freq = 20
        n_gpu = 1'''


    class Args():
        model_dir = "exp/cifa10/step0/epo515.ckpt"
        sample_dir = 'samples/cifar10/step0_epo515/'
        batch_size = 16
        n_samples = 160


    args = Args()

    config = get_config()

    d3pm = DiffusionModel(config)

    d3pm.cuda()
    state_dict = torch.load(args.model_dir)
    d3pm.load_state_dict(state_dict['state_dict'])
    d3pm.eval()

    out = []

    for k in tqdm(range(int(args.n_samples // args.batch_size))):

        sample = samples_fn(d3pm.model,
                            d3pm.diffusion,
                            (args.batch_size, 3, config.dataset.resolution, config.dataset.resolution))

        imgs = sample['samples'].float().cpu().view(args.batch_size, 3, config.dataset.resolution, config.dataset.resolution)
        out.append(imgs)

        if not os.path.isdir(args.sample_dir):
            os.makedirs(args.sample_dir)
        filepath = os.path.join(args.sample_dir, f'sample_imgs_%d.png' % k)
        save_image(imgs, filepath, normalize=True, scale_each=True, nrow=4)

    out_pt = torch.cat(out)

    out = out_pt.numpy()
    sample_path = os.path.join(args.sample_dir, f'sample_npy.npy')
    np.save(sample_path, out)
