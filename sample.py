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
import re
import ml_collections
import math

from main import DiffusionModel, samples_fn
from config import get_config

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, default='', help="Path to config.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model for loading.")
    parser.add_argument("--sample_dir", type=str, default='samples', help="Path to save generated samples.")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of generated samples in evaluation.")
    parser.add_argument("--n_samples", type=int, default=32, help="Number of generated samples in evaluation.")

    args = parser.parse_args()

    step = int(re.findall(r'\d+', args.model_dir)[-1])
    exp_dir = os.path.dirname(args.model_dir)
    if args.config_json is not None:
        with open(args.config_json, 'r') as f:
            config = ml_collections.ConfigDict(json.loads(f.read()))
    else:
        config = get_config()

    if not os.path.isdir(args.sample_dir):
        os.makedirs(args.sample_dir)

    d3pm = DiffusionModel(config, exp_dir)

    d3pm.cuda()
    state_dict = torch.load(args.model_dir)
    d3pm.load_state_dict(state_dict['state_dict'])
    d3pm.eval()

    out = []

    for k in tqdm(range(int(args.n_samples // args.batch_size))):

        sample = samples_fn(d3pm.ema,
                            d3pm.diffusion,
                            (args.batch_size, 3, config.dataset.resolution, config.dataset.resolution))

        imgs = sample['samples'].float().cpu().view(args.batch_size, 3, config.dataset.resolution, config.dataset.resolution)
        out.append(imgs)

        if k < 1:
            filepath = os.path.join(args.sample_dir, f'sample_10K_%i.png' % step)
            save_image(imgs, filepath, normalize=True, scale_each=True, nrow=int(math.sqrt(args.batch_size)))

    out_pt = torch.cat(out)

    out = out_pt.numpy()
    sample_path = os.path.join(args.sample_dir, f'sample_10K_%i.npy' % step)
    np.save(sample_path, out)
