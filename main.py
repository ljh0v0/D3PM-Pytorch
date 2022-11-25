# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CIFAR10 diffusion model."""

from typing import Dict
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torch.nn import functional as F
import os
from loguru import logger
from comet_ml import Experiment
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from PIL import Image

import model
import model_1
import utils
import datasets
from diffusion_categorical import make_diffusion
from config import get_config


def samples_fn(model, diffusion, shape, num_timesteps=None):
    samples = diffusion.p_sample_loop(model, shape, num_timesteps)

    return {
        'samples': samples
    }

def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


class DiffusionModel(pl.LightningModule):
    """diffusion model."""

    def __init__(self, config):
        super().__init__()

        self.config = config

        assert self.config.dataset.name in {'cifar10', 'MockCIFAR10'}
        self.num_bits = 8

        # Ensure that max_time in model and num_timesteps in the betas are the same.
        self.num_timesteps = self.config.model.diffusion_betas.num_timesteps
        self.ema_decay = self.config.train.ema_decay
        self.log_img_every_epoch = self.config.train.log_img_every_epoch

        assert self.config.train.num_train_steps is not None
        assert self.config.train.num_train_steps % self.config.train.substeps == 0
        assert self.config.train.retain_checkpoint_every_steps % self.config.train.substeps == 0

        # Build Unet Model
        self.model = model.UNet(
            in_channel=self.config.model.args.in_channel,
            out_channel=self.config.model.args.out_channel,
            channel=self.config.model.args.channel,
            channel_multiplier=self.config.model.args.channel_multiplier,
            n_res_blocks=self.config.model.args.n_res_blocks,
            attn_resolutions=self.config.model.args.attn_resolutions,
            dropout=self.config.model.args.dropout,
            model_output=self.config.model.args.model_output,
            num_pixel_vals=self.config.model.args.num_pixel_vals,
            img_size = self.config.dataset.resolution
        )

        self.ema = model.UNet(
            in_channel=self.config.model.args.in_channel,
            out_channel=self.config.model.args.out_channel,
            channel=self.config.model.args.channel,
            channel_multiplier=self.config.model.args.channel_multiplier,
            n_res_blocks=self.config.model.args.n_res_blocks,
            attn_resolutions=self.config.model.args.attn_resolutions,
            dropout=self.config.model.args.dropout,
            model_output=self.config.model.args.model_output,
            num_pixel_vals=self.config.model.args.num_pixel_vals,
            img_size=self.config.dataset.resolution
        )

        # self.model = model_1.Unet(
        #     dim=self.config.model.args.channel,
        #     channels=self.config.model.args.in_channel,
        #     out_dim=self.config.model.args.out_channel,
        #     dim_mults=self.config.model.args.channel_multiplier,
        #     model_output=self.config.model.args.model_output,
        #     num_pixel_vals=self.config.model.args.num_pixel_vals
        # )
        #
        # self.ema = model_1.Unet(
        #     dim=self.config.model.args.channel,
        #     channels=self.config.model.args.in_channel,
        #     out_dim=self.config.model.args.out_channel,
        #     dim_mults=self.config.model.args.channel_multiplier,
        #     model_output=self.config.model.args.model_output,
        #     num_pixel_vals=self.config.model.args.num_pixel_vals
        # )

        print(self.model)

        # Build Diffusion model
        self.diffusion = make_diffusion(self.config.model)

    def setup(self, stage):

        self.train_set, self.valid_set = datasets.get_train_data(self.config)

    def forward(self, x):
        return self.diffusion.p_sample_loop(self.model, x.shape)

    def configure_optimizers(self):

        if self.config.train.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.train.learning_rate)
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        img, _ = batch
        save_image(img, './samples/x.png', normalize=True, scale_each=True, nrow=2)
        # t = np.random.randint(size=(img.shape[0],), low=0, high=self.num_timesteps, dtype=np.int32)
        t = (torch.randint(low=0, high=(self.num_timesteps), size=(img.shape[0],))).to(img.device)
        loss = self.diffusion.training_losses(self.model, img, t).mean()

        accumulate(self.ema, self.model.module if isinstance(self.model, nn.DataParallel) else self.model, self.ema_decay)

        if batch_idx % self.config.train.log_loss_every_steps == 0:
            self.logger.log_metrics({"epoch": self.current_epoch, "steps": batch_idx, "train/loss": loss})

        return {'loss': loss}

    def train_dataloader(self):

        train_loader = DataLoader(self.train_set,
                                  batch_size=self.config.train.batch_size,
                                  shuffle=True,
                                  pin_memory=True)

        return train_loader

    def validation_step(self, batch, batch_nb):
        img, _ = batch
        # t = np.random.randint(size=(img.shape[0],), low=0, high=self.num_timesteps, dtype=np.int32)
        t = (torch.randint(low=0, high=(self.num_timesteps), size=(img.shape[0],))).to(img.device)
        loss = self.diffusion.training_losses(self.ema, img, t).mean()

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.logger.log_metrics({"epoch": self.current_epoch, "val/loss": avg_loss})

        if self.current_epoch % self.log_img_every_epoch == 0:
            shape = (16, 3, self.config.dataset.resolution, self.config.dataset.resolution)
            sample = samples_fn(self.ema, self.diffusion, shape, num_timesteps=10)

            grid = make_grid(sample['samples'], nrow=4)
            ndarr = grid.permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            self.logger.experiment.log_image(im, name='val-img-epoch' + str(self.current_epoch), step=self.current_epoch)

        return {'val_loss': avg_loss}

    def val_dataloader(self):
        valid_loader = DataLoader(self.valid_set,
                                  batch_size=self.config.train.batch_size,
                                  shuffle=False,
                                  pin_memory=True)

        return valid_loader


if __name__ == '__main__':
    class Args():
        train = True
        comet = 1
        ckpt_dir = "exp/cifa10/1122/"
        ckpt_freq = 20
        n_gpu = 1
        model_dir= 'exp/cifa10/last.ckpt'


    args = Args()
    config = get_config()

    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    d3pm = DiffusionModel(config)

    if args.train:
        checkpoint_callback = ModelCheckpoint(dirpath=args.ckpt_dir,
                                              filename='ddp_{epoch:02d}-{val_loss:.2f}',
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=-1,
                                              save_weights_only=True,
                                              mode='min',
                                              every_n_epochs=args.ckpt_freq
                                              )

        comet_logger = CometLogger(
            api_key="nGRMV8S1NSghQEh2WmxFb3ZnA",
            save_dir="logs/",  # Optional
            project_name="discrete DPM",  # Optional
            experiment_name="test-old-model",  # Optional
        )

        trainer = pl.Trainer(fast_dev_run=False,
                             gpus=args.n_gpu,
                             max_steps=config.train.num_train_steps,
                             gradient_clip_val=1.,
                             enable_progress_bar=True,
                             enable_checkpointing=True,
                             callbacks=[checkpoint_callback],
                             logger=comet_logger
                             )

        trainer.fit(d3pm)
