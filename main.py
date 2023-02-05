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

"""CIFAR10 D3PM."""

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
import ml_collections
import json
import argparse

import model
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

    def __init__(self, config, exp_dir):
        super().__init__()

        self.config = config
        self.exp_dir = exp_dir

        assert self.config.dataset.name in {'cifar10', 'MockCIFAR10'}
        self.num_bits = 8

        # Ensure that max_time in model and num_timesteps in the betas are the same.
        self.num_timesteps = self.config.model.diffusion_betas.num_timesteps
        self.ema_decay = self.config.train.ema_decay

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
            num_heads=self.config.model.args.num_heads,
            dropout=self.config.model.args.dropout,
            model_output=self.config.model.args.model_output,
            num_pixel_vals=self.config.model.args.num_pixel_vals,
            img_size=self.config.dataset.resolution
        )

        self.ema = model.UNet(
            in_channel=self.config.model.args.in_channel,
            out_channel=self.config.model.args.out_channel,
            channel=self.config.model.args.channel,
            channel_multiplier=self.config.model.args.channel_multiplier,
            n_res_blocks=self.config.model.args.n_res_blocks,
            attn_resolutions=self.config.model.args.attn_resolutions,
            num_heads=self.config.model.args.num_heads,
            dropout=self.config.model.args.dropout,
            model_output=self.config.model.args.model_output,
            num_pixel_vals=self.config.model.args.num_pixel_vals,
            img_size=self.config.dataset.resolution
        )

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
        # t = np.random.randint(size=(img.shape[0],), low=0, high=self.num_timesteps, dtype=np.int32)
        t = (torch.randint(low=0, high=(self.num_timesteps), size=(img.shape[0],))).to(img.device)
        loss = self.diffusion.training_losses(self.model, img, t).mean()

        accumulate(self.ema, self.model.module if isinstance(self.model, nn.DataParallel) else self.model,
                   self.ema_decay)

        if self.global_step % self.config.train.log_loss_every_steps == 0:
            self.logger.log_metrics({"train_loss": loss}, step=self.global_step)

        if self.global_step % self.config.train.retain_checkpoint_every_steps == 0:
            filename = f"checkpoint_{self.global_step}.ckpt"
            ckpt_path = os.path.join(self.exp_dir, "retain-checkpoint", filename)
            self.trainer.save_checkpoint(ckpt_path)

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

        bpd_dict = self.diffusion.calc_bpd_loop(self.ema, img)
        total_bpd = bpd_dict['total'].mean()
        prior_bpd = bpd_dict['prior'].mean()

        return {'val_loss': loss, "total_bpd": total_bpd, "prior_bpd": prior_bpd}
        #return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.logger.log_metrics({"val_loss": avg_loss}, step=self.global_step)

        avg_total_bpd = torch.stack([x['total_bpd'] for x in outputs]).mean()
        self.logger.log_metrics({"total bpd": avg_total_bpd}, step=self.global_step)

        avg_prior_bpd = torch.stack([x['prior_bpd'] for x in outputs]).mean()
        self.logger.log_metrics({"prior bpd": avg_prior_bpd}, step=self.global_step)

        # sample
        shape = (64, 3, self.config.dataset.resolution, self.config.dataset.resolution)
        sample = samples_fn(self.ema, self.diffusion, shape)

        grid = make_grid(sample['samples'], nrow=8)
        ndarr = grid.permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        self.logger.experiment.log_image(im, name='val-img-step' + str(self.global_step),
                                         step=self.global_step)

        return {'val_loss': avg_loss}

    def val_dataloader(self):
        valid_loader = DataLoader(self.valid_set,
                                  batch_size=self.config.train.batch_size,
                                  shuffle=False,
                                  pin_memory=True)

        return valid_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1, help="Training or evaluation?")
    parser.add_argument("--comet", type=int, default=0, help="use comet logger")
    parser.add_argument("--exp_dir", type=str, default='exp', help="Path to folder to save checkpoints.")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume.")
    parser.add_argument("--config_json", type=str, default=None, help="Path to config json file.")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of available GPUs.")
    parser.add_argument("--ckpt_freq", type=int, default=20, help="Frequency of saving the model (in epoch).")
    args = parser.parse_args()


    if args.config_json is not None:
        logger.info('Reading config from JSON: %s', args.config_json)
        with open(args.config_json, 'r') as f:
            config = ml_collections.ConfigDict(json.loads(f.read()))
    else:
        config = get_config()

    # save config as json
    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)
    cfg_path = os.path.join(args.exp_dir, 'config.json')
    if not os.path.exists(cfg_path):
        with open(cfg_path, 'w') as file:
            file.write(config.to_json_best_effort(sort_keys=True, indent=4) + '\n')

    d3pm = DiffusionModel(config, exp_dir=args.exp_dir)

    if args.train:
        checkpoint_callback = ModelCheckpoint(dirpath=args.exp_dir,
                                              verbose=False,
                                              save_last=True,
                                              save_weights_only=False,
                                              every_n_epochs=args.ckpt_freq,
                                              save_on_train_epoch_end=True
                                              )

        comet_logger = CometLogger(
            api_key="",
            save_dir="logs/",  # Optional
            project_name="D3PM",  # Optional
            experiment_name="d3pm-gaussian",  # Optional
        )

        trainer = pl.Trainer(
            max_steps=config.train.num_train_steps,
            gradient_clip_val=1.,
            check_val_every_n_epoch=config.train.eval_every_epoch,
            enable_progress_bar=True,
            enable_checkpointing=True,
            callbacks=[checkpoint_callback],
            logger=comet_logger,
            accelerator="gpu",
            devices=args.n_gpu,
            strategy="ddp"
        )

        trainer.fit(d3pm, ckpt_path=args.resume)
