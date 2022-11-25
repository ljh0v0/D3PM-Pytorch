#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import comet_ml 
from loguru import logger 
    
# Add the following code anywhere in your machine learning file
from datetime import datetime 
import json 
import yaml
import os
import pickle
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import datasets, transforms, utils

from PIL import Image
import re 

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from tool.fid.inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', type=str, nargs=2,
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--batch_size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')
parser.add_argument('--binarized', default=0, type=int,
    help='discretize the images')

def try_write_EVAL(d):
    EVAL = os.environ.get('EVAL')
    if not EVAL:  # 
        EVAL = 'unknown'
        return 
    main_json = './track/sub/%s/main.json'%(EVAL)
    if not os.path.exists(os.path.dirname(main_json)):
        os.makedirs(os.path.dirname(main_json))
    writer = os.path.basename(__file__)
    if os.path.exists(main_json):
        de = json.load(open(main_json, 'r'))
    else:
        de = {} 
    dew = de[writer] if writer in de else {} 
    dew.update(d)
    writer = '%s-%s-%s'%(writer,d['pred'],d['gt'])
    de[writer] = dew
    json.dump(de, open(main_json, 'w'), indent=2)
    logger.info('write to %s'%main_json)

def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]


@torch.no_grad()
def get_activations(files, model, batch_size=50, dims=2048,
                    cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print((f'Warning: batch size {batch_size} is bigger than the data size. {len(files)}'
               'Setting batch size to data size'))
        batch_size = len(files)

    pred_arr = np.empty((len(files), dims))
    assert(len(files[0].shape) == 3), f'expected to get 3D tensor, get: {files[0].shape}'
    if files[0].shape[-1] == 1:
        cat2rgb = 1
    else:
        assert(files[0].shape[-1] == 3), f'expected to get tensor HW3, get: {files[0].shape}' 
        cat2rgb = 0
    H,W,_ = files[0].shape 

    for i in range(0, len(files), batch_size):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i
        end = i + batch_size

        #images = np.array([imread(str(f)).astype(np.float32)
        #                   for f in files[start:end]])
        if cat2rgb:
            images = files[i:min(len(files), i+batch_size)].reshape(-1, 1, H, W) 
            images =  np.concatenate([images, images, images], 1)
        else:
            images = files[i:min(len(files), i+batch_size)].reshape(-1, H, W, 3).transpose((0,3,1,2)) # N.3.H.W

        # Reshape to (n_images, 3, height, width)
        # images = images.transpose((0, 3, 1, 2))
        if images.max() > 1:
            images = images / 255.0

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        batch = batch.to(device) # .cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg, 'sigma1:: ', sigma1.min(), sigma1.max())
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50,
                                    dims=2048, cuda=False, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, dims, cuda):
    if path.endswith('.npz') or path.endswith('.npy'):
        logger.info('[load] {}', path)
        f = np.load(path)
        if torch.is_tensor(f):
            f = f.numpy()
        if args.binarized:
            # perform bernoulli sampling 
            ## logger.info('path: {}. type={}, {}', path, type(f), f.shape)
            # assert(f.max() > 1), f.max() 
            if f.max() > 1: f = f / 255.0
            logger.info('path: {}. type={}, {} value:{}-{}', path, type(f), f.shape,
                f.min(), f.max())
            #f = f / f.max() # 0.1)*1.0
            f = torch.bernoulli(torch.from_numpy(f))*1.0
            f = f.numpy()

        assert(len(f.shape) == 4), f'expect 4D tensor, get {f.shape}'
        outp = path.replace('npy', 'png') 
        logger.info(f'load {path}; shape: {f.shape}; max: {f.max()} | min={f.min()} | type={type(f)}')
        # deal with different saved shape | target: N,H,W,3 or N,H,W,1
        if f.shape[1] == 3 and f.shape[2] > 3 and f.shape[3] > 3: 
            logger.info('get data: 3HW; do tranpose to make it HW3')
            N,_,H,W = f.shape 
            f = f.reshape(N,3,H,W).transpose((0,2,3,1))
        elif f.shape[1] == 1 and f.shape[2] > 3 and f.shape[3] > 3: 
            logger.info('get data: 1HW; do tranpose to make it HW1')
            N,_,H,W = f.shape 
            f = f.reshape(N,1,H,W).transpose((0,2,3,1))
        # save the image as png 
        for step in range(20):
            outp = path.replace('.npy', '0.png') ## %step) 
            if f.shape[0] < step*64+64:
                break
            sample = torch.from_numpy(f)[64*step:64*step+64].float()
            if sample.shape[-1] == 3: # HW3 RGB
                N1,H1,W1,_ = sample.shape
                sample = sample.view(N1,H1,W1,3).permute((0,3,1,2)).contiguous() 
            elif sample.shape[-1] == 1:
                sample = sample[:,:,:,0].unsqueeze(1) # B,H,W,1 -> B,1,H,W 
            utils.save_image(sample, outp, nrow=8, padding=1, pad_value=0, normalize=True)
            #if 'test0.png' not in outp:
            #    experiment.log_image(Image.open(outp), path.split('.npy')[0].split('/')[-1], step=step)

        m, s = calculate_activation_statistics(f, model, batch_size,
                 + dims, cuda)
        #with open(path.replace('npy', 'pkl'), 'wb') as f:
        #    pickle.dump([m,s], f) 

    elif path.endswith('.pkl'):
        logger.debug(f'load {path}; ')
        cached_path = path.replace('npy', 'pkl') if path.endswith('npy') else path 
        with open(cached_path, 'rb') as f:
            m, s = pickle.load(f) 

    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        images = np.array([imread(str(f)).astype(np.float32)
                           for f in files])
        m, s = calculate_activation_statistics(images, model, batch_size,
                                               dims, cuda)

    return m, s


def calculate_fid_given_paths(paths, batch_size, cuda, dims, model):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s; not exists' % p)
    pred_stats_cached = paths[0] + '_stats_cached.pkl'
    if os.path.exists(pred_stats_cached) and not args.binarized:
        # if samples require binarized, not load cached 
        logger.info('load cached stats: {}', pred_stats_cached)
        with open(pred_stats_cached, 'rb') as f:
            m1, s1 = pickle.load(f)
    else:
        m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size, dims, device)
        slurm_dir=f"/checkpoint/{os.getenv('USER')}/{os.getenv('SLURM_JOB_ID', None)}"
        if os.path.exists(slurm_dir) and not args.binarized:
            cachedname = pred_stats_cached.split('/')[-1]
            # savedp = slurm_dir + '/' + cachedname
            if os.path.islink(pred_stats_cached):
                os.unlink(pred_stats_cached)
            with open(pred_stats_cached, 'wb') as f:
                pickle.dump([m1, s1], f)
            # os.symlink(savedp, pred_stats_cached) 
        elif not args.binarized:
            with open(pred_stats_cached, 'wb') as f:
                pickle.dump([m1, s1], f)
            logger.info('write cached stats: {}', pred_stats_cached)

    # GT's statistics 
    gt_stats_cached = paths[1] + '_stats_cached.pkl'
    if os.path.exists(gt_stats_cached) and not args.binarized: 
        # if binarized, not load stats cached
        logger.info('load cached stats: {}', gt_stats_cached)
        with open(gt_stats_cached, 'rb') as f:
            m2, s2 = pickle.load(f)
    else:
        m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size, dims, device)
        if not args.binarized: # if binarized, dont save into stats
            with open(gt_stats_cached, 'wb') as f:
                pickle.dump([m2, s2], f)
        logger.info('write cached stats: {}', gt_stats_cached)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value

def try_get_exp_key(exp_path):
    exp_path_yml = os.path.join(os.path.dirname(exp_path), 'cfg.yml')
    logger.info('try to load: {}', exp_path_yml)
    if not os.path.exists(exp_path_yml):
        return ''
    exp_path_yml = yaml.safe_load(open(exp_path_yml, 'r'))
    key = exp_path_yml.get('exp_key', '')
    logger.info('get key: {}', key)
    if len(key) > 5:
        key = key[:5]
    return key

if __name__ == '__main__':
    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = 1 #args.gpu
    logfile = os.path.dirname(args.path[0]) + '/eval_fid.log' 
    logger.add(logfile, colorize=False, level='DEBUG')
    logger.info('logfile: {}', logfile)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]

    model = InceptionV3([block_idx])
    # if cuda:
    # model.cuda()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        logger.debug(torch.cuda.get_device_name(device))
    model = model.to(device) 
    
    gt = args.path[1] 
    pred = args.path[0] 
    if '.txt' in pred: 
        # list of samples in pred 
        pred_list = open(pred, 'r').readlines()
        pred_list = [p.strip('\n') for p in pred_list]
        logger.info('get list of npy to eval: {}', len(pred_list))
        assert(not args.binarized), 'not support yet'
    else: 
        pred_list = [pred]

    # for each npy in the list, eval the fid score 
    for p in pred_list:
        pred = args.path[0] = p 
        exp_key_prefix = try_get_exp_key(pred) 

        fid_value = calculate_fid_given_paths(args.path,
             args.batch_size, args.gpu != '', args.dims, model)
        datestr = datetime.now().strftime("%m-%d-%H-%M-%S")
        sample_key = re.findall('sample\d+.npy', args.path[0]) + re.findall('sample\d+.pkl', args.path[0]) 
        if len(sample_key) > 0:
            sample_key = '@E' + re.findall('\d+', sample_key[0])[0] 
        else: 
            sample_key = ' '
        target_fid, output_fid = args.path[1], args.path[0]
        if args.binarized:
            target_fid = target_fid + '-binarized'
            output_fid = output_fid + '-binarized'
        # key_prefix = experiment.get_key()[:5]
        # '[url] %s | '
        msg = '[FID] %.2f %s ( %s ) \n| `%s` |`%s` | '%(
               fid_value, sample_key, # , key_prefix, 
               exp_key_prefix, # experiment.url, 
               target_fid, output_fid)
        #fid_score_json
        #with open('results.md', 'a') as f:
        #    msg = '[FID] %.2f %s | [url] %s | \n | `%s` |`%s` | '%(
        #        fid_value, sample_key, experiment.url, args.path[1], args.path[0])
        #    f.write('\n %s '%datestr + msg)
        #fid_score_json[pred] = {'gt': gt, 'fid':fid_value}
        #json.dump(fid_score_json, open('.results/fid_score.json', 'w'), indent=4)
        logger.info('-'*10 + '\n' + msg)
        #if os.environ.get('EVAL') is not None:
        #    #with open('.results/eval_out/toc.md', 'a') as f: 
        #    #    f.write('[fid_score] .results/eval_out/%s.md \n'%(os.environ.get('EVAL')))
        #    with open('.results/eval_out/%s.md'%(os.environ.get('EVAL')), 'a') as f:
        #        f.write(datestr + ' fid score ' + ' \n' + msg + '\n') 
        #    logger.info('write to %s'%('.results/eval_out/%s.md'%(os.environ.get('EVAL'))))
        #json_path = os.path.dirname(args.path[0]) + '/eval_fid.json' 
        #if os.path.exists(json_path):
        #    fid_score_dict = json.load(open(json_path, 'r'))
        #else:
        #    fid_score_dict = {}
        #fid_score_dict[datestr] = {'gt': gt, 'pred': pred, 'fid':fid_value, 'url': experiment.url, 'time':datestr}
        #json.dump(fid_score_dict, open(json_path, 'w'), indent=4)
        #try_write_EVAL(fid_score_dict[datestr])'''
