#  Discrete Denoising Diffusion Probabilistic Models(D3PMs) for image

Unofficial PyTorch implementation of Discrete Denoising Diffusion Probabilistic Models(D3PMs)

🚧 UNDER CONSTRUCTION...
## Experiments
### D3PM-Gaussian on CIFAR10

![gaussian_cifar10](https://user-images.githubusercontent.com/60313002/230746837-35d37f14-b696-4235-8d7a-e3fe8e1a0314.png)

## Training
```
python3 main.py --train 1 --comet 1 --exp_dir "path/to/exp_dir/" --n_gpu 1 --ckpt_freq 20
```

## Sampling
```
python sample.py --config "path/to/config.json" --model_dir "path/to/model.ckpt" --sample_dir "path/to/save/sample" --batch_size 128 --n_samples 10000
```

## Reference

**Paper**

[Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006)

**Code**

[google-research/d3pm](https://github.com/google-research/google-research/tree/master/d3pm)
