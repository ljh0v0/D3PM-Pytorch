#!/bin/bash 

model_f=$1 

gt_file=$model_f
gt_dir="$(dirname "${gt_file}")"
echo $gt_dir
bin=0
if [[ $gt_file == *"mnist"* ]]; then
    data="mnist"
    tar="datasets/images/mnist_test.npy" 
    bin=1
elif [[ $gt_file == *"cifar"* ]]; then
    data="cifar"
    tar='datasets/images/cifar_test.pkl_stats_cached.pkl'
    conf='config/diffusion_cifar10.json'
elif [[ $gt_file == *"celeba"* ]]; then
    data="celeba"
    tar='datasets/images/celeba_valid.pkl_stats_cached.pkl'
    conf='config/diffusion_celeba.json'
elif [[ $gt_file == *"omni"* ]]; then
    data="omni"
    tar="datasets/images/omni_test.npy" 
else 
    echo "unknow data for $gt_file" 
    exit
fi
echo "eval data: $data"

ns=10000
python sample.py --config $conf --model_dir $model_f --sample_dir $gt_dir --batch_size 32 --n_samples 32
echo $gt_dir
for entry in `find $gt_dir -maxdepth 1 -name "*sample*npy"`
do
    #echo "$entry"
    sample=$entry 
    T=${sample//.npy/.pkl}
    if [[ -f "$T" ]]; then 
        T=''
    else 
	    if [ ! -r $sample ]; then 
		    echo "invalid link $sample"
		    continue 
	    fi 
        echo "eval the npy file $sample "
        python tool/pytorch-fid/fid_score.py \
        --binarized $bin \
        --batch_size 50 --gpu 1 --path $sample $tar
    fi
done
