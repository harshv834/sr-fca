# Code for  "An Improved Algorithm for Clustered Federated Learning"


 
## Install Packages
```
conda create --yes -n cluster_fl python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba networkx tqdm -c pytorch -c conda-forge
conda activate cluster_fl
conda update ffmpeg
pip install ffcv ipdb "ray[tune]" scikit-learn
```

## Running experiments
To run the all experiments except Rotated CIFAR10

```
python run.py --dataset DATASET --clustering CLUSTERING --seed SEED
```
 - `DATASET` can take one of `synthetic`, `inv_mnist`, `rot_mnist`, `femnist`, `shakespeare`.
 - `CLUSTERING` can take one of `fedavg` (Global), `sr_fca`, `ifca` . Note that the local accuracy is computed during ONE_SHOT.
 - `SEED` is the random seed for experiment.

###  Rotated CIFAR10,
To run Local, Global and SR-FCA, we use
```
python cifar_run.py
```
To run IFCA comparison ,
```
python cifar_run_ifca.py
```

## Resources 
Note that each experiment was done for 3 random seeds with the best parameters. Total running time of all experiments on a single GPU is 1 week.
