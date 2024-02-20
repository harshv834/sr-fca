# Code for  "An Improved Federated Clustering Algorithm  with Model-based Clustering"


 
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
 - `CLUSTERING` can take one of 
    - `fedavg` (Global) 
    - `sr_fca` 
    - `ifca`
    - `cfl`
    - `oneshot_kmeans`(ONE_SHOT with KMeans)
    - `soft_ifca` (FedSoft)
    - `oneshot_ifca` (ONE_SHOT to initialize IFCA)
    - `sr_fca_merge_refine` : SR_FCA with MERGE before RECLUSTER step.
 - `SEED` is the random seed for experiment.
 - `--from_init` : This is an optional argument to use local models if they have been already trained by running ONE_SHOT earlier. Applicable to `sr_fca`, `oneshot_kmeans` and `oneshot_ifca`.

Note that the Local baseline corresponds to the accuracy of local models, and is computed after ONE_SHOT has been run.

###  Rotated CIFAR10,
The code for CIFAR10 experiments is provided in `cifar_code` folder.
To run SR_FCA on CIFAR10 
```
cd cifar_code
python cifar_run.py --seed SEED --het HET
```
Here, `SEED` is the random seed and `HET` can be either `rot` or `label` corresponding to Rotated and Label CIFAR10. 

To run other baselines (`cfl`,`global`,`ifca`, `oneshot_kmeans`), use the corresponding `.py` file. For instance, for `ifca`, use `cifar_run_ifca.py` instead of `cifar_run.py` in the above command. `oneshot_kmeans` and `sr_fca`  support the `--from_init` argument here as well.

### Real Datasets (FEMNIST, Shakespeare)
Please load the leaf submodule attached to this repo and extract FEMNIST and Shakespeare from the leaf datasets. 
```
git submodule update --init --recursive
```

Also, update the `pretrained_data_path` parameter in `configs/experiment/femnist.yaml` and `configs/experiment/shakespeare.yaml` files with the location of the extracted datasets before running their experiments.

## Tuning
Tuning for datasets other than CIFAR10, can be performed by using same syntax as `run.py`, but instead using `tuning.py` with the number of trials specified by `--num_samples` command line argument. Note that this requires `ray[tune]` to be installed.



## Resources 
Note that each experiment was done for 3 random seeds with the best parameters. Total running time of all experiments on a single GPU is 1 week.

