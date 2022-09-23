# Code for  "Clustered Federated Learning with Heterogeneous Models"



## 1. MNIST 

### Install Packages
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install tqdm jupyterlab
```

### Running experiments
To run the all experiments (Local, Global, SR-FCA)

```
python mnist_run.py
```
We used 3 random seeds and all the experiments took around 2 day to run on a single GPU.

## 2. CIFAR10
### Install Packages
```
conda create -n ffcv python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c pytorch -c conda-forge 
conda activate ffcv
pip install ffcv
pip install tqdm jupyterlab                    
```
### Running experiments
To run the all experiments (Local, Global, SR-FCA)

```
python cifar_run.py
```
We used 3 random seeds and all the experiments took around 7 days to run on a single GPU.

## 3. FEMNIST


### Install Packages
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install tqdm jupyterlab pillow
```

### Download FEMNIST dataset 
```
mkdir experiments
cd experiments
mkdir datasets 
git clone https://github.com/TalwalkarLab/leaf.git
cd leaf/data/femnist
./preprocess.sh -s niid --sf 0.05 -k 0 -t sample
```


### Running experiments
To run the all experiments (Local, Global, SR-FCA)

```
python femnist_run.py 
```
We used 2 random seeds and all the experiments took around 4 days to run on a single GPU.


### Comparison to IFCA 
```
python ifca_comparison.py
```
We used 2 random seeds and all the experiments took around 4 days to run on a single GPU.

Use the same random seed for ifca and sr-fca.




## Main Requirements --
 - Run function runs everything according to arguments. Hyperparameter tuning generates hyperparameter configs for each case.
 - Add tensorboard logging
 - Different datasets (Real and simulated)
 - Different optimizers
 - Different FL schedules and subalgos
 - Parameter config for every choice of running config
 - Diffferent datasets and their evaluation params
 - Different algorithms ( SR -FCA)
 - Different distance metrics and different threshold selection techniques
 - run file should return all parameters.
 - Run base optimization in parallel (as parallel as possible)
 - Model structure tied to problem.( Let's say this is not customizable)
