FROM gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/prp:latest


## Change shell to allow conda init inside Dockerfile
SHELL ["/bin/bash", "--login", "-c"]

### Install vim
RUN sudo apt-get update
RUN sudo apt-get install vim tmux htop -y



## Create environment and install all dependencies (ffcv, lightning, optuna and ray-tune)
RUN conda create -n cluster_fl python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba networkx tqdm ipdb flake8-black pytorch-lightning optuna ray-tune -c pytorch -c conda-forge 
#&& conda activate cluster_fl && conda update ffmpeg && pip install ffcv ray_lightning 
RUN source activate cluster_fl
## Conda init
SHELL ["conda", "run", "-n", "cluster_fl", "/bin/bash", "-c"]
RUN conda update ffmpeg
RUN pip install ffcv 

RUN conda install pytorch-lightning optuna ray-tune -c conda-forge
conda activate cluster_fl && conda update ffmpeg && pip install ffcv && conda install pytorch-lightning optuna ray-tune -c conda-forge

## flwr works with python 3.7 only so it is irrelevant.
#RUN pip install --pre flwr[simulation]
