FROM gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/prp:latest


## Change shell to allow conda init inside Dockerfile
SHELL ["/bin/bash", "--login", "-c"]

### Install vim
RUN sudo apt-get update
RUN sudo apt-get install vim tmux htop -y



## Create environment and install flower
RUN conda create -n cluster_fl python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba pytorch-c pytorch -c conda-forge
RUN source activate cluster_fl
## Conda init
SHELL ["conda", "run", "-n", "cluster_fl", "/bin/bash", "-c"]

RUN pip install ffcv
## See lightning 

## flwr works with python 3.7 only so it is irrelevant.
#RUN pip install --pre flwr[simulation]
