FROM gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/prp:latest


## Change shell to allow conda init inside Dockerfile
SHELL ["/bin/bash", "--login", "-c"]

### Install vim
RUN sudo-apt get update
RUN sudo apt-get install vim


## Conda init
RUN conda init bash

## Create environment and install flower
RUN conda create -n flower_ffcv python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c pytorch -c conda-forge
RUN conda activate ffcv
RUN pip install --pre flwr[simulation]
