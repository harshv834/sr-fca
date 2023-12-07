## Start from nautilus base image
FROM gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/prp:latest 

ARG USERNAME=jovyan
## Install mamba and install ffcv dependencies with mamba
#RUN conda install -y mamba -c conda-forge &&  
RUN conda create --yes -n cluster_fl python=3.9 cupy  pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba networkx tqdm -c pytorch -c conda-forge && conda clean -tipy
 
## For running conda environment in docker
RUN conda init bash && echo "conda activate cluster_fl" >> /home/$USERNAME/.bashrc
SHELL ["conda","run", "-n", "cluster_fl", "/bin/bash","-c"]

## Update ffmpeg and install ffcv
RUN conda update ffmpeg --yes &&  pip install ffcv 