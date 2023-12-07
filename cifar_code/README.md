## Instructions to run file

Start from prp base image
```
conda create --name cluster_fl --clone base
conda install  networkx tqdm opencv libjpeg-turbo -c conda-forge --yes -n cluster_fl
mamba install  networkx opencv libjpeg-turbo -c conda-forge --yes -n cluster_fl

conda install -c conda-forge libstdcxx-ng --yes -n cluster_fl
sudo cd /lib/x86_64-linux-gnu/libstdc++.so.6
sudo rm -rf libstdc++.so.6*
sudo cp /opt/conda/pkgs/libstdcxx-ng-12.2.0-h46fd767_19/lib/libstdc++.so.6.0.30 .
sudo ln -s libstdc++.so.6.0.30 libstdc++.so.6
cd /base_vol/
conda activate environment
pip install ffcv ipdb tqdm "ray[tune]" optuna

```


- Start from prp base image
- Clone base environment (this does not clone mamba and conda)
- Install requirements for ffcv
- Fix opencv thing (ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /opt/conda/envs/cluster_fl/lib/python3.10/site-packages/scipy/spatial/_ckdtree.cpython-310-x86_64-linux-gnu.so))
- Install ffcv 
- Done ffs