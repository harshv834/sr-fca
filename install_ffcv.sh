conda create --yes -n cluster_fl python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba networkx -c pytorch -c conda-forge
source ~/.bashrc
source activate cluster_fl
conda update ffmpeg
pip install ffcv "ray[tune]" ipdb  
python -c "import ffcv;print(ffcv.__version__)"
# python tuning.py --dataset $1 --clustering $2 --seed $3