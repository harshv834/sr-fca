#conda create --yes -n cluster_fl python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba networkx -c pytorch -c conda-forge
#source ~/.bashrc
#source activate cluster_fl
#python -c "import networkx;print(networkx.__version__)"
#conda update ffmpeg
#pip install ffcv scikit-learn tqdm ipdb
#python -c "import ffcv;print(ffcv.__version__)"
for seed in 1234 12345 1729 1759 1769
do
    python cifar_run_ifca.py --seed $seed --het label
done
# python tuning.py --dataset $1 --clustering $2 --seed $3
