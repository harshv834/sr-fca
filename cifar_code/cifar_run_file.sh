for seed in 1729 1234 12345 1759 1769 
do
    #python cifar_run.py --het label --seed $seed
    python cifar_run_oneshot_kmeans.py --het label --seed $seed --from_init
done

# python cifar_run.py --het label --seed 1234
# python cifar_run_global.py --het label --seed 1234
# python cifar_run_ifca.py --het label --seed 1234
# python cifar_run_oneshot_kmeans.py --het label --seed 1234 --from_init
# python cifar_run_cfl.py --het label --seed 1234
