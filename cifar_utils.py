import torch
import os

def calc_local_acc_from_old(base_path):
    init_path = os.path.join(base_path, "init")
    test_acc = 0.0
    for i in range(16):
        metrics = torch.load(os.path.join(init_path, "node_{}".format(i), "metrics_2399.pkl"))
        test_acc += metrics['test_acc'][-1]
    test_acc = test_acc/16
    return test_acc
        