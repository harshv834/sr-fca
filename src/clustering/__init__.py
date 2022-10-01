from src.clustering.sr_fca import SRFCA

from .cfl import CFL
from .fedavg import FedAvg
from .ifca import IFCA

# from .mocha import MOCHA

# CLUSTERING_DICT = {"ifca": IFCA, "sr_fca": SRFCA, "cfl": CFL, "mocha": MOCHA}
CLUSTERING_DICT = {"sr_fca": SRFCA, "ifca": IFCA, "cfl": CFL, "fedavg": FedAvg}
