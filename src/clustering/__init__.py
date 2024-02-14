from src.clustering.sr_fca import SRFCA

from .cfl import CFL
from .fedavg import FedAvg
from .ifca import IFCA
from .oneshot_kmeans import OneShotKMeans
from .soft_ifca import SoftIFCA
from .oneshot_ifca import OneShotIFCA
from .feddrift import FedDrift
from .sr_fca_merge_refine import SRFCAMergeRefine
# from .mocha import MOCHA

# CLUSTERING_DICT = {"ifca": IFCA, "sr_fca": SRFCA, "cfl": CFL, "mocha": MOCHA}
CLUSTERING_DICT = {"sr_fca": SRFCA, 
                   "ifca": IFCA, 
                   "cfl": CFL,
                   "fedavg": FedAvg,
                   "oneshot_kmeans": OneShotKMeans,
                   "soft_ifca": SoftIFCA,
                   "oneshot_ifca" : OneShotIFCA,
                   "feddrift": FedDrift,
                   "sr_fca_merge_refine": SRFCAMergeRefine}
