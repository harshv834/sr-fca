from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.server.strategy import Strategy

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


class SR_FCA(Strategy):
    def __init__(
        self,
        *,
        dist_metric: Callable[[ClientProxy, ClientProxy], float],
        initial_parameters: Optional[Parameters] = None,
        thresh: float = None
    ):
        super.__init__()
        self.initial_parameters = initial_parameters
        self.dist_func = lambda x, y: dist_metric(x, y)
        self.thresh = thresh

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        initial_parameters = self.initial_parameters
        self.initial_parameters = None
        return initial_parameters
