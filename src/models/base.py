from .linear import OneLayer, TwoLayer
from .resnet import ResNet9
from .cnn import SimpleCNN
from .lstm import 

MODEL_DICT = {
    "one_layer_lin": OneLayer,
    "two_layer_lin": TwoLayer,
    "resnet": ResNet9,
    "simplecnn": SimpleCNN,
}

