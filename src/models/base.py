from .cnn import SimpleCNN
from .linear import OneLayer, TwoLayer
from .lstm import StackedLSTM
from .resnet import ResNet9

MODEL_DICT = {
    "one_layer_lin": OneLayer,
    "two_layer_lin": TwoLayer,
    "resnet": ResNet9,
    "simplecnn": SimpleCNN,
    "stacked_lstm": StackedLSTM,
}