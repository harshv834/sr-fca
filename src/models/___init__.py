from src.models.cnn import SimpleCNN
from src.models.linear import OneLayer, TwoLayer
from src.models.lstm import StackedLSTM
from src.models.resnet import ResNet9

MODEL_DICT = {
    "simplecnn": SimpleCNN,
    "one_layer_lin": OneLayer,
    "two_layer_lin": TwoLayer,
    "stacked_lstm": StackedLSTM,
    "resnet": ResNet9,
}
