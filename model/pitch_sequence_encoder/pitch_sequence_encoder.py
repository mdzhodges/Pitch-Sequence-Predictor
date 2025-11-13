import torch.nn as nn
from model.custom_types.data_types import ModelComponents


class PitchSequenceEncoder(nn.Module):

    def __init__(self, model_params: ModelComponents):
        super().__init__()

        self.dataset = model_params.dataset
        self.learning_rate = model_params.learning_rate
        self.dropout = model_params.dropout

    def forward(self):
        pass
