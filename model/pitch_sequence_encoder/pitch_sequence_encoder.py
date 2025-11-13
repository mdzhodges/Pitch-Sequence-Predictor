import torch.nn as nn
import torch
from model.custom_types.data_types import ModelComponents
from typing import Dict


class PitchSequenceEncoder(nn.Module):

    def __init__(self, model_params: ModelComponents):
        super().__init__()

        self.dataset = model_params.dataset
        self.learning_rate = model_params.learning_rate
        self.dropout = model_params.dropout


    def forward(self, numeric: torch.Tensor, categorical: Dict[str, torch.Tensor]):
        output = {"loss" : 1}
        
        
        return output
