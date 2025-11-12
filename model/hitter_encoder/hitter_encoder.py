from preprocessing.hitter_dataset import HitterDataset
from model.data_types import ModelComponents
from utils.logger import Logger
import torch
import torch.functional as F
import torch.nn as nn

class HitterEncoder:
    
    def __init__(self, model_params: ModelComponents[HitterDataset]):  
        
        # Encoder fields      
        self.dataset =  model_params.dataset
        self.learning_rate = model_params.learning_rate
        self.dropout = model_params.dropout
        self.hidden_dim = model_params.hidden_dim
        self.embed_dim = model_params.embed_dim
        self.logger = Logger(self.__class__.__name__)     
        self.num_features = len(self.dataset.feature_columns)
        
        
        # Input Neural Network
        self.input_projection = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.hidden_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        # Output dim (for fusion)
        self.output_dim = self.embed_dim
        
                        
    def forward(self, val: torch.Tensor):
        return self.input_projection(val)