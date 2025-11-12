from preprocessing.hitter_dataset import HitterDataset
from model.data_types import ModelComponents
from utils.logger import Logger
import torch
import torch.functional as F
import torch.nn as nn

class HitterEncoder:
    
    def __init__(self, model_params: ModelComponents[HitterDataset]):        
        self.dataset =  model_params.dataset
        self.learning_rate = model_params.learning_rate
        self.dropout = model_params.dropout
        self.logger = Logger(self.__class__.__name__)     
        
        self.num_features = self.hidden_dim = len(self.dataset.feature_cols)
        self.logger.info(str(len(self.dataset.feature_cols)))
        
                
    def forward(self):
        pass