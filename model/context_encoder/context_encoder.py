from model.data_types import ModelComponents
from preprocessing.context_dataset import ContextDataset



class ContextEncoder:
    def __init__(self, model_params: ModelComponents[ContextDataset]):
        
        self.dataset = model_params.dataset
        self.learning_rate = model_params.learning_rate
        self.dropout = model_params.dropout
        
        
    def forward(self):
        pass