from model.data_types import ModelComponents
from preprocessing.pitcher_dataset import PitcherDataset



class PitcherEncoder:
    def __init__(self, model_params: ModelComponents[PitcherDataset]):
        
        self.dataset = model_params.dataset
        self.learning_rate = model_params.learning_rate
        self.dropout = model_params.dropout
        
        
    def forward(self):
        pass