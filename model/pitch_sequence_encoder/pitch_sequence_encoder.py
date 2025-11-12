from model.data_types import ModelComponents
from preprocessing.pitch_sequence_dataset import PitchSequenceDataset
class PitchSequenceEncoder:
    
    def __init__(self, model_params: ModelComponents[PitchSequenceDataset]):
        
        self.dataset = model_params.dataset
        self.learning_rate = model_params.learning_rate
        self.dropout = model_params.dropout
        
        
    def forward(self):
        pass