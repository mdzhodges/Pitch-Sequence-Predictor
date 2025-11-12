from model.custom_types.data_types import ModelComponents
class PitchSequenceEncoder:
    
    def __init__(self, model_params: ModelComponents):
        
        self.dataset = model_params.dataset
        self.learning_rate = model_params.learning_rate
        self.dropout = model_params.dropout
        
        
    def forward(self):
        pass