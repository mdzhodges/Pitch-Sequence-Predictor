from model.custom_types.trainer_type import TrainerComponents


class PitchSequenceTrainer:
    
    def __init__(self, model_params: TrainerComponents):

        # Various training needs
        self.num_epochs = model_params.num_epochs
        self.batch_size = model_params.batch_size
        
        self.dataset = model_params.dataset
        # encoder
        self.pitch_sequence_encoder = model_params.pitch_seq_encoder
    
    def train(self):
        pass
