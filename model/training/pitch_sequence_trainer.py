
from model.hitter_encoder.hitter_encoder import HitterEncoder


class PitchSequenceTrainer:
    
    def __init__(self, hitter_encoder: HitterEncoder):
        self.hitter_encoder = hitter_encoder
        
    def train(self):
        pass