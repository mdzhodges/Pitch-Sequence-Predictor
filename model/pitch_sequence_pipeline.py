from utils.logger import Logger
from preprocessing.hitter_dataset import HitterDataset




class PitchSequencePipeline:
    
    def __init__(self):
        self.hitter_dataset = HitterDataset("data/batters_2025_full.csv")
        