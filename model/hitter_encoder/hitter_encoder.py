from preprocessing.hitter_dataset import HitterDataset

class HitterEncoder:
    
    def __init__(self, dataset: HitterDataset):
        self.dataset = dataset