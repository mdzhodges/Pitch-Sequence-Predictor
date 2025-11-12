from dataclasses import dataclass


@dataclass
class PitchSequencePipelineComponents:
    num_epochs: int
    dropout_pitch_sequence: float
    learning_rate_pitch_sequence: float
    sample: int
    batch_size: int
