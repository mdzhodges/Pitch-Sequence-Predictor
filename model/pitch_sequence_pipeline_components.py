from dataclasses import dataclass


@dataclass
class PitchSequencePipelineComponents:
    num_epochs: int
    dropout_hitter: float
    dropout_pitcher: float
    dropout_context: float
    dropout_pitch_sequence: float
    learning_rate_hitter: float
    learning_rate_pitcher: float
    learning_rate_context: float
    learning_rate_pitch_sequence: float
    sample: int
