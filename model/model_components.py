
from dataclasses import dataclass



@dataclass
class ModelComponents:
    """Represents the components and hyperparameters for a pitch sequence prediction model.
    
    This class stores encoders and training parameters used to configure and train the model.
    
    Attributes:
        hitter_encoder: Encoder for hitter features.
        pitcher_encoder: Encoder for pitcher features.
        context_encoder: Encoder for context features.
        pitch_sequence_encoder: Encoder for pitch sequence features.
        num_epochs: Number of training epochs.
        dropout_hitter: Dropout rate for hitter encoder.
        dropout_pitcher: Dropout rate for pitcher encoder.
        dropout_context: Dropout rate for context encoder.
        dropout_pitch_sequence: Dropout rate for pitch sequence encoder.
        learning_rate_hitter: Learning rate for hitter encoder.
        learning_rate_pitcher: Learning rate for pitcher encoder.
        learning_rate_context: Learning rate for context encoder.
        learning_rate_pitch_sequence: Learning rate for pitch sequence encoder.
        sample: Optional sample identifier.
    """
    hitter_encoder: object
    pitcher_encoder: object
    context_encoder: object
    pitch_sequence_encoder: object
    num_epochs: int
    dropout_hitter: float
    dropout_pitcher: float
    dropout_context: float
    dropout_pitch_sequence: float
    learning_rate_hitter: float
    learning_rate_pitcher: float
    learning_rate_context: float
    learning_rate_pitch_sequence: float
    sample: int  # optional