import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from utils.logger import Logger
from model.custom_types.data_types import ModelComponents
from utils.constants import Constants


class PitchSequenceEncoder(nn.Module):
    def __init__(self, model_params: ModelComponents):
        super().__init__()

        self.logger = Logger(self.__class__.__name__)
        self.dataset = model_params.dataset

        self.dropout = model_params.dropout
        self.learning_rate = model_params.learning_rate
        self.hidden_dim = model_params.hidden_dim
        self.embed_dim = model_params.embed_dim

        # Numeric dimension
        self.numeric_dim = len(self.dataset.numeric_cols)

        # One-hot dimension for categoricals
        self.categorical_dims = {
            col: len(self.dataset.vocab_maps[col])
            for col in self.dataset.categorical_cols
        }

        self.total_cat_dim = sum(self.categorical_dims.values())
        self.input_dim = self.numeric_dim + self.total_cat_dim

        self.num_classes = len(Constants.PITCH_TYPE_TO_IDX)

        # ------------------------------------------------------------
        # Input → Hidden → Embedding
        # ------------------------------------------------------------
        self.input_project = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(self.hidden_dim),

            nn.Linear(self.hidden_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(self.embed_dim)
        )

        # ------------------------------------------------------------
        # Classification Head
        # ------------------------------------------------------------
        self.output_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(self.embed_dim // 2),
            nn.Linear(self.embed_dim // 2, self.num_classes),
        )

    def forward(self, numeric: torch.Tensor,
                categorical: Dict[str, torch.Tensor],
                pitcher_id):

        # One-hot encode categoricals
        one_hots = [
            F.one_hot(x, num_classes=self.categorical_dims[col]).float()
            for col, x in categorical.items()
        ]

        cat_tensor = torch.cat(one_hots, dim=1)

        # Combine numeric + categorical
        combined = torch.cat([numeric, cat_tensor], dim=1)

        # Embedding
        embed = self.input_project(combined)

        # Raw logits (before masking)
        logits = self.output_head(embed)

        # ------------------------------------------------------------
        # Apply pitcher-specific masking
        # ------------------------------------------------------------
        masked_logits = logits.clone()
        batch_size = logits.size(0)

        for i in range(batch_size):
            pid = int(pitcher_id[i].item())
            allowed = self.dataset.pitcher_to_allowed.get(pid)

            # Skip masking if missing data
            if allowed is None or len(allowed) == 0:
                continue

            # Construct full mask
            mask = torch.full(
                (self.num_classes,), float("-1e9"),
                device=logits.device
            )
            mask[allowed] = 0

            masked_logits[i] = logits[i] + mask

        # Softmax AFTER masking
        probs = F.softmax(masked_logits, dim=-1)
        pred = probs.argmax(dim=1)

        return {
            "logits": masked_logits,
            "probs": probs,
            "pred": pred
        }
