import torch.nn as nn
import torch
from typing import Dict
from utils.logger import Logger
from model.custom_types.data_types import ModelComponents
import torch.nn.functional as F


class PitchSequenceEncoder(nn.Module):

    def __init__(self, model_params: ModelComponents):
        super().__init__()

        self.logger = Logger(self.__class__.__name__)
        self.dataset = model_params.dataset
        self.learning_rate = model_params.learning_rate
        self.dropout = model_params.dropout

        self.numeric_dim = len(self.dataset.numeric_cols)

        # categorical → one-hot sizes
        self.categorical_dims = {
            col: len(self.dataset.vocab_maps[col])
            for col in self.dataset.categorical_cols
        }

        self.total_cat_dim = sum(self.categorical_dims.values())
        self.input_dim = self.numeric_dim + self.total_cat_dim

        self.hidden_dim = model_params.hidden_dim   # recommended 512
        self.embed_dim = model_params.embed_dim     # recommended 256
        self.num_classes = int(self.dataset.dataset_summary["num_classes"])

        # 1. Input projection block (input_dim → embed_dim)
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

        # 2. Output classification block (embed_dim → num_classes)
        self.output_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.embed_dim // 2, self.num_classes)
        )

    def forward(self, numeric: torch.Tensor, categorical: Dict[str, torch.Tensor], pitcher_id):

        # one-hot encode categoricals
        cat_one_hots = []
        for col, x in categorical.items():
            one_hot = F.one_hot(
                x, num_classes=self.categorical_dims[col]
            ).float()
            cat_one_hots.append(one_hot)

        # this is the one-hot tensor
        cat_tensor = torch.cat(cat_one_hots, dim=1)

        # combine numeric and categorical
        combined = torch.cat([numeric, cat_tensor], dim=1)

        # 1. embed
        embed = self.input_project(combined)

        # 2. logits (before masking)
        logits = self.output_head(embed)

        # 3. pitcher-specific masking
        if pitcher_id is not None:
            batch_size = pitcher_id.size(0)
            masked_logits = torch.full_like(logits, float('-1e9'))

            for i in range(batch_size):
                pid = int(pitcher_id[i].item())
                allowed = self.dataset.pitcher_to_allowed.get(pid, [])

                if len(allowed) > 0:
                    masked_logits[i, allowed] = logits[i, allowed]
                else:
                    masked_logits[i] = logits[i]
        else:
            masked_logits = logits

        # softmax only for inference
        probs = F.softmax(masked_logits, dim=-1)
        pred = probs.argmax(dim=1)

        return {
            "logits": masked_logits,
            "probs": probs,
            "pred": pred
        }
