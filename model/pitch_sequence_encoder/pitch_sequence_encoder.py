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

        # numeric dims
        self.numeric_dim = self.dataset.x_numeric.shape[1]

        # categorical vocab sizes
        self.categorical_dims = {
            col: len(self.dataset.vocab_maps[col])
            for col in self.dataset.vocab_maps
        }

        # -----------------------------
        # 1. Build embeddings per categorical
        # -----------------------------
        
        # remap categorical names → safe pytorch module names
        self.safe_cat_names = {col: f"cat_{col}" for col in self.categorical_dims}


        self.embeddings = nn.ModuleDict()

        for col, vocab_size in self.categorical_dims.items():
            safe_name = self.safe_cat_names[col]    # e.g., "cat_type"

            emb_dim = min(8, max(4, int(vocab_size**0.5 + 2)))
            self.embeddings[safe_name] = nn.Embedding(vocab_size, emb_dim)



        self.total_cat_embed_dim = sum(
            emb.embedding_dim for emb in self.embeddings.values()
        )


        # Combined input size
        self.input_dim = self.numeric_dim + self.total_cat_embed_dim
        self.num_classes = len(Constants.PITCH_TYPE_TO_IDX)

        # -----------------------------
        # 2. Input → Hidden → Embedding layers
        # -----------------------------
        self.input_project = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(self.hidden_dim),

            nn.Linear(self.hidden_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(self.embed_dim),
        )

        # -----------------------------
        # 3. Classification head
        # -----------------------------
        self.output_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(self.embed_dim // 2),

            nn.Linear(self.embed_dim // 2, self.num_classes)
        )

    def forward(self, numeric: torch.Tensor,
                categorical: Dict[str, torch.Tensor],
                pitcher_id):


        # -----------------------------
        # 1. Embed categorical features
        # -----------------------------
        embedded = []
        for col, x in categorical.items():
            safe_name = self.safe_cat_names[col]
            embedded.append(self.embeddings[safe_name](x))

        cat_embed = torch.cat(embedded, dim=1)


        # -----------------------------
        # 2. Combine numeric + embedded categorical
        # -----------------------------
        combined = torch.cat([numeric, cat_embed], dim=1)

        # -----------------------------
        # 3. Main projection
        # -----------------------------
        embed = self.input_project(combined)

        # -----------------------------
        # 4. Pitcher-specific masking
        # -----------------------------
        # raw logits
        logits = self.output_head(embed)

        # Default: no masking
        masked_logits = logits

        # Only apply hard mask at inference / eval
        if (pitcher_id is not None) and (not self.training):
            masked_logits = logits.clone()
            batch_size = logits.size(0)

            for i in range(batch_size):
                pid = int(pitcher_id[i].item())
                allowed = self.dataset.pitcher_to_allowed.get(pid, None)

                if not allowed:
                    continue  # no repertoire info → leave row unmasked

                mask = torch.full(
                    (self.num_classes,),
                    float("-1e9"),
                    device=logits.device,
                )
                mask[allowed] = 0
                masked_logits[i] = logits[i] + mask

        # -----------------------------
        # 5. Pred + probs
        # -----------------------------
        probs = F.softmax(masked_logits, dim=-1)
        pred = probs.argmax(dim=1)

        return {
            "logits": masked_logits,
            "probs": probs,
            "pred": pred
        }
