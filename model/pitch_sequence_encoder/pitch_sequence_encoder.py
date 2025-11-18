import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

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
        self.categorical_cols = list(self.dataset.vocab_maps.keys())
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
        # 2. Sequence encoder modules
        # -----------------------------
        self.feature_dim = getattr(
            model_params, "feature_dim",
            getattr(self.dataset, "pitch_seq_feature_dim", self.input_dim)
        )
        self.max_seq_len = getattr(
            model_params, "max_seq_len",
            getattr(self.dataset, "pitch_seq_max_len", 12)
        )
        self.num_heads = getattr(
            model_params, "num_heads",
            getattr(self.dataset, "pitch_seq_num_heads", 4)
        )

        self.input_projection = nn.Linear(self.feature_dim, self.embed_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.max_seq_len, self.embed_dim)
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(self.embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        self.ff_norm = nn.LayerNorm(self.embed_dim)

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

    def build_pitch_sequence_tensor(
            self,
            pitch_seq_numeric: torch.Tensor,
            pitch_seq_categorical: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Combine numeric + categorical embeddings to transform historical pitch
        features into a dense tensor consumed by the attention stack.
        """
        seq_parts = [pitch_seq_numeric]

        for col in self.categorical_cols:
            if col not in pitch_seq_categorical:
                continue
            safe_name = self.safe_cat_names[col]
            emb = self.embeddings[safe_name](pitch_seq_categorical[col])
            seq_parts.append(emb)

        if len(seq_parts) == 1:
            return seq_parts[0]

        return torch.cat(seq_parts, dim=-1)

    def forward(self,
                pitch_seq: Optional[torch.Tensor] = None,
                numeric: Optional[torch.Tensor] = None,
                categorical: Optional[Dict[str, torch.Tensor]] = None,
                pitcher_id=None,
                pitch_seq_mask: Optional[torch.Tensor] = None):


        # -----------------------------
        # 1. Build pitch sequence tensor if needed
        # -----------------------------
        if pitch_seq is None:
            if numeric is None:
                raise ValueError(
                    "PitchSequenceEncoder requires either `pitch_seq` or "
                    "numeric/categorical inputs."
                )

            embedded = []
            if categorical:
                for col, x in categorical.items():
                    safe_name = self.safe_cat_names[col]
                    embedded.append(self.embeddings[safe_name](x))
            if embedded:
                cat_embed = torch.cat(embedded, dim=1)
                combined = torch.cat([numeric, cat_embed], dim=1)
            else:
                combined = numeric

            pitch_seq = combined.unsqueeze(1)

        if pitch_seq.size(-1) != self.feature_dim:
            raise ValueError(
                f"Expected pitch_seq feature dim {self.feature_dim}, "
                f"got {pitch_seq.size(-1)}"
            )

        # -----------------------------
        # 2. Multi-head self-attention encoder
        # -----------------------------
        if pitch_seq_mask is not None:
            pitch_seq_mask = pitch_seq_mask.to(pitch_seq.device)
            pitch_seq = pitch_seq * pitch_seq_mask.unsqueeze(-1)

        x = self.input_projection(pitch_seq)

        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        x = x + self.pos_encoding[:, :seq_len, :]

        key_padding_mask = None
        if pitch_seq_mask is not None:
            key_padding_mask = ~pitch_seq_mask.bool()

        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.attn_norm(x + attn_out)

        ff_out = self.ff(x)
        x = self.ff_norm(x + ff_out)

        if pitch_seq_mask is not None:
            mask = pitch_seq_mask.unsqueeze(-1).type_as(x)
            denom = mask.sum(dim=1).clamp_min(1e-6)
            pooled_state = (x * mask).sum(dim=1) / denom
        else:
            pooled_state = x.mean(dim=1)

        # -----------------------------
        # 4. Pitcher-specific masking
        # -----------------------------
        # raw logits
        logits = self.output_head(pooled_state)

        # Default: no masking
        masked_logits = logits

        # Apply repertoire mask whenever a pitcher id is provided
        if pitcher_id is not None:
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
            "pred": pred,
            "pooled_state": pooled_state
        }
