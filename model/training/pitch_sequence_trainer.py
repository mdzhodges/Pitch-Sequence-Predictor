import random
import copy
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate

from model.custom_types.trainer_type import TrainerComponents
from utils.logger import Logger

import torch.nn.functional as F

from utils.constants import Constants

from evaluation.eval import PitchSequenceEvaluator
from sklearn.metrics import accuracy_score, f1_score
from model.class_weights import PitcherClassWeights, WeightConfig


class PitchSequenceTrainer:

    def __init__(self, model_params: TrainerComponents):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )

        self.num_epochs = model_params.num_epochs
        self.batch_size = model_params.batch_size

        self.dataset = model_params.dataset
        self.encoder = model_params.pitch_seq_encoder.to(self.device)

        self.logger = Logger(self.__class__.__name__)

        self.patience = getattr(model_params, "patience", 10)
        self.best_val_f1 = float("-inf")
        self.best_state = None
        self.epochs_without_improvement = 0

        self.weight_manager = PitcherClassWeights(
            WeightConfig(
                dataset=self.dataset,
                num_classes=len(Constants.PITCH_TYPE_TO_IDX),
                device=self.device
            )
        )

        # splits
        (
            self.train_loader,
            self.val_loader,
            self.test_loader
        ) = self.get_loaders(
            val_split=0.1,
            test_split=0.1
        )

        # Adam with weight decay
        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.encoder.learning_rate,
            weight_decay=1e-4
        )


    def train(self):
        best_epoch = 0
        for epoch in range(self.num_epochs):
            self.encoder.train()
            total_loss = 0.0
            num_batches = 0

            for batch in self.train_loader:
                labels, pitcher_id, pitch_seq, pitch_seq_mask = self._prepare_sequence_batch(batch)

                self.optimizer.zero_grad()

                out = self.encoder(
                    pitch_seq=pitch_seq,
                    pitcher_id=pitcher_id,
                    pitch_seq_mask=pitch_seq_mask,
                )

                logits = out["logits"]
                # Example usage: obtain per-pitcher class weights for the current
                # batch and compute the weighted loss.
                loss = self._compute_weighted_loss(
                    logits=logits,
                    labels=labels,
                    pitcher_ids=pitcher_id
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 5.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches


            self.logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} — "
                f"Epoch Loss (avg): {avg_loss:.4f} | "
                f"Total Loss Sum: {total_loss:.4f}"
            )

            val_metrics = self.evaluate_split(self.val_loader, split_name="val")
            val_f1 = val_metrics["f1_macro"]

            if val_f1 > self.best_val_f1 + 1e-4:
                self.best_val_f1 = val_f1
                self.best_state = copy.deepcopy(self.encoder.state_dict())
                self.epochs_without_improvement = 0
                best_epoch = epoch + 1
                self.logger.info(
                    f"New best validation F1 (macro): {val_f1:.4f}"
                )
            else:
                self.epochs_without_improvement += 1
                self.logger.info(
                    f"No improvement this epoch. Patience: "
                    f"{self.epochs_without_improvement}/{self.patience}"
                )
                if self.epochs_without_improvement >= self.patience:
                    self.logger.info("Early stopping triggered.")
                    break

        if self.best_state is not None:
            self.encoder.load_state_dict(self.best_state)
            self.logger.info(
                f"Loaded best checkpoint from epoch {best_epoch} "
                f"(Val F1 Macro: {self.best_val_f1:.4f})"
            )
        PitchSequenceEvaluator(self.encoder, test_dataset=self.test_loader).run()

    def evaluate_split(self, data_loader: DataLoader, split_name: str) -> Dict[str, float]:
        self.encoder.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                labels, pitcher_id, pitch_seq, pitch_seq_mask = self._prepare_sequence_batch(batch)

                out = self.encoder(
                    pitch_seq=pitch_seq,
                    pitcher_id=pitcher_id,
                    pitch_seq_mask=pitch_seq_mask,
                )

                logits = out["logits"]
                loss = self._compute_weighted_loss(
                    logits=logits,
                    labels=labels,
                    pitcher_ids=pitcher_id
                )

                total_loss += loss.item()
                num_batches += 1

                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        if not all_labels:
            raise ValueError(f"{split_name} loader is empty; cannot compute metrics.")

        avg_loss = total_loss / max(num_batches, 1)
        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()

        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        self.logger.info(
            f"{split_name.capitalize()} — Loss: {avg_loss:.4f} | "
            f"F1 Macro: {f1_macro:.4f} | F1 Micro: {f1_micro:.4f} | "
            f"Accuracy: {accuracy:.4f}"
        )

        return {
            "loss": avg_loss,
            "f1_macro": float(f1_macro),
            "f1_micro": float(f1_micro),
            "accuracy": float(accuracy),
        }

    def _prepare_sequence_batch(self, batch: Dict[str, Any]):
        labels = batch["label"].to(self.device)
        pitcher_id = batch["pitcher_id"].to(self.device)
        pitch_seq_numeric = batch["pitch_seq_numeric"].to(self.device)
        pitch_seq_categorical = {
            k: v.to(self.device)
            for k, v in batch["pitch_seq_categorical"].items()
        }
        pitch_seq_mask = batch["pitch_seq_mask"].to(self.device)

        pitch_seq = self.encoder.build_pitch_sequence_tensor(
            pitch_seq_numeric=pitch_seq_numeric,
            pitch_seq_categorical=pitch_seq_categorical
        )

        return labels, pitcher_id, pitch_seq, pitch_seq_mask

    def get_indices_for_split(self, val_split: float = .1, test_split: float = .1):
        """Return train, val, test index lists based on your split ratios."""
        total_length = len(self.dataset)
        indices = list(range(total_length))

        # shuffle
        random.shuffle(indices)

        # compute split points
        test_size = int(total_length * test_split)
        val_size = int(total_length * val_split)

        train_end = total_length - test_size - val_size
        val_end = total_length - test_size

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        return train_indices, val_indices, test_indices

    def get_loaders(self, val_split: float = .1, test_split: float = .1):
        # Get indices for the split
        train_idx, val_idx, test_idx = self.get_indices_for_split(
            val_split=val_split, test_split=test_split)

        # Normalize numeric features using only the training split to avoid leakage
        self.dataset.normalize_using_indices(train_idx)

        # take subset of the data
        train_subset = Subset(self.dataset, train_idx)
        val_subset = Subset(self.dataset, val_idx)
        test_subset = Subset(self.dataset, test_idx)

        # data loaders for test/val/train
        train_loader = DataLoader(
            dataset=train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_batch
        )

        val_loader = DataLoader(
            dataset=val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_batch
        )

        test_loader = DataLoader(
            dataset=test_subset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_batch
        )

        return train_loader, val_loader, test_loader

    def _compute_weighted_loss(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            pitcher_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy with per-pitcher class weights. Each pitcher ID
        maps to a normalized weight vector over pitch classes.
        """
        per_pitcher_weights = self.weight_manager.get_pitcher_class_weights(pitcher_ids)
        if per_pitcher_weights.dim() == 1:
            per_pitcher_weights = per_pitcher_weights.unsqueeze(0).expand_as(logits)

        log_probs = F.log_softmax(logits, dim=-1)
        nll = -torch.gather(log_probs, 1, labels.unsqueeze(1)).squeeze(1)

        class_weights = torch.gather(
            per_pitcher_weights, 1, labels.unsqueeze(1)
        ).squeeze(1)
        fallback = torch.full_like(
            class_weights, 1.0 / len(Constants.PITCH_TYPE_TO_IDX)
        )
        class_weights = torch.where(class_weights > 0, class_weights, fallback)
        weighted = nll * class_weights
        return weighted.mean()

    def _collate_batch(self, samples):
        batch_size = len(samples)
        seq_lengths = [s["pitch_seq_numeric"].size(0) for s in samples]
        max_len = max(seq_lengths)
        numeric_dim = samples[0]["pitch_seq_numeric"].size(1)

        pitch_seq_numeric = torch.zeros(batch_size, max_len, numeric_dim)
        cat_keys = list(samples[0]["pitch_seq_categorical"].keys())
        pitch_seq_categorical = {
            key: torch.zeros(batch_size, max_len, dtype=torch.long)
            for key in cat_keys
        }
        pitch_seq_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

        for i, sample in enumerate(samples):
            seq_len = sample["pitch_seq_numeric"].size(0)
            pitch_seq_numeric[i, :seq_len] = sample["pitch_seq_numeric"]
            pitch_seq_mask[i, :seq_len] = True

            for key in cat_keys:
                pitch_seq_categorical[key][i, :seq_len] = sample["pitch_seq_categorical"][key]

        batch = {
            "numeric": default_collate([s["numeric"] for s in samples]),
            "categorical": {
                key: default_collate([s["categorical"][key] for s in samples])
                for key in samples[0]["categorical"].keys()
            },
            "label": default_collate([s["label"] for s in samples]),
            "pitcher_id": default_collate([s["pitcher_id"] for s in samples]),
            "pitch_seq_numeric": pitch_seq_numeric,
            "pitch_seq_categorical": pitch_seq_categorical,
            "pitch_seq_mask": pitch_seq_mask,
        }

        return batch
