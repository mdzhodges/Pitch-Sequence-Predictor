"""Evaluation utilities for the pitch sequence model."""
from __future__ import annotations

from typing import Any, Dict
import inspect

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from model.pitch_sequence_encoder.pitch_sequence_encoder import PitchSequenceEncoder
from preprocessing.fusion_dataset import FusionDataset


class PitchSequenceEvaluator:
    """Runs inference on a held-out set and reports F1 metrics."""

    def __init__(self, model: PitchSequenceEncoder, test_dataset: DataLoader):
        self.model = model
        self.test_dataset = test_dataset
        self.batch_size = 64

        self.device = next(self.model.parameters()).device
        self._forward_params = inspect.signature(self.model.forward).parameters

    def _move_batch_to_device(self, batch: Dict[str, Any]):
        numeric = batch["numeric"].to(self.device)
        labels = batch["label"].to(self.device)
        pitcher = batch["pitcher_id"].to(self.device)
        categorical = {key: tensor.to(self.device) for key, tensor in batch["categorical"].items()}
        return numeric, categorical, pitcher, labels

    def _forward_model(self, numeric: torch.Tensor, categorical: Dict[str, torch.Tensor], pitcher) -> Dict[str, torch.Tensor]:
        model_kwargs: Dict[str, Any] = {
            "numeric": numeric,
            "categorical": categorical,
            "pitcher_id": pitcher,
        }

        return self.model(**model_kwargs)

    def run(self) -> Dict[str, float]:

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.test_dataset:
                numeric, categorical, pitcher, labels = self._move_batch_to_device(batch)

                model_out = self._forward_model(numeric=numeric, categorical=categorical, pitcher=pitcher)
                logits = model_out["logits"]
                preds = torch.argmax(logits, dim=1)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        if not all_labels:
            raise ValueError("Evaluation dataset is empty; cannot compute metrics.")

        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()

        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"F1 Micro: {f1_micro:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        return {
            "f1_macro": float(f1_macro),
            "f1_micro": float(f1_micro),
            "accuracy": float(accuracy),
        }
