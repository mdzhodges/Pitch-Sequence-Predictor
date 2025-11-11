import pandas as pd
import torch
from torch.utils.data import Dataset

class BatterDataset(Dataset):
    """
    Loads the full Fangraphs batter dataset and converts all numeric stats
    (excluding ID, Name, Team, Age, Season, G) into normalized PyTorch tensors.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        df = pd.read_csv(csv_path)

        # Columns to exclude
        exclude = {"IDfg", "Season", "Name", "Team", "Age", "G"}
        numeric_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

        if not numeric_cols:
            raise ValueError("No numeric columns found after filtering non-feature columns.")

        self.feature_cols = numeric_cols
        self.names = df["Name"].values if "Name" in df.columns else None

        # Convert to float32 and normalize (z-score)
        features = df[self.feature_cols].astype(float)
        self.means = features.mean()
        self.stds = features.std().replace(0, 1)
        features = (features - self.means) / self.stds

        # Store as tensor
        self.X = torch.tensor(features.values, dtype=torch.float32)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

    def get_by_name(self, name: str):
        """Return a single player's normalized tensor by name (case-insensitive)."""
        if self.names is None:
            raise ValueError("No Name column available.")
        mask = [name.lower() in n.lower() for n in self.names]
        if idx := [i for i, m in enumerate(mask) if m]:
            return self.X[idx[0]]
        else:
            raise ValueError(f"No player found matching '{name}'")

    def to_tensor(self):
        """Return all batter stats as a single (N, D) tensor."""
        return self.X
 
 
dataset = BatterDataset("data/batters_2025_full.csv")

# Access tensor for one player
jose_tensor = dataset.get_by_name("Jose Ramirez")
