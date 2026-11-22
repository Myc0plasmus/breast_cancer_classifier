import pandas as pd
import torch

from torch.utils.data import Dataset


class TensorDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str = "target",
    ) -> None:
        self.features = torch.tensor(
            df.drop(columns=target_column).to_numpy(),
            dtype=torch.float32,
        )

        self.targets = torch.tensor(
            df[target_column].to_numpy(),
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]
