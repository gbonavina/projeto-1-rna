from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PATH, RANDOM_STATE, TEST_SIZE, VAL_FRACTION_OF_DEV


@dataclass
class DatasetSplits:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    x_scaler: StandardScaler
    y_scaler: StandardScaler

    def inverse_y(self, y_scaled: torch.Tensor):
        values = y_scaled.detach().cpu().numpy().reshape(-1, 1)
        return self.y_scaler.inverse_transform(values).ravel()


def load_xy(path: Path | str | None = None) -> tuple:
    csv_path = Path(path) if path is not None else DATA_PATH
    data = pd.read_csv(csv_path)
    return data["x"].to_numpy(), data["y"].to_numpy()


def _to_column_tensor(array) -> torch.Tensor:
    tensor = torch.tensor(array, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)
    return tensor


def _scale(train: torch.Tensor, *others: torch.Tensor):
    scaler = StandardScaler()
    train_scaled = torch.tensor(
        scaler.fit_transform(train.numpy()), dtype=torch.float32
    )
    rest = [
        torch.tensor(scaler.transform(other.numpy()), dtype=torch.float32)
        for other in others
    ]
    return (train_scaled, *rest, scaler)


def prepare_splits(
    path: Path | str | None = None,
    test_size: float = TEST_SIZE,
    val_fraction_of_dev: float = VAL_FRACTION_OF_DEV,
    random_state: int = RANDOM_STATE,
) -> DatasetSplits:
    X, y = load_xy(path)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=val_fraction_of_dev,
        random_state=random_state,
        shuffle=True,
    )

    X_train, X_val, X_test = (
        _to_column_tensor(X_train),
        _to_column_tensor(X_val),
        _to_column_tensor(X_test),
    )
    y_train, y_val, y_test = (
        _to_column_tensor(y_train),
        _to_column_tensor(y_val),
        _to_column_tensor(y_test),
    )

    X_train, X_val, X_test, x_scaler = _scale(X_train, X_val, X_test)
    y_train, y_val, y_test, y_scaler = _scale(y_train, y_val, y_test)

    return DatasetSplits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )
