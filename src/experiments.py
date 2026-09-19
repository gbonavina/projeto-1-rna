import torch

from src.config import (
    BATCH_SIZE,
    DROPOUT,
    L1_LAMBDA,
    L2_WEIGHT_DECAY,
    LEARNING_RATE,
    MODELS_DIR,
    MOMENTUM,
    NEURONS,
    NUM_EPOCHS,
    NUM_LAYERS,
)
from src.model import MLP, MLP_DROPOUT, MLP_L1, MLP_L2, MLP_MOMENTUM


def experiment_specs():
    return [
        {
            "name": "Baseline",
            "label": "Baseline (SGD vanilla)",
            "model": MLP(NUM_LAYERS, NEURONS),
            "fit_kwargs": {},
        },
        {
            "name": "Dropout",
            "label": rf"Dropout ($p={DROPOUT}$)",
            "model": MLP_DROPOUT(NUM_LAYERS, NEURONS, dropout=DROPOUT),
            "fit_kwargs": {},
        },
        {
            "name": "L2",
            "label": r"L2 ($\lambda=10^{-3}$)",
            "model": MLP_L2(NUM_LAYERS, NEURONS),
            "fit_kwargs": {"weight_decay": L2_WEIGHT_DECAY},
        },
        {
            "name": "L1",
            "label": r"L1 ($\lambda=10^{-4}$)",
            "model": MLP_L1(NUM_LAYERS, NEURONS),
            "fit_kwargs": {"l1": L1_LAMBDA},
        },
        {
            "name": "Momentum",
            "label": r"Momentum ($\beta=0.9$)",
            "model": MLP_MOMENTUM(NUM_LAYERS, NEURONS),
            "fit_kwargs": {"momentum": MOMENTUM},
        },
    ]


def train_all(splits, num_epochs=NUM_EPOCHS, verbose=False, save=True):
    common = dict(
        num_epochs=num_epochs,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        verbose=verbose,
    )
    results = []
    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for spec in experiment_specs():
        model = spec["model"]
        train_losses, val_losses = model.fit(
            splits.X_train,
            splits.y_train,
            splits.X_val,
            splits.y_val,
            **common,
            **spec["fit_kwargs"],
        )
        result = {
            "name": spec["name"],
            "label": spec["label"],
            "model": model,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
        results.append(result)

        if save:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "fit_kwargs": spec["fit_kwargs"],
                },
                MODELS_DIR / f"{spec['name'].lower()}.pt",
            )

    return results
