import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, num_layers, neurons=None, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers

        if neurons is None:
            neurons = []

        n_linear = len(neurons) - 1
        layers = []
        for i in range(n_linear):
            layers.append(nn.Linear(neurons[i], neurons[i + 1]))
            if i < n_linear - 1:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def l1_penalty(self):
        return sum(p.abs().sum() for name, p in self.named_parameters() if p.dim() > 1)

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        learning_rate=0.001,
        num_epochs=100,
        batch_size=8,
        weight_decay=0.0,
        l1=0.0,
        momentum=0.0,
        verbose=True,
    ):
        optimizer = optim.SGD(
            self.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        criterion = nn.MSELoss()
        loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )

        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            super().train(True)
            epoch_loss = 0.0
            n_samples = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                outputs = self(xb)
                mse = criterion(outputs, yb)
                loss = mse + l1 * self.l1_penalty() if l1 else mse
                loss.backward()
                optimizer.step()
                epoch_loss += mse.item() * xb.size(0)
                n_samples += xb.size(0)
            train_losses.append(epoch_loss / n_samples)

            self.eval()
            with torch.no_grad():
                val_losses.append(criterion(self(X_val), y_val).item())

            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_losses[-1]:.4f}")

        return train_losses, val_losses

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self(X)


class MLP_DROPOUT(MLP):
    def __init__(self, num_layers, neurons=None, dropout=0.1):
        super().__init__(num_layers, neurons, dropout=dropout)


class MLP_L2(MLP):
    def fit(self, *args, weight_decay=1e-3, **kwargs):
        return super().fit(*args, weight_decay=weight_decay, **kwargs)


class MLP_L1(MLP):
    def fit(self, *args, l1=1e-3, **kwargs):
        return super().fit(*args, l1=l1, **kwargs)


class MLP_MOMENTUM(MLP):
    def fit(self, *args, momentum=0.99, **kwargs):
        return super().fit(*args, momentum=momentum, **kwargs)
