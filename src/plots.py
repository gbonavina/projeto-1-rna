from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import REPORT_FIGURES_DIR


def plot_loss(
    train_losses,
    val_losses,
    xlabel="Epoch",
    ylabel="MSE",
    title=None,
    show=True,
):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Validation Loss")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_train_predictions(X, y, preds, show=True):
    fig, ax = plt.subplots()
    ax.scatter(X, y, label="Train")
    ax.scatter(X, preds, label="Train Predictions")
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def save_report_figures(splits, results, fig_dir: Path | None = None, show: bool = False):
    fig_dir = Path(fig_dir) if fig_dir is not None else REPORT_FIGURES_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    by_name = {item["name"]: item for item in results}
    baseline = by_name["Baseline"]
    l2 = by_name["L2"]

    loss_curves = [
        (item["label"], item["train_losses"], item["val_losses"]) for item in results
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(baseline["train_losses"], label="Treino", lw=1.0)
    ax.plot(baseline["val_losses"], label="Validação", lw=1.0)
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE (y padronizado)")
    ax.set_title("Convergência da MLP Baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "loss_baseline.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), sharex=True)
    axes = axes.ravel()
    for ax, (name, tr, va) in zip(axes, loss_curves):
        ax.plot(tr, label="Treino", lw=0.8)
        ax.plot(va, label="Validação", lw=0.8)
        ax.set_title(name)
        ax.set_xlabel("Época")
        ax.set_ylabel("MSE")
        ax.legend(fontsize=8)
    axes[5].axis("off")
    fig.suptitle("Dinâmica de treino e validação (MSE padronizado)")
    fig.tight_layout()
    fig.savefig(fig_dir / "loss_ablation.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    k = 51
    ker = np.ones(k) / k
    for name, tr, _ in loss_curves:
        ys = np.convolve(tr, ker, mode="valid")
        xs = np.arange(k // 2, k // 2 + len(ys))
        ax.plot(xs, ys, label=name.split(" (")[0], lw=1.6)
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE de treino (y padronizado, média móvel 51)")
    ax.set_title("Comparação da convergência no treino")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "loss_train_compare.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)

    y_true = splits.inverse_y(splits.y_test)
    pred_base = splits.inverse_y(baseline["model"].predict(splits.X_test))
    pred_l2 = splits.inverse_y(l2["model"].predict(splits.X_test))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharex=True, sharey=True)
    lims = [
        min(y_true.min(), pred_base.min(), pred_l2.min()),
        max(y_true.max(), pred_base.max(), pred_l2.max()),
    ]
    for ax, pred, title in [
        (axes[0], pred_base, "Baseline vanilla"),
        (axes[1], pred_l2, r"L2 ($\lambda=10^{-3}$), melhor modelo"),
    ]:
        ax.scatter(y_true, pred, s=18, alpha=0.65, edgecolors="none")
        ax.plot(lims, lims, color="black", lw=1, ls="--", label=r"$y=\hat{y}$")
        ax.set_title(title)
        ax.set_xlabel(r"$y$ real (teste)")
        ax.set_ylabel(r"$y$ predito")
        ax.legend(frameon=False)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Parity plot no conjunto de teste (escala original de $y$)")
    fig.tight_layout()
    fig.savefig(fig_dir / "parity.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)

    resid = y_true - pred_l2
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    axes[0].scatter(pred_l2, resid, s=18, alpha=0.65, edgecolors="none")
    axes[0].axhline(0, color="black", lw=1, ls="--")
    axes[0].set_xlabel(r"$\hat{y}$ (teste)")
    axes[0].set_ylabel(r"resíduo $y-\hat{y}$")
    axes[0].set_title("Resíduos vs. predito (L2)")
    axes[1].hist(resid, bins=25, color="steelblue", edgecolor="white")
    axes[1].set_xlabel(r"resíduo $y-\hat{y}$")
    axes[1].set_ylabel("frequência")
    axes[1].set_title("Distribuição dos resíduos (L2)")
    fig.tight_layout()
    fig.savefig(fig_dir / "residuals.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)

    return fig_dir
