import argparse

from src.config import NUM_EPOCHS, REPORT_FIGURES_DIR
from src.data import prepare_splits
from src.evaluate import comparison_table
from src.experiments import train_all
from src.plots import save_report_figures


def parse_args():
    parser = argparse.ArgumentParser(description="Treina a ablação de MLPs do Projeto 1.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-save", action="store_true", help="Não grava checkpoints em models/.")
    return parser.parse_args()


def main():
    args = parse_args()
    splits = prepare_splits()
    results = train_all(
        splits,
        num_epochs=args.epochs,
        verbose=args.verbose,
        save=not args.no_save,
    )

    experiments = [
        (item["name"], item["model"], item["val_losses"]) for item in results
    ]
    table = comparison_table(experiments, splits)
    print(table.round(4))

    fig_dir = save_report_figures(splits, results, show=False)
    print("Figuras salvas em", fig_dir.resolve())
    print("Relatório espera os PNGs em", REPORT_FIGURES_DIR)


if __name__ == "__main__":
    main()
