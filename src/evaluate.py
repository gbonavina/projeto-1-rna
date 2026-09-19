import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def epochs_to_convergence(val_losses) -> int:
    return int(np.argmin(val_losses) + 1)


def evaluate_test(model, splits) -> dict:
    y_true = splits.inverse_y(splits.y_test)
    y_pred = splits.inverse_y(model.predict(splits.X_test))
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
    }


def comparison_table(experiments, splits) -> pd.DataFrame:
    rows = []
    for name, trained_model, val_losses in experiments:
        row = {"Modelo": name, **evaluate_test(trained_model, splits)}
        row["Épocas até convergência"] = epochs_to_convergence(val_losses)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Modelo")
