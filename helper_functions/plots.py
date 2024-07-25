import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_1DToy_plot(bnn_model, X, y, fun, dir, x_scaler=None, y_scaler=None):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))

    min1, max1 = [-6, 6]
    min2, max2 = [-99, 99]

    x_grid = np.arange(min1 - 0.1, max1 + 0.1, 0.1)
    if x_scaler is not None:
        x_grid = x_scaler.transform(x_grid.reshape(-1, 1))

    x_grid = x_grid.reshape(-1, 1)
    y_pred, y_cov = bnn_model.predict(torch.from_numpy(x_grid).float())
    if x_scaler is not None:
        x_grid = x_scaler.inverse_transform(x_grid)
        X = x_scaler.inverse_transform(X)

    if y_scaler is not None:
        y = y * y_scaler[1] + y_scaler[0]
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_xlim([min1, max1])
    ax.set_ylim([min2, max2])

    l = r"$y = x^3$"

    ax.plot(x_grid.squeeze(), fun(x_grid.squeeze()), color='black', alpha=0.8, label=l, linewidth=3.0)
    ax.scatter(X, y, color='red', s=8, alpha=0.7, label="Ground Truth")
    ax.plot(x_grid, y_pred.squeeze(), color='red', label="TAGI", linewidth=3.0)

    int1 = y_pred.squeeze() - 2 * np.sqrt(y_cov).squeeze()
    int2 = y_pred.squeeze() + 2 * np.sqrt(y_cov).squeeze()
    ax.fill_between(x_grid.squeeze(), int1, int2, color='red', alpha=0.2, label=r'$\pm 2 \sigma $ Confidence')

    plt.savefig(os.path.join(dir, "regression.pdf"), dpi=300, bbox_inches="tight")

