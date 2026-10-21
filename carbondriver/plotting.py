import warnings
from typing import Optional

import pandas as pd
import torch


def plot_2d_debug(
    opt,
    predictor,
    AF_q,
    raw_bounds: torch.Tensor,
    candidate: pd.Series,
    possible_data: Optional[pd.DataFrame] = None,
    n: int = 10,
) -> None:
    """
    Debug-only visualization for 2D input spaces: pcolormesh of the acquisition
    function, posterior mean and posterior std over the input grid, overlaid with
    observed data (opt.df), candidate points (possible_data) and the chosen
    next experiment.
    """
    import matplotlib.pyplot as plt

    target_idx = opt.output_labels.index(opt.quantity)
    means = torch.tensor(opt._means[opt.input_labels].to_numpy())
    stds = torch.tensor(opt._stds[opt.input_labels].to_numpy())

    x0 = torch.linspace(raw_bounds[0, 0], raw_bounds[1, 0], n)
    x1 = torch.linspace(raw_bounds[0, 1], raw_bounds[1, 1], n)
    xx, yy = torch.meshgrid(x0, x1, indexing="ij")
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    grid_norm = ((grid - means) / stds).float().unsqueeze(1)
    grid_norm.requires_grad_(True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Output shape checks failed!")
        af = AF_q(grid_norm)
    af.sum().backward()
    grad = grid_norm.grad.squeeze(1).reshape(n, n, -1)
    af = af.detach().reshape(n, n)

    posterior = predictor.posterior(grid_norm.detach())
    mean = posterior.mean.view(n**2, -1)[:, target_idx].detach().reshape(n, n)
    std = (
        posterior.variance.sqrt()
        .view(n**2, -1)[:, target_idx]
        .detach()
        .reshape(n, n)
    )

    fig, ax = plt.subplots(ncols=3, figsize=(15, 4))
    for a, z, title in zip(ax, [af, mean, std], ["Acquisition", "Mean", "Std"]):
        pc = a.pcolormesh(xx, yy, z, shading="auto")
        fig.colorbar(pc, ax=a)
        a.contour(xx, yy, z, levels=n, colors="k", linewidths=0.5, alpha=0.6)
        a.scatter(
            *opt.df[opt.input_labels].to_numpy().T,
            c="k",
            marker="o",
            label="observed",
        )
        if possible_data is not None:
            a.scatter(
                *possible_data[opt.input_labels].to_numpy().T,
                c="w",
                marker="x",
                label="candidates",
            )
        a.scatter(
            *candidate[opt.input_labels],
            c="r",
            marker="*",
            s=200,
            label="next experiment",
        )
        a.set_title(title)
        a.set_xlabel(opt.input_labels[0])
        a.set_ylabel(opt.input_labels[1])
    step = max(n // 15, 1)
    ax[0].quiver(
        xx[::step, ::step],
        yy[::step, ::step],
        grad[::step, ::step, 0],
        grad[::step, ::step, 1],
        color="r",
    )
    ax[0].legend()
    fig.tight_layout()
    plt.show()
