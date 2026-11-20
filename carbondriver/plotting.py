import warnings
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm


def plot_2d_debug(
    opt,
    predictor,
    AF_q,
    raw_bounds: torch.Tensor,
    candidate: pd.Series,
    possible_data: Optional[pd.DataFrame] = None,
    n: int = 40,
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

    af = []
    grad = []
    mean = []
    std = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Output shape checks failed!")
        print("Computing acquisition function and posterior over 2D grid...")
        for batch in tqdm(grid_norm.split(grid_norm.shape[0] // 100, dim=0)):
            af_tmp = AF_q(batch)
            batch.retain_grad()
            af_tmp.sum().backward()
            af.append(af_tmp.detach())
            grad.append(batch.grad.detach())
            posterior = predictor.posterior(batch.detach())
            mean.append(posterior.mean[:, target_idx].detach())
            std.append(posterior.variance.sqrt()[:, target_idx].detach())
            
    af = torch.cat(af, dim=0).reshape(n, n)
    grad = torch.cat(grad, dim=0).squeeze(1).reshape(n, n, -1)
    mean = torch.cat(mean, dim=0).reshape(n, n)
    std = torch.cat(std, dim=0).reshape(n, n)

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
