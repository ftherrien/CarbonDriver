from pathlib import Path
import pickle
from typing import Optional, Tuple, Callable
import numpy as np
import pandas as pd
import torch
from torch.nn.parameter import UninitializedParameter
from torch.func import functional_call
from torch import vmap
import copy
from torch.func import stack_module_state
import gpytorch
import matplotlib.pyplot as plt
from math import ceil
from rich.progress import track
from carbondriver.models import (
    EnsPredictor,
    MultitaskGPhysModel,
    MultitaskGPModel,
    BoTorchGP,
)


def get_cov(batch: torch.Tensor) -> torch.Tensor:
    """Compute batch covariance with regularization.

    :param batch: tensor of samples
    :returns: covariance matrix
    """
    batch = batch.reshape(*batch.shape[:-2], -1)
    return torch.cov(batch.transpose(-1, -2)) + 1e-6 * torch.eye(batch.shape[1])


def get_nll(
    predictions: gpytorch.distributions.MultitaskMultivariateNormal,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute negative log likelihood from GP predictions by sampling.

    :param predictions: GP predictive distribution
    :param targets: target values
    :returns: NLL scalar
    """
    samples = predictions.sample(sample_shape=torch.Size([1000]))
    return get_nll_samples(samples, targets)


def get_nll_samples(
    samples: torch.Tensor,
    targets: torch.Tensor,
    mean_shifter: Optional[torch.Tensor] = None,
    covariance_scaler: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute NLL from samples, optionally scaling diagonal covariance.

    :param samples: samples from predictive distribution
    :param targets: target values
    :param mean_shifter: optional scalar to shift the mean
    :param covariance_scaler: optional scalar to scale variance
    :returns: NLL scalar
    """
    mean = samples.mean(dim=0)
    covariance = get_cov(samples)
    if mean_shifter is not None:
        mean += mean_shifter
    if covariance_scaler is not None:
        inds = torch.arange(covariance.shape[0])
        covariance[inds, inds] *= covariance_scaler
    gmodel = gpytorch.distributions.MultitaskMultivariateNormal(
        mean=mean, covariance_matrix=covariance
    )
    return -(gmodel.log_prob(targets) / gmodel.event_shape.numel())


def train_model_ens(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    model_constructor: Callable,
    num_iter: int,
    DNAME: str,
    i: int,
    progress: bool = False,
    plot: bool = False,
    shift_mean: bool = True,
) -> Tuple[pd.DataFrame, EnsPredictor]:
    """Train 50-model ensemble with per-model bagging and variance scaling.

    :param X_train: training features of shape (n, d)
    :param y_train: training targets of shape (n, m)
    :param model_constructor: callable that creates model instances
    :param num_iter: number of training iterations
    :param DNAME: directory name for saving stats
    :param i: iteration index for file naming
    :param progress: whether to show progress bar
    :param plot: whether to plot diagnostics
    :param shift_mean: whether to include learnable mean shift parameter
    :returns: tuple of (stats_df, ensemble_predictor)
    """
    DNAME = Path(DNAME)
    DNAME.mkdir(exist_ok=True)
    if plot:
        fig, ax = plt.subplots(ncols=3, figsize=(10, 3))
        ax[1].axline((0, 0), slope=1, c="k", ls="--")
        ax[2].axline((0, 0), slope=1, c="k", ls="--")
        # Clarify what each subplot shows: NLL and parity plots for the two FE targets
        ax[0].set_title("NLL (per-dimension) — lower is better")
        ax[1].set_title("Parity plot: FE (C2H5OH, Ethanol)")
        ax[2].set_title("Parity plot: FE (CO, Carbon monoxide)")

    # set up model and optimizer
    num_models = 50
    model = [model_constructor() for _ in range(num_models)]

    params, buffers = stack_module_state(model)
    base_model = copy.deepcopy(model[0])
    base_model = base_model.to("meta")

    def fmodel(params, buffers, x):
        return functional_call(base_model, (params, buffers), (x,))

    optimizer = torch.optim.Adam(params.values(), lr=0.001)
    variance_scaler = torch.tensor(1.0, requires_grad=True)
    if shift_mean:
        mean_shifter = torch.tensor(0.0, requires_grad=True)
        nll_optimizer = torch.optim.Adam([mean_shifter, variance_scaler], lr=1)
    else:
        mean_shifter = torch.tensor(0.0)
        nll_optimizer = torch.optim.Adam([variance_scaler], lr=1)

    # batch the train data
    num_data_per_model = ceil(X_train.shape[0] * 0.5)
    # inds = torch.stack([torch.arange(num_data_per_model) for _ in range(num_models)], dim=0)
    inds = torch.stack(
        [
            torch.randperm(X_train.shape[0])[:num_data_per_model]
            for _ in range(num_models)
        ],
        dim=0,
    )
    X_train_b = torch.stack([X_train[inds[i], :] for i in range(num_models)], dim=0)
    y_train_b = torch.stack([y_train[inds[i], :] for i in range(num_models)], dim=0)

    base_model.train()

    stats = {x: np.nan for x in ["loss", "val_loss"]} | {"step": np.arange(num_iter)}
    stats = pd.DataFrame(stats).set_index("step")

    iterator = range(num_iter)
    if progress:
        iterator = track(iterator)
    for it in iterator:
        optimizer.zero_grad()

        output = vmap(fmodel, in_dims=(0, 0, 0), randomness="different")(
            params, buffers, X_train_b
        )
        loss = torch.mean((output - y_train_b) ** 2)
        loss.backward()
        optimizer.step()
        stats.loc[it, "loss"] = loss.item()

        # test
        if it % ceil(num_iter / 100) == 0:

            base_model.eval()

            with torch.no_grad():
                fe_train = vmap(fmodel, in_dims=(0, 0, None), randomness="different")(
                    params, buffers, X_train
                )
                mean_train = fe_train.mean(dim=0) + mean_shifter
                std_train = fe_train.std(dim=0) * variance_scaler.sqrt()
            nll_optimizer.zero_grad()
            nll_train = get_nll_samples(
                fe_train, y_train,
                covariance_scaler=variance_scaler, mean_shifter=mean_shifter
            )
            nll_train.backward()
            nll_optimizer.step()
            stats.loc[it, "nll"] = nll_train.item()
            stats.loc[it, "variance_scaler"] = variance_scaler.item()
            stats.loc[it, "mean_shifter"] = mean_shifter.item()

            base_model.train()

        f = lambda x: round(x.item() if isinstance(x, torch.Tensor) else x, 5)
    if plot:
        stats["nll"].dropna().plot(
            y="nll", c="C0", ls="--", lw=0.7, alpha=0.5, ax=ax[0]
        )

    if plot:
        # plot parity plots with confidence intervals
        ax[1].errorbar(
            y_train[:, 0].numpy(),
            mean_train[:, 0].numpy(),
            yerr=std_train[:, 0].numpy(),
            fmt="o",
            alpha=0.5,
            mfc=f"C0",
            mec="white",
        )
        ax[2].errorbar(
            y_train[:, 1].numpy(),
            mean_train[:, 1].numpy(),
            yerr=std_train[:, 1].numpy(),
            fmt="o",
            alpha=0.5,
            mfc=f"C0",
            mec="white",
        )
        fig.tight_layout()
        # Add axis labels and more descriptive titles

        ax[0].set_xlabel("iteration")
        ax[0].set_ylabel("NLL (per-dimension)")
        ax[1].set_xlabel("true FE (C2H5OH)")
        ax[1].set_ylabel("predicted mean FE (C2H5OH)")
        ax[2].set_xlabel("true FE (CO)")
        ax[2].set_ylabel("predicted mean FE (CO)")
        # Ensure titles/labels render across backends: set again just before draw with padding
        ax[0].set_title(ax[0].get_title(), fontsize=11, pad=8, weight="semibold")
        ax[1].set_title(ax[1].get_title(), fontsize=11, pad=8, weight="semibold")
        ax[2].set_title(ax[2].get_title(), fontsize=11, pad=8, weight="semibold")
        # Add a small figure-level suptitle to clarify what the panels represent
        fig.suptitle("Training diagnostics (NLL and parity plots)", fontsize=12, y=0.98)
        # reserve space so titles/suptitle are not clipped
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.show()
    # save average losses to file
    stats.to_csv(DNAME / f"stats_{i:02d}.csv")

    def scaled_model(x):
        y = vmap(fmodel, in_dims=(0, 0, None), randomness="different")(
            params, buffers, x
        )
        y_diff = y - y.mean(dim=0, keepdim=True)
        return y.mean(dim=0) + y_diff * variance_scaler.sqrt()

    return stats, EnsPredictor(scaled_model)


def train_GP_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    num_iter: int,
    DNAME: str,
    i: int,
    progress: bool = False,
    plot: bool = False,
) -> Tuple[pd.DataFrame, Callable, BoTorchGP, gpytorch.likelihoods.Likelihood]:
    """Train multitask GP on full dataset.

    :param X_train: training features of shape (n, d)
    :param y_train: training targets of shape (n, 2)
    :param num_iter: number of training iterations
    :param DNAME: directory name for saving stats
    :param i: iteration index for file naming
    :param progress: whether to show progress bar
    :param plot: whether to plot diagnostics
    :returns: tuple of (stats_df, predict_fn, botorch_wrapper, likelihood)
    """
    DNAME = Path(DNAME)
    DNAME.mkdir(exist_ok=True, parents=True)
    if plot:
        fig, ax = plt.subplots(ncols=3, figsize=(10, 3))
        ax[1].axline((0, 0), slope=1, c="k", ls="--")
        ax[2].axline((0, 0), slope=1, c="k", ls="--")

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultitaskGPModel(X_train, y_train, likelihood)
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    model.train()
    likelihood.train()

    stats = {x: np.nan for x in ["loss", "val_loss"]} | {"step": np.arange(num_iter)}
    stats = pd.DataFrame(stats).set_index("step")

    iterator = range(num_iter)
    if progress:
        iterator = track(iterator)
    for it in iterator:
        model.train()
        likelihood.train()

        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        stats.loc[it, "loss"] = loss.item()

        # test
        if it % 5 == 0:

            model.eval()
            likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                predictions = likelihood(
                    model(X_train + 1e-6 * (torch.rand(X_train.shape) - 0.5))
                )
                mean_train = predictions.mean
                std_train = predictions.stddev
                stats.loc[it, "nll"] = get_nll(predictions, y_train).item()

    if plot:
        # loss curves
        stats["loss"].dropna().plot(
            y="loss", c="C0", ls="--", lw=0.7, alpha=0.5, ax=ax[0]
        )

        # plot parity plots with confidence intervals
        ax[1].errorbar(
            y_train[:, 0].numpy(),
            mean_train[:, 0].numpy(),
            yerr=std_train[:, 0].numpy(),
            fmt="o",
            alpha=0.5,
            mfc=f"C{i}",
            mec="white",
        )
        ax[2].errorbar(
            y_train[:, 1].numpy(),
            mean_train[:, 1].numpy(),
            yerr=std_train[:, 1].numpy(),
            fmt="o",
            alpha=0.5,
            mfc=f"C{i}",
            mec="white",
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Add axis labels and titles for clarity
        """
        ax[0].set_title('Loss (train) over iterations')
        ax[0].set_xlabel('iteration')
        ax[0].set_ylabel('negative marginal log likelihood')
        ax[1].set_title('Parity plot: FE (C2H5OH, Ethanol)')
        ax[1].set_xlabel('true FE (C2H5OH, Ethanol)')
        ax[1].set_ylabel('predicted mean FE (C2H5OH, Ethanol)')
        ax[2].set_title('Parity plot: FE (CO)')
        ax[2].set_xlabel('true FE (CO)')
        ax[2].set_ylabel('predicted mean FE (CO)')
        # Force title draw and short pause for some interactive backends, with padding
        ax[0].set_title(ax[0].get_title(), fontsize=11, pad=8, weight='semibold')
        ax[1].set_title(ax[1].get_title(), fontsize=11, pad=8, weight='semibold')
        ax[2].set_title(ax[2].get_title(), fontsize=11, pad=8, weight='semibold')
        fig.suptitle('GP training diagnostics (loss and parity)', fontsize=12, y=0.98)
        # reserve space so titles/suptitle are not clipped
        """
        plt.show()

    # save average losses to file
    stats.to_csv(DNAME / f"stats_{i:02d}.csv")

    def predict(X_test):
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = model(X_test)
            predictions = likelihood(output)
            mean_test = predictions.mean
            std_test = predictions.stddev
        return mean_test, std_test

    return stats, predict, BoTorchGP(model), likelihood


def train_Ph_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    model_constructor: Callable,
    num_iter: int,
) -> Tuple[pd.DataFrame, torch.nn.Module]:
    """Train physics-informed model with MSE loss.

    :param X_train: training features of shape (n, d)
    :param y_train: training targets of shape (n, 2)
    :param model_constructor: callable that creates model instances
    :param num_iter: number of training iterations
    :returns: tuple of (stats_df, trained_model)
    """
    model = model_constructor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    stats = {x: np.nan for x in ["loss", "val_loss", "nll"]} | {
        "step": np.arange(num_iter)
    }
    stats = pd.DataFrame(stats).set_index("step")

    model.train()
    for it in range(num_iter):
        optimizer.zero_grad()
        output = model(X_train)
        loss = torch.nn.functional.mse_loss(output, y_train)
        loss.backward()
        optimizer.step()
        stats.loc[it, "loss"] = loss.item()

    return stats, model


## For GP+Ph model training, not to be confused with train_GP_model above
def train_GP(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    mean_model: torch.nn.Module,
    num_iter: int,
) -> Tuple[pd.DataFrame, MultitaskGPhysModel]:
    """Train GP+Physics model with frozen mean model.

    :param X_train: training features of shape (n, d)
    :param y_train: training targets of shape (n, 2)
    :param mean_model: physics-informed model for mean function (will be frozen)
    :param num_iter: number of training iterations
    :returns: tuple of (stats_df, gp_physics_model)
    """
    # set up model and optimizer
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)

    model = MultitaskGPhysModel(
        X_train, y_train, likelihood, model=mean_model, freeze_model=True
    )

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    stats = {x: np.nan for x in ["loss", "val_loss", "nll"]} | {
        "step": np.arange(num_iter)
    }
    stats = pd.DataFrame(stats).set_index("step")

    for it in range(num_iter):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        stats.loc[it, "loss"] = loss.item()

        # Compute NLL for evaluation
        if it % 5 == 0:
            model.eval()
            likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                predictions = likelihood(
                    model(X_train + 1e-6 * (torch.rand(X_train.shape) - 0.5))
                )
                stats.loc[it, "nll"] = get_nll(predictions, y_train).item()

            model.train()
            likelihood.train()

    return stats, model


def train_GP_Ph_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    model_constructor: Callable,
    num_iter: int,
    DNAME: str,
    i: int,
    progress: bool = False,
    plot: bool = False,
    ph_frac: float = 0.5,
) -> Tuple[pd.DataFrame, Callable, BoTorchGP, gpytorch.likelihoods.Likelihood]:
    """Train physics model on subset, then GP+Physics on remainder.

    :param X_train: training features of shape (n, d)
    :param y_train: training targets of shape (n, 2)
    :param model_constructor: callable that creates physics model instances
    :param num_iter: number of training iterations per phase
    :param DNAME: directory name for saving stats
    :param i: iteration index for file naming
    :param progress: whether to show progress bar
    :param plot: whether to plot diagnostics
    :param ph_frac: fraction of data for physics model training
    :returns: tuple of (stats_df, predict_fn, botorch_wrapper, likelihood)
    """
    DNAME = Path(DNAME)
    DNAME.mkdir(exist_ok=True, parents=True)

    # split into subset
    N = X_train.shape[0]
    inds = torch.randperm(N)
    inds_ph = inds[: int(ph_frac * N)]
    inds_gp = inds[int(ph_frac * N) :]

    X_ph, y_ph = X_train[inds_ph], y_train[inds_ph]
    X_gp, y_gp = X_train[inds_gp], y_train[inds_gp]

    stats, model = train_Ph_model(X_ph, y_ph, model_constructor, num_iter)
    stats[["loss", "val_loss"]] = np.log(stats[["loss", "val_loss"]])

    stats2, model = train_GP(X_gp, y_gp, model, num_iter)
    stats2.index += stats.index.max()
    stats = pd.concat([stats, stats2], axis=0)
    stats.to_csv(DNAME / f"stats_{i:02d}.csv")

    def predict(X_test):
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = model(X_test)
            predictions = model.likelihood(output)
            mean_test = predictions.mean
            std_test = predictions.stddev
        return mean_test, std_test

    return stats, predict, BoTorchGP(model), model.likelihood
