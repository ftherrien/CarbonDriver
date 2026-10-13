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
) -> torch.Tensor:
    """Compute NLL from samples, optionally scaling diagonal covariance.
    Assumes samples are of shape (num_samples, num_data, num_tasks) and targets are of shape (num_data, num_tasks).

    :param samples: samples from predictive distribution
    :param targets: target values
    :returns: NLL scalar
    """
    mean = samples.mean(dim=0)
    covariance = get_cov(samples)
    gmodel = gpytorch.distributions.MultitaskMultivariateNormal(
        mean=mean, covariance_matrix=covariance
    )
    return -(gmodel.log_prob(targets) / gmodel.event_shape.numel())

def get_mean_shift(
    samples: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute the mean shift to optimize NLL.
    Assumes samples are of shape (num_samples, num_data, num_tasks) and targets are of shape (num_data, num_tasks).
    
    :param samples: samples from predictive distribution
    :param targets: target values
    :returns: optimal shift
    """
    targets = targets.reshape(-1, 1)
    mean = samples.reshape(samples.shape[0], -1, 1).mean(dim=0)
    covariance = get_cov(samples)
    
    B = 0.5 * ((targets - mean).T + (targets - mean)) * torch.linalg.inv(covariance)

    return B.sum() / (torch.linalg.inv(covariance).sum())

def train_model_ens(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    model_constructor: Callable,
    num_iter: int,
    DNAME: str,
    i: int,
    progress: bool = False,
    plot: bool = False,
    num_models: int = 50,
    batch_fraction: float = 0.5,
    variance_scaling: bool = True,
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
    :param num_models: Number of models in the ensemble
    :param batch_fraction: Fraction of training data to use for each model
    :param variance_scaling: whether to apply variance scaling to ensemble predictions
    :returns: tuple of (stats_df, ensemble_predictor)
    """
    
    
    DNAME = Path(DNAME)
    DNAME.mkdir(exist_ok=True)
    n_dims = y_train.shape[-1]
    if plot:
        fig, ax = plt.subplots(ncols=1 + n_dims, figsize=(3 * (1 + n_dims), 3))
        ax[0].set_title("NLL (per-dimension) — lower is better")
        for j in range(n_dims):
            ax[1 + j].axline((0, 0), slope=1, c="k", ls="--")
            ax[1 + j].set_title(f"Parity plot: dim {j}")

    # set up model and optimizer
    model = [model_constructor() for _ in range(num_models)]

    params, buffers = stack_module_state(model)
    base_model = copy.deepcopy(model[0])
    base_model = base_model.to("meta")

    def fmodel(params, buffers, x):
        return functional_call(base_model, (params, buffers), (x,))

    optimizer = torch.optim.Adam(params.values(), lr=0.001)

    # batch the train data
    num_data_per_model = ceil(X_train.shape[0] * batch_fraction)
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
        

    with torch.no_grad():
        fe_train = vmap(fmodel, in_dims=(0, 0, None), randomness="different")(
        params, buffers, X_train
    )
    mean_train = fe_train.mean(dim=0)
    std_train = fe_train.std(dim=0)


    if variance_scaling:
        mean_shift = get_mean_shift(fe_train, y_train)
    
        # Analytical best scaler to minimize NLL
        variance_scaler = 2*(get_nll_samples(fe_train + mean_shift, y_train) - get_nll_samples(fe_train + mean_shift, fe_train.mean(dim=0) + mean_shift))

    else:
        mean_shift = torch.tensor(0.0)
        variance_scaler = torch.tensor(1.0)
        
    def scaled_model(x):
        y = vmap(fmodel, in_dims=(0, 0, None), randomness="different")(
            params, buffers, x
        )
        y_diff = y - y.mean(dim=0, keepdim=True)
        return y.mean(dim=0) + mean_shift + y_diff * variance_scaler.sqrt()

    shifted_fe_train = scaled_model(X_train)
    shifted_mean_train = shifted_fe_train.mean(dim=0)
    scaled_std_train = shifted_fe_train.std(dim=0)
    
    if plot:
        stats["loss"].dropna().plot(
            y="loss", c="C0", ls="--", lw=0.7, alpha=0.5, ax=ax[0]
        )

    if plot:
        # parity plots with confidence intervals, one panel per output dimension
        for j in range(n_dims):
            ax[1 + j].errorbar(
                y_train[:, j].numpy(),
                mean_train[:, j].numpy(),
                yerr=std_train[:, j].numpy(),
                fmt="o",
                alpha=0.5,
                mfc="C0",
                mec="white",
            )
            ax[1 + j].errorbar(
                y_train[:, j].numpy(),
                shifted_mean_train[:, j].detach().numpy(),
                yerr=scaled_std_train[:, j].detach().numpy(),
                fmt="o",
                alpha=0.5,
                mfc="C1",
                mec="white",
            )
            ax[1 + j].set_xlabel(f"true dim {j}")
            ax[1 + j].set_ylabel(f"predicted mean dim {j}")
        ax[0].set_xlabel("iteration")
        ax[0].set_ylabel("Loss")
        fig.suptitle("Training diagnostics (Loss and parity plots)", fontsize=12, y=0.98)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.show()
    # save average losses to file
    stats.to_csv(DNAME / f"stats_{i:02d}.csv")

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
    n_dims = y_train.shape[-1] if y_train.ndim > 1 else 1
    if plot:
        fig, ax = plt.subplots(ncols=1 + n_dims, figsize=(3 * (1 + n_dims), 3))
        for j in range(n_dims):
            ax[1 + j].axline((0, 0), slope=1, c="k", ls="--")

    num_tasks = n_dims
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
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

        # parity plots with confidence intervals, one panel per output dimension
        for j in range(n_dims):
            ax[1 + j].errorbar(
                y_train[:, j].numpy(),
                mean_train[:, j].numpy(),
                yerr=std_train[:, j].numpy(),
                fmt="o",
                alpha=0.5,
                mfc=f"C{i}",
                mec="white",
            )
            ax[1 + j].set_title(f"Parity plot: dim {j}")
            ax[1 + j].set_xlabel(f"true dim {j}")
            ax[1 + j].set_ylabel(f"predicted mean dim {j}")
        ax[0].set_title("Loss (train) over iterations")
        ax[0].set_xlabel("iteration")
        ax[0].set_ylabel("negative marginal log likelihood")
        fig.suptitle("GP training diagnostics (loss and parity)", fontsize=12, y=0.98)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
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
    :param y_train: training targets of shape (n, num_tasks)
    :param mean_model: physics-informed model for mean function (will be frozen)
    :param num_iter: number of training iterations
    :returns: tuple of (stats_df, gp_physics_model)
    """
    # set up model and optimizer
    num_tasks = y_train.shape[-1] if y_train.ndim > 1 else 1
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

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
