import warnings

import pandas as pd
import torch
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    LogExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.objective import ScalarizedPosteriorTransform

from .models import BoTorchGP


def get_acquisition_function(
    opt, predictor: torch.nn.Module
) -> ExpectedImprovement | LogExpectedImprovement | ProbabilityOfImprovement:
    """
    Get the acquisition function based on the specified acquisition type.

    :param opt: GDEOptimizer instance
    :param predictor: trained model for predictions
    :returns: BoTorch acquisition function (EI, logEI, or PI)

    Note: The acquisition function is in normalized space if normalization is enabled, so the expected improvement is not the actual value for example. Also normalizing here just for consistency because y is not actually normalized.
    """

    _, y = opt._get_data_tensors()

    target_idx = opt.output_labels.index(opt.quantity)

    posterior_transform = None
    if len(opt.output_labels) > 1 and isinstance(predictor, BoTorchGP):
        weights = torch.zeros(len(opt.output_labels), dtype=torch.float32)
        weights[target_idx] = 1.0
        posterior_transform = ScalarizedPosteriorTransform(weights=weights)

    if opt.config["EI_reference"] == "max":
        best_f = y[:, target_idx].max()
    elif opt.config["EI_reference"] == "min":
        best_f = y[:, target_idx].min()
    else:
        raise ValueError(
            f"Unsupported EI_reference {opt.config['EI_reference']}, expected 'max' or 'min'"
        )

    if opt.acquisition == "EI":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ExpectedImprovement(
                predictor,
                best_f=best_f,
                maximize=opt.maximize,
                posterior_transform=posterior_transform,
            )
    if opt.acquisition == "logEI":
        return LogExpectedImprovement(
            predictor,
            best_f=best_f,
            maximize=opt.maximize,
            posterior_transform=posterior_transform,
        )
    if opt.acquisition == "PI":
        return ProbabilityOfImprovement(
            predictor,
            best_f=best_f,
            maximize=opt.maximize,
            posterior_transform=posterior_transform,
        )
    if opt.acquisition == "UCB":
        beta = opt.config.get("UCB_beta", 1.0)
        return UpperConfidenceBound(
            predictor,
            beta=beta,
            maximize=opt.maximize,
            posterior_transform=posterior_transform,
        )
    raise ValueError(f"Unsupported acquisition function: {opt.acquisition}")


def select_acquisition_target(vals: torch.Tensor, target_idx: int) -> torch.Tensor:
    # vals can be:
    #  - 0D: single-candidate, single-output acquisition value
    #  - 1D: (batch,) already scalar per point
    #  - 2D: (batch, m) for m outputs
    if vals.dim() == 0:
        return vals.reshape(1)
    if vals.dim() == 1:
        return vals
    if vals.dim() == 2:
        if vals.size(1) == 1:
            return vals.squeeze(1)
        if target_idx >= vals.size(1):
            raise RuntimeError(
                f"Acquisition returned {vals.size(1)} outputs but target_idx={target_idx}"
            )
        return vals[:, target_idx]
    # Fallback: flatten all but batch dim and take first column
    return vals.view(vals.shape[0], -1)[:, 0]


def check_underdetermined_fallback(config: dict, error: Exception) -> None:
    """Raise if the random-candidate fallback is disabled, else warn about it."""
    if not config.get("propose_random_when_underdetermined", True):
        raise RuntimeError(
            "Carbon Driver could not fit the optimization model; the training "
            "data may be underdetermined (duplicate conditions, too little "
            "variation, or too few observations). No recommendation was "
            "generated. Set config['propose_random_when_underdetermined'] = "
            "True to fall back to a random candidate instead."
        ) from error
    warnings.warn(
        f"System may be underdetermined ({error}). Returning a random candidate."
    )


def random_candidate(input_labels: list, raw_bounds: torch.Tensor):
    x_candidate = (
        torch.rand(len(input_labels)) * (raw_bounds[1, :] - raw_bounds[0, :])
        + raw_bounds[0, :]
    )
    return pd.Series(
        x_candidate.detach().cpu().numpy().flatten(), index=input_labels
    )
