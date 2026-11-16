import warnings

import pandas as pd
import torch
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    LogExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)
from botorch.acquisition.objective import GenericMCObjective, ScalarizedPosteriorTransform
from botorch.models.multitask import MultiTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler

from .models import BoTorchGP

MC_EQUIVALENCE = {
    "EI": qExpectedImprovement,
    "logEI": qLogExpectedImprovement,
    "PI": qProbabilityOfImprovement,
    "UCB": qUpperConfidenceBound,
}

ACQF_NAME = {
    "EI": ExpectedImprovement,
    "logEI": LogExpectedImprovement,
    "PI": ProbabilityOfImprovement,
    "UCB": UpperConfidenceBound,
    "qEI": qExpectedImprovement,
    "qlogEI": qLogExpectedImprovement,
    "qPI": qProbabilityOfImprovement,
    "qUCB": qUpperConfidenceBound,
}


def get_acquisition_function(opt, predictor: torch.nn.Module, X_pending: torch.Tensor | None = None):
    """
    Get the acquisition function based on the specified acquisition type.

    :param opt: GDEOptimizer instance
    :param predictor: trained model for predictions
    :param X_pending: points already committed to being observed (e.g. duplicates
        of existing data chosen earlier in a retry loop). If given, switches to
        the Monte-Carlo (q-batch) counterpart of the acquisition function.
    :returns: BoTorch acquisition function (EI, logEI, PI, UCB, or their q-batch counterparts)
    """

    _, y = opt._get_data_tensors()

    target_idx = opt.output_labels.index(opt.quantity)

    posterior_transform = None
    if len(opt.output_labels) > 1 and isinstance(predictor, (BoTorchGP, MultiTaskGP)):
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

    if opt.acquisition in MC_EQUIVALENCE:
        if X_pending is None:
            kwargs = dict(maximize=opt.maximize)
            if opt.acquisition == "UCB":
                kwargs["beta"] = opt.config.get("UCB_beta", 1.0)
            else:
                kwargs["best_f"] = best_f
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # ExpectedImprovement's deprecation warning
                return ACQF_NAME[opt.acquisition](predictor, posterior_transform=posterior_transform, **kwargs)
        acqf = MC_EQUIVALENCE[opt.acquisition]
    else:
        acqf = ACQF_NAME[opt.acquisition]

    # The q-batch classes have no `maximize` flag
    sign = 1.0 if opt.maximize else -1.0
    objective = GenericMCObjective(lambda samples, X=None: sign * samples.squeeze(-1))
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([opt.config.get("mc_samples", 256)]))
    kwargs = dict(
        sampler=sampler, objective=objective, posterior_transform=posterior_transform, X_pending=X_pending
    )
    kwargs["beta" if "UCB" in opt.acquisition else "best_f"] = (
        opt.config.get("UCB_beta", 1.0) if opt.acquisition == "UCB" else sign * best_f
    )
    return acqf(predictor, **kwargs)


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


def check_random_fallback(config: dict, reason: str) -> None:
    """Raise if the random-candidate fallback is disabled, else warn about it."""
    if not config.get("propose_random_on_fallback", True):
        raise RuntimeError(
            f"{reason} No recommendation was generated. Set "
            "config['propose_random_on_fallback'] = True to fall back to a "
            "random candidate instead."
        )
    warnings.warn(f"{reason} Returning a random candidate.")


def random_candidate(input_labels: list, raw_bounds: torch.Tensor):
    x_candidate = (
        torch.rand(len(input_labels)) * (raw_bounds[1, :] - raw_bounds[0, :])
        + raw_bounds[0, :]
    )
    return pd.Series(x_candidate.detach().cpu().numpy().flatten(), index=input_labels)
