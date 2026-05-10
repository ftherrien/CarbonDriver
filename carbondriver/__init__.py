from .models import PhModel, MLPModel, MultitaskGPModel, BoTorchGP, MultitaskGPhysModel
from .train import train_model_ens, train_GP_model, train_GP_Ph_model
from .loaders import feature_stats, default_input_labels, default_output_labels
from .config import default_config
import pandas as pd
import torch
import numpy as np
from typing import Tuple, Optional
from botorch.acquisition.analytic import LogExpectedImprovement, ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.acquisition.objective import ScalarizedPosteriorTransform
import warnings

SUPPORTED_AFs = ["EI", "logEI"]


class GDEOptimizer:
    """
    Class to optimize gas diffusion electrodes experimental parameters based with Bayesian optimization using various models.
    """

    def __init__(
        self,
        model_name="GP+Ph",
        aquisition="EI",
        quantity="FE (Eth)",
        maximize=True,
        output_dir="./out",
        config=default_config,
        bounds=None,
        input_labels=None,
        output_labels=None,
    ) -> None:
        """
        Initialize the optimizer with the specified model and acquisition function.

        :param model_name: Name of the model to use (e.g., 'GP', 'Ph', 'MLP', 'GP+Ph')
        :param aquisition: Acquisition function to use (e.g., 'EI' for Expected Improvement)
        :param quantity: The quantity to optimize (e.g., 'FE (Eth)')
        :param maximize: Whether to maximize or minimize the quantity
        :param output_dir: Directory to save output files
        :param config: Configuration dictionary with parameters for training and normalization
        :param bounds: Bounds for the optimization, should be a tensor of shape (2, num_features)
        :param input_labels: Custom input feature labels (default: GDE electrode parameters)
        :param output_labels: Custom output labels (default: FE (Eth), FE (CO))
        """

        if model_name == "GP":
            self.model = MultitaskGPModel
        elif model_name == "Ph":
            self.model = PhModel
        elif model_name == "MLP":
            self.model = MLPModel
        elif model_name == "GP+Ph":
            self.model = MultitaskGPhysModel
        else:
            raise ValueError(
                f"Unsupported model_name '{model_name}'. Supported options are 'GP', 'Ph', 'MLP', 'GP+Ph'."
            )

        if aquisition in SUPPORTED_AFs:
            self.aquisition = aquisition
            if self.aquisition == "EI":
                print(
                    "WARNING: You are using expected improvement, logEI is recommended instead."
                )
        else:
            raise ValueError(
                f"Only {' and '.join(SUPPORTED_AFs)} are supported for now."
            )

        self.output_dir = output_dir

        self.config = default_config | config
        dataset = self.config.get("dataset", "gas")

        self.maximize = maximize

        self.quantity = quantity

        self.i = 0

        self.df = pd.DataFrame()

        self._bounds = bounds

        if input_labels is None:
            self.input_labels = default_input_labels(dataset)
        else:
            # If input_labels are provided, use them
            self.input_labels = input_labels

        if output_labels is None:
            self.output_labels = default_output_labels(dataset)
        else:
            # If output_labels are provided, use them
            self.output_labels = output_labels

        # Stats for normalization of feature columns (set in get_predictor when normalize=True)
        self._means = None  # torch.Tensor of shape (d,)
        self._stds = None  # torch.Tensor of shape (d,)

    def _get_data_tensors(
        self, data: Optional[pd.DataFrame] = None, update_stats: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert DataFrame to tensors, applying normalization if configured.

        :param data: DataFrame with input and output columns (default: self.df)
        :param update_stats: whether to recompute normalization statistics
        :returns: tuple of (X, y) tensors
        """

        if data is None:
            data = self.df

        df_clean = data.loc[:, self.input_labels + self.output_labels]

        if update_stats or self._means is None or self._stds is None:
            self._means = df_clean.mean()
            self._stds = df_clean.std(ddof=0)

        if self.config["normalize_inputs"]:
            if self._stds[self.input_labels].min() < 1e-10:
                print(
                    "Note: Input feature standard deviation is < 1e-10, normalizing with 1"
                )
                self._stds.loc[
                    self._stds.index.isin(self.input_labels) & (self._stds < 1e-10)
                ] = 1

            X = (
                df_clean.loc[:, self.input_labels] - self._means[self.input_labels]
            ) / self._stds[self.input_labels]
        else:
            X = df_clean.loc[:, self.input_labels]

        if self.config["normalize_outputs"]:
            if self._stds[self.output_labels].min() < 1e-10:
                print(
                    "Note: Output feature standard deviation is < 1e-10, normalizing with 1"
                )
                self._stds.loc[
                    self._stds.index.isin(self.output_labels) & (self._stds < 1e-10)
                ] = 1

            y = (
                df_clean.loc[:, self.output_labels] - self._means[self.output_labels]
            ) / self._stds[self.output_labels]
        else:
            y = df_clean.loc[:, self.output_labels]

        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)

        return X, y

    @property
    def bounds(self) -> torch.Tensor:
        """
        Get the bounds for the optimization based on the data if not specified in the declaration.

        :returns: tensor of shape (2, d) with min and max bounds for each feature
        """
        if self._bounds is None:
            bds_max = self.df.loc[:, self.input_labels].values.max(axis=0)
            bds_min = self.df.loc[:, self.input_labels].values.min(axis=0)
            raw_bounds = torch.tensor([bds_min, bds_max], dtype=torch.float32)
            # Debug: show computed raw bounds
            # print(f"[bounds] raw min: {raw_bounds[0].tolist()} raw max: {raw_bounds[1].tolist()}")
            return raw_bounds
        else:
            return self._bounds

    def load(path):
        """
        Load pretrained model.
        """
        raise NotImplementedError

    def update_data(self, new_data: pd.DataFrame | pd.Series) -> None:
        """
        Add new experimental data to the dataset (and sort by 'triplet' for now).

        :param new_data: New data to add (pd.Series or pd.DataFrame)
        :returns: None. Modifies self.df in-place.
        """

        if isinstance(new_data, pd.Series):
            new_data = new_data.to_frame().T

        self.df = pd.concat([self.df, new_data], axis=0)

        self.df.sort_values(by="triplet", inplace=True)  # TMP

    def get_predictor(self) -> Tuple[torch.nn.Module | BoTorchGP, pd.DataFrame]:
        """
        Train and return the new predictor based on the new data.

        :returns: (model, stats) tuple where model is the trained predictor and stats is a DataFrame with training metrics.
        """
        X, y = self._get_data_tensors(update_stats=True)

        mu = None
        sigma = None
        zlt_index = None
        if self.model in (PhModel, MultitaskGPhysModel):
            if "Zero_eps_thickness" not in self._means.index:
                raise ValueError(
                    "Zero_eps_thickness must be present in input_labels for PhModel-based models."
                )
            mu = float(self._means["Zero_eps_thickness"])
            sigma = float(self._stds["Zero_eps_thickness"])
            zlt_index = self.input_labels.index("Zero_eps_thickness")

        # Special handling for GP and GP+Ph models: these use gpytorch training functions
        # (they are not compatible with the ensemble training pipeline used for MLP/Ph).
        if self.model == MultitaskGPModel:

            # Train GP and return BoTorch-compatible model
            stats, _, model, likelihood = train_GP_model(
                X,
                y,
                num_iter=self.config["num_iter"],
                DNAME=self.output_dir,
                i=self.i,
                plot=self.config["make_plots"],
            )

        elif self.model == MultitaskGPhysModel:
            # GP+Physics: Ph model constructor must be provided to the GP+Ph trainer.

            ph_model_constructor = lambda: PhModel(
                zlt_mu=mu,
                zlt_sigma=sigma,
                current_target=233,
                config=self.config,
                n_inputs=len(self.input_labels),
                zlt_index=zlt_index,
            )

            # Train GP+Ph and return BoTorch-compatible model
            stats, _, model, likelihood = train_GP_Ph_model(
                X,
                y,
                ph_model_constructor,
                num_iter=self.config["num_iter"],
                DNAME=self.output_dir,
                i=self.i,
                plot=self.config["make_plots"],
            )

        # Handle ensemble models (MLP and Ph)
        else:
            if self.model == PhModel:

                model_factory = lambda: PhModel(
                    zlt_mu=mu,
                    zlt_sigma=sigma,
                    current_target=233,
                    config=self.config,
                    dropout=0.0,
                    n_inputs=len(self.input_labels),
                    zlt_index=zlt_index,
                )

            elif self.model == MLPModel:
                # MLP model with explicit input/output sizes
                n_in = len(self.input_labels)
                n_out = len(self.output_labels)
                model_factory = lambda: MLPModel(
                    n_inputs=n_in,
                    n_outputs=n_out,
                )

            stats, model = train_model_ens(
                X,
                y,
                model_factory,
                DNAME=self.output_dir,
                i=self.i,
                num_iter=self.config["num_iter"],
                plot=self.config["make_plots"],
            )

        # Compute a simple training-set MAE for reporting. This is a summary
        # metric only; it does not affect acquisition.
        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=gpytorch.utils.warnings.GPInputWarning)
            if self.model in (MultitaskGPModel, MultitaskGPhysModel):
                if self.model == MultitaskGPModel:
                    model.eval()
                    likelihood.eval()
                    mean_predictions = likelihood(model(X)).mean
                else:
                    model.eval()
                    likelihood.eval()
                    mean_predictions = likelihood(model(X)).mean
            else:
                model.eval()
                mean_predictions = model(X)

            mae = torch.mean(torch.abs(mean_predictions - y)).item()

        if "mae" not in stats.columns:
            stats["mae"] = np.nan
        stats.loc[stats.index.max(), "mae"] = mae

        return model, stats

    def _get_acquisition_function(
        self, predictor: torch.nn.Module
    ) -> ExpectedImprovement | LogExpectedImprovement:
        """
        Get the acquisition function based on the specified acquisition type.

        :param predictor: trained model for predictions
        :returns: BoTorch acquisition function (EI or logEI)

        Note: The acquisition function is in normalized space if normalizatipon is enabled, so the expected imporovment is not the actual value for example. Also normalizing here just for consistency because y is not actually normalized.
        """

        _, y = self._get_data_tensors()

        target_idx = self.output_labels.index(self.quantity)

        if self.config["EI_reference"] == "max":
            best_f = y[:, target_idx].max()
        elif self.config["EI_reference"] == "min":
            best_f = y[:, target_idx].min()
        else:
            raise ValueError(
                f"Unsupported EI_reference {self.config['EI_reference']}, expected 'max' or 'min'"
            )

        if self.aquisition == "EI":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return ExpectedImprovement(
                    predictor,
                    best_f=best_f,
                    maximize=self.maximize,
                )
        if self.aquisition == "logEI":
            return LogExpectedImprovement(
                predictor,
                best_f=best_f,
                maximize=self.maximize,
            )
        else:
            raise ValueError("Unsupported acquisition function.")

    def step(
        self, new_data: pd.DataFrame, bounds: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, pd.Series]:
        """
        Perform a step in the optimization process using the new data and bounds.

        :param new_data: New data to be added to the training set. Do not input data that was alerady given to the object, only new data.
        :param bounds: Optional bounds for the optimization (default: inferred from data)
        :returns: tuple of (acquisition_function_value, next_experiment_parameters)
        """
        self.update_data(new_data)

        # Determine raw bounds (always in original feature scale)
        # Use the property-created tensor by default. If the caller supplied `bounds`,
        # require that it already be a torch.Tensor.
        raw_bounds = self.bounds if bounds is None else bounds
        if bounds is not None and not isinstance(raw_bounds, torch.Tensor):
            raise TypeError(
                "bounds must be a torch.Tensor of shape (2, d). "
                "Convert lists/arrays with torch.as_tensor(..., dtype=torch.float32) before calling step."
            )
        assert raw_bounds.shape[0] == 2, "Bounds should have shape (2, d)"

        try:
            predictor, stats = self.get_predictor()
        except torch._C._LinAlgError:
            print(
                "LinAlgError during ensemble training. System may be underdetermined. Returning a random candidate."
            )
            x_candidate = (
                torch.randn(len(self.input_labels))
                * (raw_bounds[1, :] - raw_bounds[0, :])
                + raw_bounds[0, :]
            )

            return torch.nan, pd.Series(
                x_candidate.detach().cpu().numpy().flatten(), index=self.input_labels
            )
        except RuntimeError as e:
            # Handle gpytorch ExactGP runtime error when model is called with inputs
            # that don't exactly match the stored training inputs (raised in debug mode).
            msg = str(e)
            if (
                "You must train on the training inputs" in msg
                or "train_inputs cannot be None" in msg
            ):
                print(
                    "RuntimeError during GP training (likely mismatched training inputs). Treating as underdetermined and returning a random candidate."
                )
                x_candidate = (
                    torch.randn(len(self.input_labels))
                    * (raw_bounds[1, :] - raw_bounds[0, :])
                    + raw_bounds[0, :]
                )
                return torch.nan, pd.Series(
                    x_candidate.detach().cpu().numpy().flatten(),
                    index=self.input_labels,
                )
            else:
                # Unknown runtime error: re-raise so we don't silently swallow unrelated failures
                raise

        AF = self._get_acquisition_function(predictor)

        # Select the output column index for the quantity by name from the last two columns

        try:
            target_idx = self.output_labels.index(self.quantity)
        except ValueError:
            raise ValueError(
                f"Quantity '{self.quantity}' not found in output columns {self.output_labels}"
            )
        # print(f"[step] optimizing target column index (target_idx): {target_idx} for quantity '{self.quantity}'")

        if self.config["normalize_inputs"]:
            means, stds = (
                self._means[self.input_labels],
                self._stds[self.input_labels],
            )  # feature-only stats

            bounds_norm = torch.stack(
                [
                    (raw_bounds[0] - means.values) / stds.values,
                    (raw_bounds[1] - means.values) / stds.values,
                ],
                dim=0,
            )
            # print(f"[step] normalized bounds min: {bounds_norm[0].tolist()} max: {bounds_norm[1].tolist()}")
            opt_bounds = bounds_norm.float()
        else:
            opt_bounds = raw_bounds

        def AF_q(x):
            vals = AF(x)
            # vals can be:
            #  - 1D: (batch,) already scalar per point
            #  - 2D: (batch, m) for m outputs
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

        next_experiment, _ = optimize_acqf(
            acq_function=AF_q,
            bounds=opt_bounds,
            q=1,
            num_restarts=20,
            raw_samples=30,
            options={},
        )

        if self.config["normalize_inputs"]:
            # Denormalize the candidate if optimization was done in normalized space
            x_candidate = next_experiment * stds.values + means.values

        else:
            x_candidate = next_experiment

        self.i += 1

        # Evaluate AF at the (normalized) next point for returning EI value
        ei_val = AF_q(next_experiment)

        return ei_val, pd.Series(
            x_candidate.detach().cpu().numpy().flatten(), index=self.input_labels
        )

    def step_within_data(
        self,
        new_data: pd.DataFrame,
        possible_data: pd.DataFrame,
        return_metrics: bool = False,
    ) -> Tuple[float, int] | Tuple[float, int, dict]:
        """
        Selects the best next point from a set of candidates (possible_dat) considering the new data (new_data) and the existing data (if any).

        :param new_data: New data to be added to the training set. Do not input data that was alerady given to the object, only new data.
        :param possible_data: DataFrame of candidate points to select from
        :param return_metrics: whether to return training metrics (nll, loss)
        :returns: (best_ei_value, best_index) or (best_ei_value, best_index, metrics) if return_metrics=True
        """

        self.update_data(new_data)

        try:
            predictor, stats = self.get_predictor()
        except torch._C._LinAlgError:
            print(
                "LinAlgError during ensemble training. System may be underdetermined."
            )
            return torch.nan, torch.randint(len(possible_data) - 1, (1,)).squeeze(), {}
        except RuntimeError as e:
            msg = str(e)
            if (
                "You must train on the training inputs" in msg
                or "train_inputs cannot be None" in msg
                or "cholesky_cpu" in msg
            ):
                print("RuntimeError during GP training. Treating as underdetermined.")
                if return_metrics:
                    return (
                        torch.nan,
                        torch.randint(len(possible_data) - 1, (1,)).squeeze(),
                        {},
                    )
                else:
                    return (
                        torch.nan,
                        torch.randint(len(possible_data) - 1, (1,)).squeeze()
                    )
            else:
                raise

        X, _ = self._get_data_tensors(data=possible_data)

        AF = self._get_acquisition_function(predictor)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            scores = AF(X.unsqueeze(1))

        if isinstance(scores, torch.Tensor) and scores.dim() == 1:
            scores = scores.unsqueeze(0)

        self.i += 1

        target_idx = self.output_labels.index(self.quantity)

        if isinstance(scores, torch.Tensor):
            if scores.shape[1] <= target_idx:
                raise RuntimeError(
                    f"AF scores shape {tuple(scores.shape)} has no column {target_idx}"
                )
            target_scores = scores[:, target_idx]
        else:
            raise RuntimeError("AF returned non-tensor scores, expected torch.Tensor")
        print(f"Target scores : {target_scores.tolist()}")
        best_idx = int(target_scores.argmax().item())
        best_ei = float(target_scores[best_idx].item())

        # Extract final training metrics from stats
        metrics = {}
        if "nll" in stats.columns:
            # Get last non-NaN NLL value
            nll_vals = stats["nll"].dropna()
            if len(nll_vals) > 0:
                metrics["nll"] = float(nll_vals.iloc[-1])
            else:
                metrics["nll"] = np.nan
        else:
            metrics["nll"] = np.nan

        if "loss" in stats.columns:
            # Get last non-NaN loss (often MSE/MAE proxy)
            loss_vals = stats["loss"].dropna()
            if len(loss_vals) > 0:
                metrics["loss"] = float(loss_vals.iloc[-1])
            else:
                metrics["loss"] = np.nan
        else:
            metrics["loss"] = np.nan

        if "mae" in stats.columns:
            mae_vals = stats["mae"].dropna()
            if len(mae_vals) > 0:
                metrics["mae"] = float(mae_vals.iloc[-1])
            else:
                metrics["mae"] = np.nan
        else:
            metrics["mae"] = np.nan

        if return_metrics:
            return best_ei, best_idx, metrics
        else:
            return best_ei, best_idx
