import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from botorch.optim import optimize_acqf

from .acquisition import (
    check_random_fallback,
    get_acquisition_function,
    random_candidate,
    select_acquisition_target,
)
from .config import default_config
from .llm import llm_suggest
from .models import (
    MODELED_QUANTITIES,
    BoTorchGP,
    MLPModel,
    MultitaskGPhysModel,
    MultitaskGPModel,
    PhModel,
)
from .plotting import plot_2d_debug
from .train import (
    train_BoTorch_GP_model,
    train_GP_model,
    train_GP_Ph_model,
    train_model_ens,
)

SUPPORTED_AFs = ["EI", "logEI", "PI", "UCB"]


class GDEOptimizer:
    """
    Class to optimize gas diffusion electrodes experimental parameters based with Bayesian optimization using various models.
    """

    def __init__(
        self,
        model_name="GP+Ph",
        acquisition="EI",
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
        :param acquisition: Acquisition function to use (e.g., 'EI' for Expected Improvement)
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
        elif model_name == "BoTorch-GP":
            self.model = "BoTorch-GP"
        elif model_name == "LLM":
            self.model = "LLM"
        else:
            raise ValueError(
                f"Unsupported model_name '{model_name}'. Supported options are 'GP', 'Ph', 'MLP', 'GP+Ph', 'BoTorch-GP', 'LLM'."
            )

        if acquisition in SUPPORTED_AFs:
            self.acquisition = acquisition
            if self.acquisition == "EI":
                print(
                    "WARNING: You are using expected improvement, logEI is recommended instead."
                )
        else:
            raise ValueError(
                f"Only {' and '.join(SUPPORTED_AFs)} are supported for now."
            )

        self.output_dir = output_dir

        self.config = default_config | config

        self.maximize = maximize

        self.quantity = quantity

        self.i = 0

        self.df = pd.DataFrame()

        self.llm_history = []  # list of {"step": i, "suggestion": ..., "reason": ...}

        self.predictor = None  # cached by step()/step_within_data(): the model fit used for the last recommendation

        self._bounds = bounds

        if input_labels is None:
            self.input_labels = [
                "AgCu Ratio",
                "Naf vol (ul)",
                "Sust vol (ul)",
                "zero_eps_thickness",
                "Catalyst mass loading",
            ]
        else:
            self.input_labels = input_labels

        if output_labels is None:
            self.output_labels = ["FE (Eth)", "FE (CO)"]
        else:
            self.output_labels = output_labels

        # Stats for normalization of feature columns (set in get_predictor when normalize=True)
        self._means = pd.Series(0.0, self.input_labels + self.output_labels)
        self._stds = pd.Series(1.0, self.input_labels + self.output_labels)

    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in vars(self).items())

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

        output_labels = [col for col in self.output_labels if col in data.columns]

        df_clean = data.loc[:, self.input_labels + output_labels]

        if update_stats:
            if self.config["normalize_inputs"]:
                self._means.loc[self.input_labels] = df_clean.loc[
                    :, self.input_labels
                ].mean()
                self._stds.loc[self.input_labels] = df_clean.loc[
                    :, self.input_labels
                ].std(ddof=0)

            if self.config["normalize_outputs"]:
                self._means.loc[output_labels] = df_clean.loc[:, output_labels].mean()
                self._stds.loc[output_labels] = df_clean.loc[:, output_labels].std(
                    ddof=0
                )

            if self._stds.min() < 1e-10:
                print(
                    "Note: Some features standard deviation are < 1e-10, normalizing with 1"
                )
                self._stds.loc[self._stds < 1e-10] = 1

        df_clean = (df_clean - self._means) / self._stds

        X = torch.tensor(
            df_clean.loc[:, self.input_labels].to_numpy().copy(), dtype=torch.float32
        )
        y = torch.tensor(
            df_clean.loc[:, output_labels].to_numpy().copy(), dtype=torch.float32
        )

        return X, y

    @property
    def bounds(self) -> torch.Tensor:
        """
        Get the bounds for the optimization based on the data if not specified in the declaration.

        :returns: tensor of shape (2, d) with min and max bounds for each feature
        """
        if self._bounds is None:
            bds_max = self.df.loc[:, self.input_labels].max().to_list()
            bds_min = self.df.loc[:, self.input_labels].min().to_list()
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
        Add new experimental data to the dataset.

        :param new_data: New data to add (pd.Series or pd.DataFrame)
        :returns: None. Modifies self.df in-place.
        """

        if isinstance(new_data, pd.Series):
            new_data = new_data.to_frame().T

        if self.model in {PhModel, MultitaskGPhysModel}:
            for label in MODELED_QUANTITIES:
                if (
                    label in new_data.columns
                    and not new_data[label].between(0, 1).all()
                ):
                    raise ValueError(
                        f"PhModel output '{label}' must be a fraction in [0, 1], got values outside this range. "
                    )

        self.df = pd.concat([self.df, new_data], axis=0)

    def get_predictor(self) -> Tuple[torch.nn.Module | BoTorchGP, pd.DataFrame]:
        """
        Train and return the new predictor based on the new data.

        :returns: (model, stats) tuple where model is the trained predictor and stats is a DataFrame with training metrics.
        """
        if self.config.get("torch_seed") is not None:
            torch.manual_seed(self.config["torch_seed"])

        X, y = self._get_data_tensors(update_stats=True)

        system_phase = self.config.get("system_phase", None)

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
                config=self.config,
                n_inputs=len(self.input_labels),
                system_phase=system_phase,
                means=self._means,
                stds=self._stds,
                output_labels=self.output_labels,
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
                ph_frac=self.config.get("ph_frac", 0.5),
            )

        elif self.model == "BoTorch-GP":
            # Vanilla BoTorch MultiTaskGP baseline (BoTorch defaults, fit_gpytorch_mll).
            stats, _, model, likelihood = train_BoTorch_GP_model(
                X,
                y,
                DNAME=self.output_dir,
                i=self.i,
            )

        # Handle ensemble models (MLP and Ph)
        else:
            if self.model == PhModel:

                model_factory = lambda: PhModel(
                    config=self.config,
                    dropout=self.config.get("dropout", 0.1),
                    n_inputs=len(self.input_labels),
                    system_phase=system_phase,
                    means=self._means,
                    stds=self._stds,
                    output_labels=self.output_labels,
                )

            elif self.model == MLPModel:
                # MLP model with explicit input/output sizes
                n_in = len(self.input_labels)
                n_out = len(self.output_labels)
                model_factory = lambda: MLPModel(
                    n_inputs=n_in,
                    n_outputs=n_out,
                    dropout=self.config.get("dropout", 0.1),
                )

            stats, model = train_model_ens(
                X,
                y,
                model_factory,
                DNAME=self.output_dir,
                i=self.i,
                num_iter=self.config["num_iter"],
                plot=self.config["make_plots"],
                num_models=self.config.get("num_models", 50),
                batch_fraction=self.config.get("batch_fraction", 0.5),
                variance_scaling=self.config.get("variance_scaling", True),
                per_output_scaling=self.config.get("per_output_scaling", False),
                lr=self.config.get("lr", 1e-3),
                tol=self.config.get("tol", 1e-5),
                early_stopping=self.config.get("early_stopping", False),
            )

        return model, stats

    def acquisition_value(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate the acquisition function (using the cached predictor) at raw-scale points.

        :param x: DataFrame of input feature rows in raw (unnormalized) scale, columns matching self.input_labels
        :returns: DataFrame with a single "acquisition" column, one row per row of x
        """
        if self.predictor is None:
            raise RuntimeError(
                "No predictor available yet; call step() or step_within_data() first."
            )

        target_idx = self.output_labels.index(self.quantity)
        X, _ = self._get_data_tensors(data=x)
        AF = get_acquisition_function(self, self.predictor)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Output shape checks failed!")
            vals = AF(X.unsqueeze(1))
        scores = select_acquisition_target(vals, target_idx)
        return pd.DataFrame(
            {"acquisition": scores.detach().cpu().numpy()}, index=x.index
        )

    def step(
        self,
        new_data: pd.DataFrame,
        bounds: Optional[torch.Tensor] = None,
        _repeat: int = 0,
    ) -> Tuple[torch.Tensor, pd.Series]:
        """
        Perform a step in the optimization process using the new data and bounds.

        :param new_data: New data to be added to the training set. Do not input data that was already given to the object, only new data.
        :param bounds: Optional bounds for the optimization (default: inferred from data)
        :param _repeat: internal counter tracking retries when the proposed candidate duplicates an existing point
        :returns: tuple of (acquisition_function_value, next_experiment_parameters)
        """

        self.update_data(new_data)

        if self.model == "LLM":
            attempt = 0
            self.raw_messages = []  # Reset message history for this step
            error_message = None
            for attempt in range(self.config.get("llm_max_attempts", 3)):
                try:
                    result = llm_suggest(
                        self, "step", bounds=bounds, error_message=error_message
                    )
                    suggestion = {l: result[l] for l in self.input_labels}

                except Exception as e:
                    if attempt + 1 == self.config.get("llm_max_attempts", 3):
                        print("LLM raw messages:")
                        print(self.raw_messages)
                        raise
                    else:
                        print(
                            f"LLM call failed with error: {e}. Retrying (attempt {attempt + 1})..."
                        )
                        error_message = repr(e)
                        continue

                reason = result.pop("reason", None)

                if (
                    np.isclose(
                        pd.Series(suggestion),
                        self.df.loc[:, self.input_labels],
                        rtol=self.config.get("similarity_tolerance", 1e-5),
                    )
                    .all(axis=1)
                    .any()
                ):
                    print(
                        "LLM suggested a point that is already in the dataset. Retrying..."
                    )
                    error_message = "Your suggested experiment already exists in the dataset or is very close to an existing one."
                    continue
                break

            self.llm_history.append(
                {"step": self.i, "suggestion": suggestion, "reason": reason}
            )
            if reason:
                print(f"LLM reason: {reason}")
            self.i += 1
            return None, pd.Series(suggestion)

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
        except torch._C._LinAlgError as error:
            check_random_fallback(
                self.config,
                f"Carbon Driver could not fit the optimization model; the training data may be underdetermined (duplicate conditions, too little variation, or too few observations). ({error})",
            )
            return None, random_candidate(self.input_labels, raw_bounds)
        except RuntimeError as e:
            # Handle gpytorch ExactGP runtime error when model is called with inputs
            # that don't exactly match the stored training inputs (raised in debug mode).
            msg = str(e)
            if (
                "You must train on the training inputs" in msg
                or "train_inputs cannot be None" in msg
            ):
                check_random_fallback(
                    self.config,
                    f"Carbon Driver could not fit the optimization model; the training data may be underdetermined (duplicate conditions, too little variation, or too few observations). ({e})",
                )
                return None, random_candidate(self.input_labels, raw_bounds)
            else:
                # Unknown runtime error: re-raise so we don't silently swallow unrelated failures
                raise
        self.predictor = predictor

        AF = get_acquisition_function(self, predictor)

        # Select the output column index for the quantity by name from the last two columns

        try:
            target_idx = self.output_labels.index(self.quantity)
        except ValueError:
            raise ValueError(
                f"Quantity '{self.quantity}' not found in output columns {self.output_labels}"
            )
        # print(f"[step] optimizing target column index (target_idx): {target_idx} for quantity '{self.quantity}'")

        means, stds = (
            torch.tensor(
                self._means[self.input_labels].to_numpy().copy()
            ),  # Will be 0 if not normalized
            torch.tensor(
                self._stds[self.input_labels].to_numpy().copy()
            ),  # Will be 1 if not normalized
        )  # feature-only stats

        bounds_norm = (raw_bounds - means) / stds

        # print(f"[step] normalized bounds min: {bounds_norm[0].tolist()} max: {bounds_norm[1].tolist()}")
        opt_bounds = bounds_norm.float()

        def AF_q(x):
            return select_acquisition_target(AF(x), target_idx)

        next_experiment, _ = optimize_acqf(
            acq_function=AF_q,
            bounds=opt_bounds,
            q=1,
            num_restarts=self.config.get("num_restarts", 100),
            raw_samples=self.config.get("raw_samples", 1000),
            options={"ftol": 1e-12, "factr": None},  # See note on next line
        )
        # factr sets the same value as ftol, but needs to be set explicitely
        # to None when ftol is set (which is weird)

        # Denormalize the candidate, mean=0 and std=1 if not normalized.
        x_candidate = next_experiment * stds + means

        if _repeat == 0:
            self.i += 1

        # Evaluate AF at the (normalized) next point for returning EI value
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Output shape checks failed!")
            ei_val = AF_q(next_experiment)

        candidate = pd.Series(
            x_candidate.detach().cpu().numpy().flatten(), index=self.input_labels
        )

        # If the candidate duplicates an existing point, feed the matching point(s)
        # back in as "new" data (data augmentation) and retrain, up to
        # max_repeats_before_random times, then fall back to a random candidate.
        duplicate_mask = np.isclose(
            candidate,
            self.df.loc[:, self.input_labels],
            rtol=self.config.get("similarity_tolerance", 1e-5),
        ).all(axis=1)
        if duplicate_mask.any():
            max_repeats = self.config.get("max_repeats_before_random", 3)
            if _repeat < max_repeats:
                warnings.warn(
                    f"Candidate duplicates an existing point (retry {_repeat + 1}/{max_repeats}); "
                    "augmenting the dataset with the matching point(s) and retraining."
                )
                return self.step(
                    self.df.loc[duplicate_mask], bounds=bounds, _repeat=_repeat + 1
                )
            check_random_fallback(
                self.config,
                f"Candidate still duplicates an existing point after {max_repeats} retries.",
            )
            ei_val = None
            candidate = random_candidate(self.input_labels, raw_bounds)

        if self.config["make_plots"] and len(self.input_labels) == 2:
            plot_2d_debug(self, predictor, AF_q, raw_bounds, candidate)

        return ei_val, candidate

    def predict(self, x: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate the cached predictor (from the last step()/step_within_data() call).

        :param x: DataFrame of input feature rows in raw (unnormalized) scale, columns matching self.input_labels
        :returns: (mean, std) DataFrames with columns self.output_labels, one row per row of x
        """

        if not isinstance(x, pd.DataFrame):
            # This makes sure X will always be of size (batch, features) otherwise the squeeze below will fail.
            raise TypeError("Input x must be a pandas DataFrame.")

        if self.predictor is None:
            raise RuntimeError(
                "No predictor available yet; call step() or step_within_data() first."
            )

        X, _ = self._get_data_tensors(data=x)

        posterior = self.predictor.posterior(X)
        pred_mean = posterior.mean.squeeze(1).detach().numpy()
        pred_std = posterior.variance.sqrt().squeeze(1).detach().numpy()

        output_means = self._means[self.output_labels].to_numpy(dtype=float)
        output_stds = self._stds[self.output_labels].to_numpy(dtype=float)
        pred_mean = pred_mean * output_stds + output_means
        pred_std = pred_std * output_stds
        return (
            pd.DataFrame(pred_mean, columns=self.output_labels, index=x.index),
            pd.DataFrame(pred_std, columns=self.output_labels, index=x.index),
        )

    def step_within_data(
        self,
        new_data: pd.DataFrame,
        possible_data: pd.DataFrame,
        return_metrics: bool = False,
        _repeat: int = 0,
    ) -> Tuple[float, int] | Tuple[float, int, dict]:
        """
        Selects the best next point from a set of candidates (possible_data) considering the new data (new_data) and the existing data (if any).

        :param new_data: New data to be added to the training set. Do not input data that was already given to the object, only new data.
        :param possible_data: DataFrame of candidate points to select from
        :param return_metrics: whether to return training metrics (nll, loss)
        :param _repeat: internal counter tracking retries when the selected point duplicates an existing point
        :returns: (best_ei_value, best_index) or (best_ei_value, best_index, metrics) if return_metrics=True
        """

        self.update_data(new_data)

        if any([(label in possible_data.columns) for label in self.output_labels]):
            print(
                "Warning: possible_data contains output columns. Are you sure you input the right data? Labels will be ignored."
            )
            possible_data = possible_data.drop(
                columns=self.output_labels, errors="ignore"
            )

        if self.model == "LLM":
            self.raw_messages = []  # Reset message history for this step
            error_message = None

            for attempt in range(self.config.get("llm_max_attempts", 3)):
                try:
                    result = llm_suggest(
                        self,
                        "step_within_data",
                        possible_data=possible_data,
                        error_message=error_message,
                    )
                    best_df_index = result["index"]
                    if best_df_index not in possible_data.index:
                        raise KeyError(
                            f"Index {best_df_index!r} is not in possible_data.index. "
                            f"Valid indices are {possible_data.index.tolist()}."
                        )
                except Exception as error:
                    if attempt + 1 == self.config.get("llm_max_attempts", 3):
                        print("LLM raw messages:")
                        print(self.raw_messages)
                        raise
                    print(
                        f"LLM call failed with error: {error}. "
                        f"Retrying (attempt {attempt + 1})..."
                    )
                    error_message = repr(error)
                    continue

                reason = result.get("reason")
                break
            self.llm_history.append(
                {"step": self.i, "suggestion": best_df_index, "reason": reason}
            )
            if reason:
                print(f"LLM reason: {reason}")
            self.i += 1
            if return_metrics:
                return None, best_df_index, {}
            return None, best_df_index

        try:
            predictor, stats = self.get_predictor()
        except torch._C._LinAlgError as error:
            check_random_fallback(
                self.config,
                f"Carbon Driver could not fit the optimization model; the training data may be underdetermined (duplicate conditions, too little variation, or too few observations). ({error})",
            )
            random_idx = possible_data.index[int(torch.randint(len(possible_data) - 1, ()))]
            return (
                (torch.nan, random_idx, {})
                if return_metrics
                else (torch.nan, random_idx)
            )
        except RuntimeError as e:
            msg = str(e)
            if (
                "You must train on the training inputs" in msg
                or "train_inputs cannot be None" in msg
                or "cholesky_cpu" in msg
            ):
                check_random_fallback(
                    self.config,
                    f"Carbon Driver could not fit the optimization model; the training data may be underdetermined (duplicate conditions, too little variation, or too few observations). ({e})",
                )
                random_idx = possible_data.index[int(torch.randint(len(possible_data) - 1, ()))]
                return (
                    (torch.nan, random_idx, {})
                    if return_metrics
                    else (torch.nan, random_idx)
                )
            else:
                raise

        self.predictor = predictor

        X, _ = self._get_data_tensors(data=possible_data)

        AF = get_acquisition_function(self, predictor)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Output shape checks failed!")
            scores = AF(X.unsqueeze(1))

        if isinstance(scores, torch.Tensor) and (
            scores.dim() == 0
            or (scores.dim() == 1 and scores.shape[0] != len(possible_data))
        ):
            scores = scores.unsqueeze(0)

        if _repeat == 0:
            self.i += 1

        target_idx = self.output_labels.index(self.quantity)

        if isinstance(scores, torch.Tensor):
            if scores.dim() == 1:
                target_scores = scores  # already one score per candidate
            else:
                if scores.shape[1] <= target_idx:
                    raise RuntimeError(
                        f"AF scores shape {tuple(scores.shape)} has no column {target_idx}"
                    )
                target_scores = scores[:, target_idx]
        else:
            raise RuntimeError("AF returned non-tensor scores, expected torch.Tensor")
        print(f"Target scores :")
        ts = possible_data.copy()
        ts["AF Score"] = target_scores.tolist()
        print(ts)
        best_idx = int(target_scores.argmax().item())
        best_df_index = possible_data.iloc[best_idx].name
        best_ei = float(target_scores[best_idx].item())

        duplicate_mask = np.isclose(
            possible_data.loc[best_df_index, self.input_labels].astype(float),
            self.df.loc[:, self.input_labels].astype(float),
            rtol=1e-9,
        ).all(axis=1)
        if duplicate_mask.any():
            max_repeats = self.config.get("max_repeats_before_random", 3)
            if _repeat < max_repeats:
                warnings.warn(
                    f"Chose a candidate already in the dataset (retry {_repeat + 1}/{max_repeats}); "
                    "augmenting the dataset and retraining."
                )
                return self.step_within_data(
                    self.df.loc[duplicate_mask],
                    possible_data,
                    return_metrics=return_metrics,
                    _repeat=_repeat + 1,
                )
            check_random_fallback(
                self.config,
                f"Still chose a candidate already in the dataset after {max_repeats} retries.",
            )
            random_idx = possible_data.index[int(torch.randint(len(possible_data) - 1, ()))]
            return (
                (torch.nan, random_idx, {})
                if return_metrics
                else (torch.nan, random_idx)
            )

        if self.config["make_plots"] and len(self.input_labels) == 2:
            all_points = pd.concat(
                [self.df[self.input_labels], possible_data[self.input_labels]]
            )
            raw_bounds = torch.tensor(
                [all_points.min().to_list(), all_points.max().to_list()],
                dtype=torch.float32,
            )
            AF_q = lambda x: select_acquisition_target(AF(x), target_idx)
            plot_2d_debug(
                self,
                predictor,
                AF_q,
                raw_bounds,
                possible_data.loc[best_df_index],
                possible_data,
            )

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
            loss_vals = stats["loss"].dropna()
            if len(loss_vals) > 0:
                metrics["loss"] = float(loss_vals.iloc[-1])
            else:
                metrics["loss"] = np.nan
        else:
            metrics["loss"] = np.nan

        if return_metrics:
            return best_ei, best_df_index, metrics
        else:
            return best_ei, best_df_index
