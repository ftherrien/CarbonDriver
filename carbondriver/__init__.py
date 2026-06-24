from .models import PhModel, MLPModel, MultitaskGPModel, BoTorchGP, MultitaskGPhysModel
from .train import train_model_ens, train_GP_model, train_GP_Ph_model
from .loaders import feature_stats
from .config import default_config
import pandas as pd
import torch
import numpy as np
import os, json
from typing import Tuple, Optional
from botorch.acquisition.analytic import LogExpectedImprovement, ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.acquisition.objective import ScalarizedPosteriorTransform
import warnings
import gpytorch

SUPPORTED_AFs = ["EI", "logEI", "PI", "UCB"]


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
        elif model_name == "LLM":
            self.model = "LLM"
        else:
            raise ValueError(
                f"Unsupported model_name '{model_name}'. Supported options are 'GP', 'Ph', 'MLP', 'GP+Ph', 'LLM'."
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

        self.llm_history = []  # list of {"step": i, "suggestion": ..., "reason": ...}

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
                self._means.loc[self.input_labels] = df_clean.loc[:,self.input_labels].mean()
                self._stds.loc[self.input_labels] = df_clean.loc[:,self.input_labels].std(ddof=0)
                
            if self.config["normalize_outputs"]:
                self._means.loc[output_labels] = df_clean.loc[:,output_labels].mean()
                self._stds.loc[output_labels] = df_clean.loc[:,output_labels].std(ddof=0)

            if self._stds.min() < 1e-10:
                print(
                    "Note: Some features standard deviation are < 1e-10, normalizing with 1"
                )
                self._stds.loc[self._stds < 1e-10] = 1

        df_clean = (df_clean - self._means) / self._stds

        X = torch.tensor(df_clean.loc[:, self.input_labels].values, dtype=torch.float32)
        y = torch.tensor(df_clean.loc[:, output_labels].values, dtype=torch.float32)

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

        system_phase = self.config.get("system_phase") or ("liquid" if self.config.get("dataset") == "bicarb" else "gas")

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
                stds=self._stds
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
                    config=self.config,
                    dropout=0.0,
                    n_inputs=len(self.input_labels),
                    system_phase=system_phase,
                    means=self._means,
                    stds=self._stds
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

        return model, stats

    def _get_acquisition_function(
        self, predictor: torch.nn.Module
    ) -> ExpectedImprovement | LogExpectedImprovement | ProbabilityOfImprovement:
        """
        Get the acquisition function based on the specified acquisition type.

        :param predictor: trained model for predictions
        :returns: BoTorch acquisition function (EI, logEI, or PI)

        Note: The acquisition function is in normalized space if normalization is enabled, so the expected improvement is not the actual value for example. Also normalizing here just for consistency because y is not actually normalized.
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
        if self.aquisition == "PI":
            return ProbabilityOfImprovement(
                predictor,
                best_f=best_f,
                maximize=self.maximize,
            )
        if self.aquisition == "UCB":
            beta = self.config.get("UCB_beta", 1.0)
            return UpperConfidenceBound(
                predictor,
                beta=beta,
                maximize=self.maximize,
            )
        raise ValueError(f"Unsupported acquisition function: {self.aquisition}")

    def _create_prompt(
        self,
        mode: str,
        bounds: Optional[torch.Tensor] = None,
        possible_data: Optional[pd.DataFrame] = None,
    ) -> str:
        """Build the user prompt from history, data, and task description."""
        direction = "maximize" if self.maximize else "minimize"

        history_str = ""
        if self.llm_history:
            entries = "\n".join(
                f"  Step {e['step']}: suggested {e['suggestion']} — reason: {e['reason']}"
                for e in self.llm_history
            )
            history_str = f"Your previous suggestions and reasoning:\n{entries}\n\n"

        data_str = f"Experimental data collected so far:\n{self.df.to_string()}\n\n"

        if mode == "step":
            raw_bounds = self.bounds if bounds is None else bounds
            bounds_str = "\n".join(
                f"  {label}: [{raw_bounds[0, i].item():.4g}, {raw_bounds[1, i].item():.4g}]"
                for i, label in enumerate(self.input_labels)
            )
            return (
                f"{history_str}"
                f"{data_str}"
                f"Input parameters and their allowed ranges:\n{bounds_str}\n\n"
                f"Suggest the next experiment that will lead to {direction}ing '{self.quantity}' "
                f"in the fewest number of experiments.\n"
                f"Respond with ONLY a JSON object with:\n"
                f'  parameter names mapped to their suggested values\n'
                f'  "reason": brief explanation\n'
                f'Example: {json.dumps({**{l: 0.0 for l in self.input_labels}, "reason": "..."})}'
            )
        else:  # step_within_data
            return (
                f"{history_str}"
                f"{data_str}"
                f"Candidate experiments to choose from (index on the left):\n{possible_data.to_string()}\n\n"
                f"Select the candidate that will lead to {direction}ing '{self.quantity}' "
                f"in the fewest number of experiments.\n"
                f"Respond with ONLY a JSON object with:\n"
                f'  "index": index of the best candidate in the dataframe (leftmost column)\n'
                f'  "reason": brief explanation\n'
                f'Example: {{"index": 0, "reason": "..."}}'
            )

    def _read_response(self, text: str) -> dict:
        """Strip markdown code fences from LLM output and parse as JSON."""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())

    def _llm_suggest(
        self,
        mode: str,
        bounds: Optional[torch.Tensor] = None,
        possible_data: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Call the configured LLM API to suggest the next experiment.

        :param mode: "step" for free-form suggestion within bounds, or "step_within_data" to select from candidates
        :param bounds: tensor of shape (2, d) with parameter bounds (used in "step" mode)
        :param possible_data: DataFrame of candidate experiments (used in "step_within_data" mode)
        :returns: parsed JSON dict from the LLM response
        """
        api = self.config.get("llm_api", "gemini")
        system = self.config["llm_experiment_context"]
        user = self._create_prompt(mode, bounds=bounds, possible_data=possible_data)

        api_key = self.config.get("llm_api_key", None)

        if api == "gemini":
            from google import genai
            client = genai.Client(api_key=api_key)
            attempt = 0
            for attempt in range(self.config.get("llm_max_attempts", 3)):
                try:
                    response = client.models.generate_content(
                        model=self.config["llm_model"],
                        contents=[system, user])
                    break
                except Exception as e:
                    if attempt + 1 == self.config.get("llm_max_attempts", 3):
                        raise
                    else:
                        print(f"LLM API call failed (attempt {attempt+1}/{self.config.get('llm_max_attempts', 3)}): {e}")
                    
            text = response.text

        elif api == "openai":
            from openai import OpenAI
            response = OpenAI(api_key=api_key).chat.completions.create(
                model=self.config["llm_model"],
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            text = response.choices[0].message.content

        elif api == "claude":
            from anthropic import Anthropic
            response = Anthropic(api_key=api_key).messages.create(
                model=self.config["llm_model"],
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = response.content[0].text

        else:
            raise ValueError(f"Unsupported llm_api '{api}'. Choose 'gemini', 'openai', or 'claude'.")

        return self._read_response(text)

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

        if self.model == "LLM":
            result = self._llm_suggest("step", bounds=bounds)
            reason = result.pop("reason", None)
            suggestion = {l: result[l] for l in self.input_labels}
            self.llm_history.append({"step": self.i, "suggestion": suggestion, "reason": reason})
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

        means, stds = (
            torch.tensor(self._means[self.input_labels].values), # Will be 0 if not normalized
            torch.tensor(self._stds[self.input_labels].values), # Will be 1 if not normalized
        )  # feature-only stats

        bounds_norm = (raw_bounds - means) / stds
        
        # print(f"[step] normalized bounds min: {bounds_norm[0].tolist()} max: {bounds_norm[1].tolist()}")
        opt_bounds = bounds_norm.float()

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
        
        # Denormalize the candidate, mean=0 and std=1 if not normalized. 
        x_candidate = next_experiment * stds + means

        self.i += 1

        # Evaluate AF at the (normalized) next point for returning EI value
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Output shape checks failed!")
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

        if any([(label in possible_data.columns) for label in self.output_labels]):
            print("Warning: possible_data contains output columns. Are you sure you input the right data? Labels will be ignored.")
            possible_data = possible_data.drop(columns=self.output_labels, errors="ignore")

        if self.model == "LLM":
            result = self._llm_suggest("step_within_data", possible_data=possible_data)
            reason = result.get("reason")
            best_df_index = result["index"]
            self.llm_history.append({"step": self.i, "suggestion": best_df_index, "reason": reason})
            if reason:
                print(f"LLM reason: {reason}")
            self.i += 1
            if return_metrics:
                return None, best_df_index, {}
            return None, best_df_index

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
            warnings.filterwarnings("ignore", message="Output shape checks failed!")
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
        best_df_index = possible_data.iloc[best_idx].name
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
