from .models import PhModel, MLPModel, MultitaskGPModel, BoTorchGP, MultitaskGPhysModel
from .train import train_model_ens, train_GP_model, train_GP_Ph_model
from .loaders import feature_stats
from .config import default_config
from .domains import PhysicsDomain
import pandas as pd
import torch
import numpy as np
import os, json
from typing import Tuple, Optional
from botorch import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement, ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound
from botorch.models.gp_regression import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.acquisition.objective import ScalarizedPosteriorTransform
import warnings
import gpytorch

SUPPORTED_AFs = ["EI", "logEI", "PI", "UCB"]


class OptimizerTrainingError(RuntimeError):
    """Raised when a model cannot be fitted safely enough to recommend a point."""


def _candidate_scores(
    scores: torch.Tensor, target_idx: int, num_candidates: int
) -> torch.Tensor:
    """Return one acquisition score per candidate without losing its axis."""
    if not isinstance(scores, torch.Tensor):
        raise RuntimeError("AF returned non-tensor scores, expected torch.Tensor")
    if target_idx < 0:
        raise RuntimeError(f"target_idx must be non-negative, received {target_idx}")

    if scores.ndim == 0:
        if num_candidates == 1:
            return scores.reshape(1)
        raise RuntimeError(
            f"AF returned one scalar score for {num_candidates} candidates; "
            "expected one score per candidate"
        )
    if scores.ndim == 1:
        if scores.shape[0] != num_candidates:
            raise RuntimeError(
                f"AF scores shape {tuple(scores.shape)} does not match "
                f"{num_candidates} candidates"
            )
        return scores
    if scores.ndim == 2:
        if scores.shape[0] != num_candidates:
            raise RuntimeError(
                f"AF scores shape {tuple(scores.shape)} has the candidate axis in "
                "the wrong position; expected (N,) or (N, M)."
            )
        if scores.shape[1] == 0:
            raise RuntimeError(f"AF scores shape {tuple(scores.shape)} has no score columns")
        if scores.shape[1] == 1:
            return scores[:, 0]
        if target_idx >= scores.shape[1]:
            raise RuntimeError(
                f"AF scores shape {tuple(scores.shape)} has no output column "
                f"{target_idx}"
            )
        return scores[:, target_idx]
    raise RuntimeError(
        f"AF scores must have shape (N,) or (N, M), received {tuple(scores.shape)}"
    )


def _canonical_label(label: str) -> str:
    return "".join(ch for ch in str(label).lower() if ch.isalnum())


class PhysicsOutputAdapter(torch.nn.Module):
    """Select and optionally standardize physics-model output columns."""

    def __init__(
        self,
        model: torch.nn.Module,
        output_indices: list[int],
        output_means: np.ndarray | None = None,
        output_stds: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.output_indices = output_indices
        self.register_buffer(
            "output_means",
            None if output_means is None else torch.as_tensor(output_means, dtype=torch.float32),
        )
        self.register_buffer(
            "output_stds",
            None if output_stds is None else torch.as_tensor(output_stds, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)[..., self.output_indices]
        if self.output_means is not None and self.output_stds is not None:
            output = (output - self.output_means) / self.output_stds
        return output


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
        self.last_prediction_means = None
        self.last_prediction_stds = None

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

    def _physical_output_indices(self) -> list[int]:
        if self.config.get("dataset") == "bicarb":
            physical_outputs = ["FE_CO", "CO2 utilization"]
        else:
            physical_outputs = ["FE (Eth)", "FE (CO)"]

        canonical_outputs = {
            _canonical_label(label): idx
            for idx, label in enumerate(physical_outputs)
        }
        requested_indices = []
        missing = []
        for label in self.output_labels:
            canonical_label = _canonical_label(label)
            if canonical_label in canonical_outputs:
                requested_indices.append(canonical_outputs[canonical_label])
            else:
                missing.append(label)

        if missing:
            raise ValueError(
                "Physics-based Carbon Driver models can only predict "
                f"{physical_outputs}. Requested unsupported objective(s): {missing}."
            )

        return requested_indices

    def _make_physics_model(self, system_phase: str, dropout: float = 0.1) -> torch.nn.Module:
        model = PhModel(
            config=self.config,
            dropout=dropout,
            n_inputs=len(self.input_labels),
            system_phase=system_phase,
            means=self._means,
            stds=self._stds,
        )
        output_indices = self._physical_output_indices()
        normalize_outputs = self.config.get("normalize_outputs", False)
        if output_indices != list(range(2)) or normalize_outputs:
            output_means = None
            output_stds = None
            if normalize_outputs:
                output_means = self._means[self.output_labels].to_numpy(dtype=float)
                output_stds = self._stds[self.output_labels].to_numpy(dtype=float)
            return PhysicsOutputAdapter(
                model,
                output_indices,
                output_means=output_means,
                output_stds=output_stds,
            )
        return model

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

        X = torch.tensor(
            np.ascontiguousarray(df_clean.loc[:, self.input_labels].to_numpy()),
            dtype=torch.float32,
        )
        y = torch.tensor(
            np.ascontiguousarray(df_clean.loc[:, output_labels].to_numpy()),
            dtype=torch.float32,
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
        Add new experimental data to the dataset (and sort by 'triplet' for now).

        :param new_data: New data to add (pd.Series or pd.DataFrame)
        :returns: None. Modifies self.df in-place.
        """

        if isinstance(new_data, pd.Series):
            new_data = new_data.to_frame().T

        self.df = pd.concat([self.df, new_data], axis=0)

        if "triplet" in self.df.columns:
            self.df.sort_values(by="triplet", inplace=True)

    def get_predictor(self) -> Tuple[torch.nn.Module | BoTorchGP, pd.DataFrame]:
        """
        Train and return the new predictor based on the new data.

        :returns: (model, stats) tuple where model is the trained predictor and stats is a DataFrame with training metrics.
        """
        if self.model in {PhModel, MultitaskGPhysModel}:
            self._apply_configured_physics_seed()

        X, y = self._get_data_tensors(update_stats=True)

        system_phase = self.config.get("system_phase") or ("liquid" if self.config.get("dataset") == "bicarb" else "gas")

        # Special handling for GP and GP+Ph models: these use gpytorch training functions
        # (they are not compatible with the ensemble training pipeline used for MLP/Ph).
        if self.model == MultitaskGPModel:
            if len(self.output_labels) == 1:
                model = SingleTaskGP(X.double(), y.double())
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)
                stats = pd.DataFrame(
                    {"loss": [np.nan], "val_loss": [np.nan], "nll": [np.nan]},
                    index=pd.Index([0], name="step"),
                )
                return model, stats

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

            ph_model_constructor = lambda: self._make_physics_model(system_phase)

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

                model_factory = lambda: self._make_physics_model(system_phase, dropout=0.0)

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

    def _apply_configured_physics_seed(self) -> Optional[int]:
        """Seed Physics and GP+Physics fitting from the driver configuration."""
        configured_seed = self.config.get("torch_seed")
        if configured_seed is None:
            return None
        if isinstance(configured_seed, bool):
            raise ValueError("torch_seed must be an integer, not a boolean.")
        try:
            seed = int(configured_seed)
        except (TypeError, ValueError) as error:
            raise ValueError("torch_seed must be an integer.") from error
        if isinstance(configured_seed, float) and not configured_seed.is_integer():
            raise ValueError("torch_seed must be an integer.")

        try:
            torch.manual_seed(seed)
        except RuntimeError as error:
            raise ValueError(f"torch_seed {seed} is outside PyTorch's valid range.") from error
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

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
        posterior_transform = None
        is_gp_wrapper = isinstance(predictor, BoTorchGP) or hasattr(predictor, "likelihood")
        if len(self.output_labels) > 1 and is_gp_wrapper:
            weights = torch.zeros(len(self.output_labels), dtype=torch.float32)
            weights[target_idx] = 1.0
            posterior_transform = ScalarizedPosteriorTransform(weights=weights)

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
                    posterior_transform=posterior_transform,
                )
        if self.aquisition == "logEI":
            return LogExpectedImprovement(
                predictor,
                best_f=best_f,
                maximize=self.maximize,
                posterior_transform=posterior_transform,
            )
        if self.aquisition == "PI":
            return ProbabilityOfImprovement(
                predictor,
                best_f=best_f,
                maximize=self.maximize,
                posterior_transform=posterior_transform,
            )
        if self.aquisition == "UCB":
            beta = self.config.get("UCB_beta", 1.0)
            return UpperConfidenceBound(
                predictor,
                beta=beta,
                maximize=self.maximize,
                posterior_transform=posterior_transform,
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
            prediction_example = {label: 0.0 for label in self.output_labels}
            return (
                f"{history_str}"
                f"{data_str}"
                f"Input parameters and their allowed ranges:\n{bounds_str}\n\n"
                f"Suggest the next experiment that will lead to {direction}ing '{self.quantity}' "
                f"in the fewest number of experiments.\n"
                f"Respond with ONLY a JSON object with:\n"
                f'  parameter names mapped to their suggested values\n'
                f'  "predicted_objectives": estimated values for every objective\n'
                f'  "reason": brief explanation\n'
                f'Example: {json.dumps({**{l: 0.0 for l in self.input_labels}, "predicted_objectives": prediction_example, "reason": "..."})}'
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
        """Parse a JSON object even when the provider adds fences or brief prose."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("The LLM returned an empty response.")

        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].strip().lower() in {"```", "```json"}:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as original_error:
            decoder = json.JSONDecoder()
            result = None
            for position, character in enumerate(cleaned):
                if character != "{":
                    continue
                try:
                    result, _ = decoder.raw_decode(cleaned[position:])
                    break
                except json.JSONDecodeError:
                    continue
            if result is None:
                preview = cleaned[:200].replace("\n", " ")
                raise ValueError(
                    f"The LLM response was not valid JSON. Response began with: {preview!r}"
                ) from original_error

        if not isinstance(result, dict):
            raise ValueError(
                f"The LLM must return a JSON object, but returned {type(result).__name__}."
            )
        return result

    @staticmethod
    def _gemini_empty_response_details(response) -> str:
        """Summarize Gemini metadata without exposing request credentials."""
        details = []
        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback is not None:
            block_reason = getattr(prompt_feedback, "block_reason", None)
            if block_reason:
                details.append(f"prompt block reason: {block_reason}")

        for candidate in getattr(response, "candidates", None) or []:
            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason:
                details.append(f"finish reason: {finish_reason}")
        return "; ".join(dict.fromkeys(details))

    def _llm_suggest(
        self,
        mode: str,
        bounds: Optional[torch.Tensor] = None,
        possible_data: Optional[pd.DataFrame] = None,
        error_message: Optional[str] = None,
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
        if error_message is not None:
            user += (
                "\n\nYour previous response could not be used. Correct this error and "
                f"answer the original request again: {error_message}"
            )

        api_key = self.config.get("llm_api_key", None)

        if api == "gemini":
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=self.config["llm_model"],
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            try:
                text = response.text
            except (AttributeError, ValueError):
                text = None
            if not text or not text.strip():
                details = self._gemini_empty_response_details(response)
                suffix = f" ({details})" if details else ""
                raise ValueError(
                    "Gemini returned no response text"
                    f"{suffix}. Check the selected model, safety feedback, quota, and prompt size."
                )

        elif api == "openai":
            from openai import OpenAI
            response = OpenAI(api_key=api_key).chat.completions.create(
                model=self.config["llm_model"],
                messages=[{"role": "system", "content": system}] + self.raw_messages + [{"role": "user",   "content": user}],
            )
            text = response.choices[0].message.content

        elif api == "claude":
            import anthropic

            response = anthropic.Anthropic(api_key=api_key).messages.create(
                model=self.config["llm_model"],
                max_tokens=self.config.get("llm_max_tokens", 1024),
                system=system,
                messages=self.raw_messages + [{"role": "user", "content": user}],
            )
            text_parts = []
            structured_result = None
            for block in response.content:
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    block_text = getattr(block, "text", None)
                    if block_text:
                        text_parts.append(block_text)
                elif block_type == "tool_use":
                    block_input = getattr(block, "input", None)
                    if isinstance(block_input, dict):
                        structured_result = block_input

            if structured_result is not None:
                text = json.dumps(structured_result)
            else:
                text = "\n".join(text_parts).strip()
                if not text:
                    stop_reason = getattr(response, "stop_reason", None)
                    suffix = f" (stop reason: {stop_reason})" if stop_reason else ""
                    raise ValueError(
                        "Claude returned no text or structured JSON"
                        f"{suffix}. Check the selected model, safety response, quota, and prompt size."
                    )

        else:
            raise ValueError(f"Unsupported llm_api '{api}'. Choose 'gemini', 'openai', or 'claude'.")

        self.raw_messages.extend(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": text},
            ]
        )

        return self._read_response(text)

    def step(
        self,
        new_data: pd.DataFrame,
        bounds: Optional[torch.Tensor] = None,
        *,
        fixed_features: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, pd.Series]:
        """
        Perform a step in the optimization process using the new data and bounds.

        :param new_data: New data to be added to the training set. Do not input data that was alerady given to the object, only new data.
        :param bounds: Optional bounds for the optimization (default: inferred from data)
        :returns: tuple of (acquisition_function_value, next_experiment_parameters)
        """
        self.update_data(new_data)

        if self.model == "LLM":
            attempt = 0
            self.raw_messages = []  # Reset message history for this step
            error_message = None
            for attempt in range(self.config.get("llm_max_attempts", 3)):
                try:
                    result = self._llm_suggest("step", bounds=bounds, error_message=error_message)
                    predicted_objectives = result.pop("predicted_objectives", None)
                    suggestion = {l: result[l] for l in self.input_labels}

                except Exception as e:
                    if attempt + 1 == self.config.get("llm_max_attempts", 3):
                        print("LLM raw messages:")
                        print(self.raw_messages)
                        raise
                    else:
                        print(f"LLM call failed with error: {e}. Retrying (attempt {attempt + 1})...")
                        error_message = repr(e)
                        continue                        

                reason = result.pop("reason", None)
                    
                if np.isclose(pd.Series(suggestion), self.df.loc[:, self.input_labels], rtol=self.config.get("similarity_tolerance", 1e-5)).all(axis=1).any():
                    print("LLM suggested a point that is already in the dataset. Retrying...")
                    error_message = "Your suggested experiment already exists in the dataset or is very close to an existing one."
                    continue
                break
                
            self.llm_history.append({"step": self.i, "suggestion": suggestion, "reason": reason})
            if isinstance(predicted_objectives, dict):
                self.last_prediction_means = {
                    label: float(predicted_objectives[label])
                    for label in self.output_labels
                    if label in predicted_objectives
                }
            else:
                self.last_prediction_means = None
            self.last_prediction_stds = None
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

        domain = None
        if self.model in {PhModel, MultitaskGPhysModel}:
            domain = PhysicsDomain(self.input_labels, raw_bounds, self.config, fixed_features)
        elif fixed_features:
            raise ValueError("fixed_features is currently supported only for physics models.")

        try:
            predictor, stats = self.get_predictor()
        except torch._C._LinAlgError as error:
            raise OptimizerTrainingError(
                "Carbon Driver could not fit the optimization model because the "
                "training data produced an unstable linear-algebra system. No "
                "recommendation was generated. Review duplicate conditions, input "
                "variation, and the amount of training data before retrying."
            ) from error
        except RuntimeError as e:
            # Handle gpytorch ExactGP runtime error when model is called with inputs
            # that don't exactly match the stored training inputs (raised in debug mode).
            msg = str(e)
            if (
                "You must train on the training inputs" in msg
                or "train_inputs cannot be None" in msg
            ):
                raise OptimizerTrainingError(
                    "Carbon Driver could not fit the Gaussian Process because its "
                    "stored training inputs are inconsistent with the current data. "
                    "No recommendation was generated. Rebuild the model from the "
                    "active campaign observations before retrying."
                ) from e
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

        free_indices = domain.free_indices if domain else list(range(len(self.input_labels)))
        free_means, free_stds = means[free_indices], stds[free_indices]
        free_bounds = domain.free_bounds if domain else raw_bounds
        bounds_norm = (free_bounds - free_means) / free_stds
        
        # print(f"[step] normalized bounds min: {bounds_norm[0].tolist()} max: {bounds_norm[1].tolist()}")
        opt_bounds = bounds_norm.float()

        def full_normalized(x):
            if domain is None:
                return x
            raw = x * free_stds.to(x) + free_means.to(x)
            return (domain.expand(raw) - means.to(x)) / stds.to(x)

        def AF_q(x):
            vals = AF(full_normalized(x))
            count = x.shape[0] if x.ndim >= 3 else 1
            return _candidate_scores(vals, target_idx, count)

        next_experiment, _ = optimize_acqf(
            acq_function=AF_q,
            bounds=opt_bounds,
            q=1,
            num_restarts=20,
            raw_samples=30,
            options={},
        )
        
        # Denormalize the candidate, mean=0 and std=1 if not normalized. 
        full_candidate = full_normalized(next_experiment)
        x_candidate = (
            domain.expand(next_experiment * free_stds + free_means)
            if domain else full_candidate * stds + means
        )

        try:
            with torch.no_grad():
                posterior = predictor.posterior(full_candidate)
                posterior_mean = posterior.mean
                posterior_std = posterior.variance.clamp_min(0).sqrt()
            prediction_values = posterior_mean.detach().cpu().reshape(-1).tolist()
            uncertainty_values = posterior_std.detach().cpu().reshape(-1).tolist()
            if len(prediction_values) < len(self.output_labels):
                raise ValueError("Predictor returned fewer values than configured outputs.")

            output_means = self._means[self.output_labels].to_numpy(dtype=float)
            output_stds = self._stds[self.output_labels].to_numpy(dtype=float)
            prediction_values = np.asarray(
                prediction_values[-len(self.output_labels):], dtype=float
            )
            prediction_values = prediction_values * output_stds + output_means
            uncertainty_values = np.asarray(
                uncertainty_values[-len(self.output_labels):], dtype=float
            ) * output_stds
            self.last_prediction_means = dict(
                zip(self.output_labels, prediction_values.tolist())
            )
            self.last_prediction_stds = dict(
                zip(self.output_labels, uncertainty_values.tolist())
            )
        except (AttributeError, RuntimeError, ValueError):
            self.last_prediction_means = None
            self.last_prediction_stds = None

        self.i += 1

        # Evaluate AF at the (normalized) next point for returning EI value
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Output shape checks failed!")
            ei_val = AF_q(next_experiment)

        def evaluate_raw(free_raw):
            normalized = (free_raw - free_means.to(free_raw)) / free_stds.to(free_raw)
            return AF_q(normalized.float())

        # Kept in memory only: callers may generate slices without refitting the model.
        self.acquisition_context = {
            "evaluate": evaluate_raw,
            "labels": domain.free_labels if domain else list(self.input_labels),
            "bounds": free_bounds.detach().cpu(),
            "reference": (next_experiment * free_stds + free_means).detach().cpu().reshape(-1),
            "constraints": domain.metadata() if domain else {},
            "acquisition": self.aquisition,
        }

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

        if possible_data.empty:
            raise ValueError("possible_data must contain at least one candidate.")

        self.update_data(new_data)

        if any([(label in possible_data.columns) for label in self.output_labels]):
            print("Warning: possible_data contains output columns. Are you sure you input the right data? Labels will be ignored.")
            possible_data = possible_data.drop(columns=self.output_labels, errors="ignore")

        if self.model == "LLM":
            self.raw_messages = []  # Reset message history for this step
            error_message = None

            for attempt in range(self.config.get("llm_max_attempts", 3)):
                try:
                    result = self._llm_suggest(
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
            self.llm_history.append({"step": self.i, "suggestion": best_df_index, "reason": reason})
            if reason:
                print(f"LLM reason: {reason}")
            self.i += 1
            if return_metrics:
                return None, best_df_index, {}
            return None, best_df_index

        try:
            predictor, stats = self.get_predictor()
        except torch._C._LinAlgError as error:
            raise OptimizerTrainingError(
                "Carbon Driver could not fit the optimization model because the "
                "training data produced an unstable linear-algebra system. No "
                "candidate was selected. Review duplicate conditions, input "
                "variation, and the amount of training data before retrying."
            ) from error
        except RuntimeError as e:
            msg = str(e)
            if (
                "You must train on the training inputs" in msg
                or "train_inputs cannot be None" in msg
                or "cholesky_cpu" in msg
            ):
                raise OptimizerTrainingError(
                    "Carbon Driver could not fit the Gaussian Process for candidate "
                    "selection. No candidate was selected. Review the active training "
                    "data and rebuild the model before retrying."
                ) from e
            else:
                raise

        X, _ = self._get_data_tensors(data=possible_data)

        AF = self._get_acquisition_function(predictor)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Output shape checks failed!")
            scores = AF(X.unsqueeze(1))

        self.i += 1

        target_idx = self.output_labels.index(self.quantity)
        target_scores = _candidate_scores(
            scores,
            target_idx=target_idx,
            num_candidates=len(possible_data),
        )
        print(f"Target scores : {target_scores.tolist()}")
        best_position = int(target_scores.argmax().item())
        best_df_index = possible_data.iloc[best_position].name
        best_ei = float(target_scores[best_position].item())

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
