from typing import Tuple, Optional, Dict, Any, Literal
import torch
import gpytorch
from carbondriver import gde_multi
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultitaskMultivariateNormal
from botorch.models.ensemble import EnsembleModel


class PhModel(torch.nn.Module):
    """
    Model for predicting the Faradaic efficiency of CO and C2H4 on a catalyst.

    The model is a neural network that takes in the following inputs:
    - AgCu Ratio
    - Nafion volume (ul)
    - Sustain volume (ul)
    - Zero_eps_thickness
    - Catalyst mass loading

    The model outputs the Faradaic efficiency of CO and C2H4.

    Values of particle radius r and porosity ε are in
    sensible ranges (40 nm < r < 65 nm, 0.5 < ε < 0.8).
    """

    def __init__(
        self,
        zlt_mu: float,
        zlt_sigma: float,
        current_target: float = 200,
        dropout: float = 0.1,
        ldim: int = 64,
        n_inputs: int = 5,
        zlt_index: int = 3,
        config: Optional[Dict[str, Any]] = None,
        system_phase: Literal["gas", "liquid"] = "gas",
    ) -> None:
        """
        :param zlt_mu: mean of Zero_eps_thickness for normalization
        :param zlt_sigma: std of Zero_eps_thickness for normalization
        :param current_target: target current density for electrode simulation
        :param dropout: dropout probability
        :param ldim: hidden layer dimension
        :param n_inputs: number of input features
        :param zlt_index: index of Zero_eps_thickness in the input vector
        :param config: configuration dict with 'normalize_inputs' flag
        :param system_phase: phase of the electrochemical system ('gas' or 'liquid')
        """
        super().__init__()
        if n_inputs < 1:
            raise ValueError("n_inputs must be >= 1")
        if zlt_index < 0 or zlt_index >= n_inputs:
            raise ValueError("zlt_index must be within [0, n_inputs)")
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, ldim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ldim, ldim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ldim, ldim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ldim, 6),
        )

        erc = gde_multi.electrode_reaction_kinetics | {}
        erc["i_0_CO"] = torch.nn.parameter.Parameter(torch.tensor(erc["i_0_CO"]))
        erc["i_0_C2H4"] = torch.nn.parameter.Parameter(torch.tensor(erc["i_0_C2H4"]))
        erc["i_0_H2b"] = torch.nn.parameter.Parameter(torch.tensor(erc["i_0_H2b"]))
        erc["alpha_CO"] = torch.nn.parameter.Parameter(torch.tensor(erc["alpha_CO"]))
        erc["alpha_C2H4"] = torch.nn.parameter.Parameter(
            torch.tensor(erc["alpha_C2H4"])
        )
        erc["alpha_H2b"] = torch.nn.parameter.Parameter(torch.tensor(erc["alpha_H2b"]))
        self.ph_model = gde_multi.System(
            diffusion_coefficients=gde_multi.diffusion_coefficients,
            salting_out_exponents=gde_multi.salting_out_exponents,
            electrode_reaction_kinetics=erc,
            electrode_reaction_potentials=gde_multi.electrode_reaction_potentials,
            chemical_reaction_rates=gde_multi.chemical_reaction_rates,
            system_phase=system_phase,
        )
        self.softmax = torch.nn.Softmax(dim=1)
        # zero-eps thickness normalization stats
        self.zlt_mu = float(zlt_mu)
        self.zlt_sigma = float(zlt_sigma)
        self.n_inputs = int(n_inputs)
        self.zlt_index = int(zlt_index)
        self.current_target = current_target
        # configuration (normalization status is read from here only)
        self.config = config or {"normalize_inputs": False}
        # persistent forward counter to track how many times forward() was called
        self._forward_counter = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict Faradaic efficiency for given experimental parameters.

        :param x: input features of shape (batch, n_inputs)
        :returns: predicted FE values of shape (batch, 2). Gas: [FE(C2H4), FE(CO)], Liquid/Bicarb: [FE(CO), CO2 utilization]
        """
        if x.shape[-1] != self.n_inputs:
            raise ValueError(
                f"Expected {self.n_inputs} input features, got {x.shape[-1]}"
            )
        # increment and print persistent forward counter
        self._forward_counter += 1
        # columns of x: AgCu Ratio, Naf vol (ul), Sust vol (ul), Zero_eps_thickness, Catalyst mass loading
        latents = self.net(x)
        r = 40e-9 * torch.exp(latents[..., [0]])
        eps = torch.sigmoid(latents[..., [1]])

        # If inputs are normalized, denormalize Zero_eps_thickness (feature index 3)
        if bool(self.config.get("normalize_inputs", False)):
            zlt = (x[..., self.zlt_index] * self.zlt_sigma + self.zlt_mu).view(-1, 1)
        else:
            zlt = x[..., self.zlt_index].view(-1, 1)
        # Prevent division by zero in L calculation
        L = zlt / (1 - eps)
        K_dl_factor = torch.exp(latents[..., [2]])
        thetas = self.softmax(2 * latents[..., 3:])
        # CO activation must not be zero
        theta0 = thetas[..., [0]]
        theta1 = thetas[..., [1]]
        theta2 = thetas[..., [2]]
        thetas = {"CO": theta0, "C2H4": theta1, "H2b": theta2}
        gdl_mass_transfer_coefficient = (
            K_dl_factor
            * self.ph_model.bruggeman(gde_multi.diffusion_coefficients["CO2"], eps)
            / r
        )
        solution = self.ph_model.solve_current(
            i_target=self.current_target,
            eps=eps,
            r=r,
            L=L,
            thetas=thetas,
            gdl_mass_transfer_coeff=gdl_mass_transfer_coefficient,
            grid_size=1000,
            voltage_bounds=(-1.25, 0),
        )

        if self.config.get("dataset") == "bicarb":
            out = torch.cat([solution["fe_co"], solution["co2_utilization"]], dim=-1)
        else:
            out = torch.cat([solution["fe_c2h4"], solution["fe_co"]], dim=-1)
        return out


class MLPModel(torch.nn.Module):
    """Multi-layer perceptron for regression or multitask prediction."""

    def __init__(
        self, n_inputs: int, n_outputs: int = 2, dropout: float = 0.1, ldim: int = 64
    ) -> None:
        """
        :param n_inputs: number of input features
        :param n_outputs: number of output targets (e.g. 1 for single objective, 2 for FE(Eth)/FE(CO), etc.)
        :param dropout: dropout probability
        :param ldim: hidden layer dimension
        """
        super().__init__()
        self.mlp = torch.nn.Sequential(
            # Input dimension is inferred at first forward pass
            torch.nn.Linear(n_inputs, ldim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ldim, ldim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ldim, ldim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ldim, n_outputs),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class EnsPredictor(EnsembleModel):
    """Wraps ensemble models for BoTorch-compatible shape handling."""

    def __init__(self, model_input: torch.nn.Module) -> None:
        """
        :param model_input: ensemble model to wrap
        """
        super().__init__()
        self._num_outputs = 1
        self.model_input = model_input

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Expect X of shape (batch, q, d). We handle q=1 by squeezing.
        if X.dim() == 3 and X.shape[1] == 1:
            X_in = X.squeeze(1)  # (batch, d)
        elif X.dim() == 2:
            X_in = X  # already (batch, d)
        else:
            # Attempt to reshape conservatively: collapse all but last dim into batch
            X_in = X.view(-1, X.shape[-1])

        model_output = self.model_input(X_in)
        # Handle different output dimensions
        if model_output.dim() == 3:
            # Output is [ensemble, batch, features] -> [batch, ensemble, features]
            result = model_output.permute((1, 0, 2)).unsqueeze(2)
        elif model_output.dim() == 4:
            # Output is [ensemble, batch, features, extra] -> [batch, ensemble, features, extra]
            result = model_output.permute((2, 0, 1, 3))
        else:
            # Fallback: return as is
            result = model_output

        return result


class BoTorchGP(GPyTorchModel):
    """Wrapper for GPyTorch models to work with BoTorch acquisition functions."""

    def __init__(self, model: gpytorch.models.ExactGP) -> None:
        """
        :param model: GPyTorch ExactGP model
        """
        super().__init__()
        self.model = model
        self._num_outputs = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] == 1:
            X_in = x.squeeze(1)  # (batch, d)
        elif x.dim() == 2:
            X_in = x  # already (batch, d)
        return self.model(X_in)


class MultitaskGPModel(gpytorch.models.ExactGP):
    """Multitask Gaussian Process for joint prediction of two outputs."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
    ) -> None:
        """
        :param train_x: training inputs of shape (n, d)
        :param train_y: training targets of shape (n, 2)
        :param likelihood: GPyTorch likelihood
        """
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )
        self.num_outputs = 2

    def forward(
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultitaskMultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class MyMean(gpytorch.means.Mean):
    """
    Mean function for physics-informed GP using a neural network model.
    """

    def __init__(
        self, model: Optional[torch.nn.Module] = None, freeze_model: bool = False
    ) -> None:
        """
        :param model: optional model to use as mean function (default: PhModel)
        :param freeze_model: whether to freeze model parameters and remove dropout
        """
        super().__init__()
        if model is not None:
            self.model = model
        else:
            print(
                "No model provided, using default PhModel with placeholder parameters."
            )
            self.model = PhModel(
                zlt_mu_stds=(means["Zero_eps_thickness"], stds["Zero_eps_thickness"]),
                current_target=233,
            )

        if freeze_model:

            def remove_dropout(m: torch.nn.Module):
                for child in m.children():
                    if isinstance(child, torch.nn.Dropout):
                        child.p = 0
                    else:
                        remove_dropout(child)

            for param in self.model.parameters():
                param.requires_grad = False
            remove_dropout(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate mean function.

        :param x: input tensor
        :returns: mean function output squeezed to 1D
        """
        return self.model(x).squeeze()


class MultitaskGPhysModel(gpytorch.models.ExactGP):
    """Combines physics-informed mean (PhModel) with GP covariance."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        model: Optional[torch.nn.Module] = None,
        freeze_model: bool = False,
    ) -> None:
        """
        :param train_x: training inputs of shape (n, d)
        :param train_y: training targets of shape (n, 2)
        :param likelihood: GPyTorch likelihood
        :param model: optional physics-informed model for mean function
        :param freeze_model: whether to freeze model parameters
        """
        super(MultitaskGPhysModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = MyMean(model=model, freeze_model=freeze_model)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )

    def forward(
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultitaskMultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
