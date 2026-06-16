from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Union, Optional
from types import MappingProxyType  # immutable dictionary
from functools import cached_property
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
import torch
import torch.nn.functional

diffusion_coefficients = MappingProxyType(
    {
        "CO2": 1.91e-09,
        "OH": 5.293e-09,
        "CO3": 9.23e-10,
        "HCO3": 1.18e-09,
        "H": 9.311e-09,
        "K": 1.96e-09,
        "CO": 2.03e-09,
        "H2": 4.5e-09,
    }
)  # [m^2/s]

salting_out_exponents = MappingProxyType(
    {
        "h_OH": 6.67e-05,
        "h_CO3": 0.0001251,
        "h_HCO3": 7.95e-05,
        "h_K": 7.5e-05,
        "h_CO2": 0.0,
    }
)  # [m^3/mol]

electrode_reaction_kinetics = MappingProxyType(
    {
        "i_0_CO": 0.00471,  # [A/m^2]
        "i_0_C2H4": 1e-5,  # [A/m^2]
        "i_0_H2a": 0.00979,  # [A/m^2] not used in the calculation
        "i_0_H2b": 1.16e-05,  # [A/m^2]
        "alpha_CO": 0.44,
        "alpha_C2H4": 0.4,
        "alpha_H2a": 0.27,
        "alpha_H2b": 0.36,
    }
)

electrode_reaction_potentials = MappingProxyType(
    {
        "E_0_CO": -0.11,  # [V]
        "E_0_C2H4": 0.09,  # [V]
        "E_0_H2a": 0.0,  # [V]
        "E_0_H2b": 0.0,  # [V]
    }
)

chemical_reaction_rates = MappingProxyType(
    {
        "k1f": 2.23,  # [m^3/(mol s)]
        # 'k1r': 0.000840308, # [1/s]
        # 'k2f': 6000000.0, # [m^3/(mol s)]
        # 'k2r': 345200.0, # [1/s]
        "c_ref": 1000.0,  # [mol/m^3]
    }
)


class ModelData(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        toprint = []
        for k, v in kwargs.items():
            toprint.append(k)
            if isinstance(v, torch.nn.parameter.Parameter):
                self.register_parameter(k, v)
            else:
                self.register_buffer(k, torch.tensor(v))
        self.toprint = toprint

    def __repr__(self):
        return (
            "ModuleData("
            + ", ".join([f"{k}={getattr(self, k)}" for k in self.toprint])
            + ")"
        )

    def __getitem__(self, key):
        return getattr(self, key)


class PositiveModelData(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        toprint = []
        for k, v in kwargs.items():
            toprint.append(k)
            if isinstance(v, torch.nn.parameter.Parameter):
                self.register_parameter(k, v)
            else:
                self.register_buffer(k, torch.tensor(v))
        self.toprint = toprint

    def __repr__(self):
        return (
            "ModuleData("
            + ", ".join([f"{k}={getattr(self, k)}" for k in self.toprint])
            + ")"
        )

    def __getitem__(self, key):
        return torch.nn.functional.relu(getattr(self, key))


class System(torch.nn.Module):
    R: float = 8.3145  # Universal gas constant [J/(mol*K)]
    F: float = 96485  # Faraday constant [C/mol]

    TOL: float = 1e-36
    MAX_ITER: int = 50

    def __init__(
        self,
        T: float = 298.15,  # [K]
        p0: float = 2.38,  # [bar]
        Q: float = 30,  # [cm^3/min]
        flow_chan_width: float = 1.5e-3,  # [m]
        flow_chan_length: float = 0.02,  # [m]
        flow_chan_height: float = 0.005,  # [m]
        c_khco3: float = 500,  # [mol/l]
        c_k: float = 500,  # [mol/l]
        dic: float = 10 ** (-3.408),
        diffusion_coefficients: Optional[Dict[str, float]] = None,
        salting_out_exponents: Optional[Dict[str, float]] = None,
        electrode_reaction_kinetics: Optional[Dict[str, float]] = None,
        electrode_reaction_potentials: Optional[Dict[str, float]] = None,
        chemical_reaction_rates: Optional[Dict[str, float]] = None,
        co2_equilibrium: Optional[Dict[str, float]] = None,
        method: Literal["DIC", "CO2 eql"] = "CO2 eql",
        system_phase: Literal["gas", "liquid"] = "gas",
        extra_sink: bool = False,
        constant_J_in: bool = False,
    ):
        super().__init__()
        self.T = T
        self.p0 = p0
        self.Q = Q
        self.flow_chan_width = flow_chan_width
        self.flow_chan_length = flow_chan_length
        self.flow_chan_height = flow_chan_height
        self.c_khco3 = c_khco3
        self.c_k = c_k
        self.dic = dic
        self.system_phase = system_phase
        self.extra_sink = extra_sink
        self.constant_J_in = constant_J_in
        if self.system_phase == "liquid":
            self.method = "DIC"
        else:
            self.method = method

        if diffusion_coefficients is not None:
            self.diffusion_coefficients = PositiveModelData(**diffusion_coefficients)
        if salting_out_exponents is not None:
            self.salting_out_exponents = PositiveModelData(**salting_out_exponents)
        if electrode_reaction_kinetics is not None:
            self.electrode_reaction_kinetics = PositiveModelData(
                **electrode_reaction_kinetics
            )
        if electrode_reaction_potentials is not None:
            self.electrode_reaction_potentials = ModelData(
                **electrode_reaction_potentials
            )
        if chemical_reaction_rates is not None:
            self.chemical_reaction_rates = PositiveModelData(**chemical_reaction_rates)

        self.set_initial_carbonate_equilibria()

        if co2_equilibrium is not None:
            self.co2_equilibrium = co2_equilibrium
        else:
            if self.method == "DIC":
                self.co2_equilibrium = self.calculate_dic_solution
            else:
                self.co2_equilibrium = self._co2_equilibrium()

    @property
    def Hnr(self) -> float:
        T = self.T
        return 1000 * np.exp(
            93.4517 * 100 / T - 60.2409 + 23.3585 * np.log(T / 100)
        )  # Henry's constant for CO2 [mol/(L*atm)]

    @cached_property
    def v(self):  # flow velocity [m/s]
        return self.Q / (self.flow_chan_height * self.flow_chan_width) * 1e-6 / 60

    @property
    def volumetric_surface_area(self):
        return 3 * (1 - self.eps) / self.r  # m^-1

    def volumetric_surface_area(self, eps, r):
        return 3 * (1 - eps) / r

    @staticmethod
    def bruggeman(dif_coef, eps):
        return (
            dif_coef * eps**1.5
        )  # Bruggeman correction for diffusion coefficient [m^2/s]
        # self.diffusion_coefficients['Gas diffusion layer mass transfer coefficient'] = self.diffusion_coefficients['DCO2'] / self.input_parameters.r # [m/s]

    @cached_property
    def butler_volmer_factor(self):
        return self.F / (self.R * self.T)

    @cached_property
    def flow_channel_characteristics(self):
        flow_channel_characteristics = {
            "mu": 0.00000093944,  # [m^2/s]
        }
        flow_channel_characteristics["Reynolds Number"] = (
            self.v * self.flow_chan_length / flow_channel_characteristics["mu"]
        )
        flow_channel_characteristics["Hydrodynamic Entrance length"] = (
            0.0099
            * flow_channel_characteristics["Reynolds Number"]
            * self.flow_chan_width
        )
        flow_channel_characteristics["Parallel plate effective boundary layer"] = (
            13 / 35 * self.flow_chan_width
        )
        diff_coeff_to_bl_thickness = (
            lambda x: 3
            * 1.607
            / 4
            * (self.flow_chan_width * x * self.flow_chan_length / self.v) ** (1 / 3)
        )
        diff_coeff_to_K_L = (
            lambda D, L: D
            * torch.sqrt(
                L**-2
                + flow_channel_characteristics[
                    "Parallel plate effective boundary layer"
                ]
                ** -2
            )
            / np.sqrt(2)
        )
        for substance in ["CO2", "OH", "CO3", "HCO3", "H", "K", "CO", "H2"]:
            flow_channel_characteristics[
                f"Developing boundary layer thickness {substance} (average)"
            ] = diff_coeff_to_bl_thickness(
                getattr(self.diffusion_coefficients, substance)
            )
        for substance in ["CO2", "OH", "CO3", "HCO3", "H", "K", "CO", "H2"]:
            flow_channel_characteristics[f"K_L_{substance}"] = diff_coeff_to_K_L(
                getattr(self.diffusion_coefficients, substance),
                flow_channel_characteristics[
                    f"Developing boundary layer thickness {substance} (average)"
                ],
            )
        return flow_channel_characteristics

    def set_initial_carbonate_equilibria(self):
        T = self.T
        kco3_to_salinity = lambda x: x / (1.005 * x / 1000 + 19.924)
        initial_carbonate_equilibria = {
            "Salinity": kco3_to_salinity(self.c_khco3),
            "pK1_0": -126.34048 + 6320.813 / T + 19.568224 * np.log(T),
            "pK2_0": -90.18333 + 5143.692 / T + 14.613358 * np.log(T),
        }
        salinity_to_A1 = lambda x: 13.4191 * x**0.5 + 0.0331 * x - 5.33e-5 * x**2
        initial_carbonate_equilibria["A1"] = salinity_to_A1(
            initial_carbonate_equilibria["Salinity"]
        )
        salinity_to_A2 = lambda x: 21.0894 * x**0.5 + 0.1248 * x - 3.687e-4 * x**2
        initial_carbonate_equilibria["A2"] = salinity_to_A2(
            initial_carbonate_equilibria["Salinity"]
        )
        salinity_to_B1 = lambda x: -530.123 * x**0.5 - 6.103 * x
        initial_carbonate_equilibria["B1"] = salinity_to_B1(
            initial_carbonate_equilibria["Salinity"]
        )
        salinity_to_B2 = lambda x: -772.483 * x**0.5 - 20.051 * x
        initial_carbonate_equilibria["B2"] = salinity_to_B2(
            initial_carbonate_equilibria["Salinity"]
        )
        initial_carbonate_equilibria["C1"] = (
            -2.0695 * initial_carbonate_equilibria["Salinity"] ** 0.5
        )
        initial_carbonate_equilibria["C2"] = (
            -3.3336 * initial_carbonate_equilibria["Salinity"] ** 0.5
        )
        initial_carbonate_equilibria["pK1"] = (
            initial_carbonate_equilibria["pK1_0"]
            + initial_carbonate_equilibria["A1"]
            + initial_carbonate_equilibria["B1"] / T
            + initial_carbonate_equilibria["C1"] * np.log(T)
        )
        initial_carbonate_equilibria["pK2"] = (
            initial_carbonate_equilibria["pK2_0"]
            + initial_carbonate_equilibria["A2"]
            + initial_carbonate_equilibria["B2"] / T
            + initial_carbonate_equilibria["C2"] * np.log(T)
        )
        initial_carbonate_equilibria["K1"] = 10 ** -initial_carbonate_equilibria["pK1"]
        initial_carbonate_equilibria["K2"] = 10 ** -initial_carbonate_equilibria["pK2"]
        initial_carbonate_equilibria["Kw"] = np.exp(
            148.96502
            - 13847.26 / T
            - 23.6521 * np.log(T)
            + (-5.977 + 118.67 / T + 1.0495 * np.log(T))
            * initial_carbonate_equilibria["Salinity"] ** 0.5
            - 0.01615 * initial_carbonate_equilibria["Salinity"]
        )

        self.initial_carbonate_equilibria = initial_carbonate_equilibria

    # @cached_property
    def _co2_equilibrium(self):
        ice = self.initial_carbonate_equilibria

        co2_init = self.Hnr * self.dic
        a = 1
        b = self.c_khco3 / 1000
        c = lambda co2: -(ice["Kw"] + ice["K1"] * co2 / 1000)
        d = lambda co2: -2 * co2 / 1000 * ice["K1"] * ice["K2"]
        f_xn = lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d
        df_xn = lambda x, a, b, c, d: 3 * a * x**2 + 2 * b * x + c
        xn_to_co2 = (
            lambda xn, co2: self.Hnr
            * self.dic
            * torch.exp(
                -self.c_k * self.salting_out_exponents["h_K"]
                - self.salting_out_exponents["h_HCO3"] * co2 * ice["K1"] / xn
                - co2
                * self.salting_out_exponents["h_CO3"]
                * ice["K1"]
                * ice["K2"]
                / xn**2
                - self.salting_out_exponents["h_OH"] * ice["Kw"] / xn * 1000
            )
        )
        # find root of f(x) = 0
        co2 = co2_init
        for i in range(3):
            x_n = opt.newton(
                f_xn,
                x0=1e-5,
                fprime=df_xn,
                tol=self.TOL,
                args=(a, b, c(co2), d(co2)),
                maxiter=self.MAX_ITER,
            )  # rewrite to analytical solution
            co2 = xn_to_co2(x_n, co2)

        co2_equilibrium_sol = {
            "final_pH": -torch.log10(x_n),
            "CO2": co2,
            "HCO3": co2 * ice["K1"] / x_n,
            "OH": ice["Kw"] / x_n * 1000,
            "K": self.c_k,
        }
        co2_equilibrium_sol["CO3"] = co2_equilibrium_sol["HCO3"] * ice["K2"] / x_n
        co2_equilibrium_sol
        return co2_equilibrium_sol

    @cached_property
    def calculate_dic_solution(self):
        ice = self.initial_carbonate_equilibria
        dic_electrolyte_solution = {
            "initial_pH": 7,
        }
        a = 1
        b = ice["K1"] + self.c_khco3 / 1000
        c = ice["K1"] * ice["K2"] - ice["Kw"]
        d = -ice["K1"] * (ice["Kw"] + ice["K2"] * self.c_khco3 / 1000)
        e = -ice["Kw"] * ice["K2"] * ice["K1"]
        x_n = 1e-4
        f_xn = lambda x: a * x**4 + b * x**3 + c * x**2 + d * x + e
        df_xn = lambda x: 4 * a * x**3 + 3 * b * x**2 + 2 * c * x + d
        # find root of f(x) = 0
        x_n = opt.newton(f_xn, x_n, fprime=df_xn, tol=1e-35)
        dic_electrolyte_solution["final_pH"] = -np.log10(x_n)
        dic_electrolyte_solution["CO2"] = self.c_khco3 / (
            1 + ice["K1"] / x_n + ice["K1"] * ice["K2"] / x_n**2
        )
        dic_electrolyte_solution["HCO3"] = (
            dic_electrolyte_solution["CO2"] * ice["K1"] / x_n
        )
        dic_electrolyte_solution["CO3"] = (
            dic_electrolyte_solution["HCO3"] * ice["K2"] / x_n
        )
        dic_electrolyte_solution["OH"] = ice["Kw"] / x_n * 1000
        dic_electrolyte_solution["K"] = self.c_khco3
        return dic_electrolyte_solution

    def solve(
        self,
        phi_ext: torch.Tensor,
        eps: torch.Tensor,
        r: torch.Tensor,
        L: torch.Tensor,
        thetas: Dict[str, torch.Tensor],
        gdl_mass_transfer_coeff: torch.Tensor,
        i_target: Optional[torch.Tensor] = None,
    ):
        A = self.volumetric_surface_area(eps, r)
        F = self.F
        R = self.R
        T = self.T
        Hnr = self.Hnr
        Hnr_c = lambda c1, c2: Hnr * torch.exp(
            -self.salting_out_exponents["h_OH"] * c1
            - self.salting_out_exponents["h_CO3"] * c2
            - self.salting_out_exponents["h_K"] * (c1 + 2 * c2)
        )
        DCO2 = self.bruggeman(self.diffusion_coefficients["CO2"], eps)
        E_CO = self.electrode_reaction_potentials["E_0_CO"]
        co2_equilibrium = self.co2_equilibrium

        OH_neg = co2_equilibrium["OH"]
        HCO3_neg = co2_equilibrium["HCO3"]
        CO3_2neg = co2_equilibrium["CO3"]

        overpotential_CO = phi_ext - E_CO
        overpotential_C2H4 = phi_ext - self.electrode_reaction_potentials["E_0_C2H4"]
        M = lambda k: torch.sqrt(k * L**2 / DCO2)

        _BV_MAX = 40.0  # clamp BV exponent to prevent exp -> inf

        k0_CO = (
            A
            / (2 * F)
            * self.electrode_reaction_kinetics["i_0_CO"]
            * thetas["CO"]
            / self.chemical_reaction_rates["c_ref"]
            * torch.exp(
                torch.clamp(
                -overpotential_CO
                * self.butler_volmer_factor
                * self.electrode_reaction_kinetics["alpha_CO"],
                max=_BV_MAX)
            )
        )
        k0_C2H4 = (
            A
            / (6 * F)
            * self.electrode_reaction_kinetics["i_0_C2H4"]
            * thetas["C2H4"]
            / self.chemical_reaction_rates["c_ref"]
            * torch.exp(
                torch.clamp(
                -overpotential_C2H4
                * self.butler_volmer_factor
                * self.electrode_reaction_kinetics["alpha_C2H4"],
                max=_BV_MAX)
            )
        )
        k0 = k0_CO + k0_C2H4

        r_H2 = (
            A
            * self.electrode_reaction_kinetics["i_0_H2b"]
            * thetas["H2b"]
            / F
            * torch.exp(
                torch.clamp(
                -phi_ext
                * self.butler_volmer_factor
                * self.electrode_reaction_kinetics["alpha_H2b"],
                max=_BV_MAX)
            )
        )  # phi_ext is with respect to SHE

        if self.system_phase == "gas":
            # ----------------------------------------------------------------
            # GAS / SOLID BRANCH
            # 3-step iterative coupling with salting-out corrections
            # ----------------------------------------------------------------

            # Step 1: Initial guess (no homogeneous reaction)
            eff_0 = 1 / (M(k0) / torch.tanh(M(k0)) + k0 * L / gdl_mass_transfer_coeff)
            c00 = Hnr * self.p0 * eff_0

            # Step 2: First OH- estimate
            r_H2_CO2 = (2 * k0_CO + 6 * k0_C2H4) * c00
            c10 = OH_neg + (
                self.flow_channel_characteristics["K_L_OH"] * OH_neg
                - 1
                / (
                    1 / (L * (r_H2 + r_H2_CO2))
                    + 1 / (self.flow_channel_characteristics["K_L_HCO3"] * HCO3_neg)
                )
                + L * (r_H2 + r_H2_CO2)
            ) / (
                self.flow_channel_characteristics["K_L_OH"]
                + 2 * eps * L * self.chemical_reaction_rates["k1f"] * c00
            )

            # Update CO2 with homogeneous reaction sink
            k1 = k0 + eps * self.chemical_reaction_rates["k1f"] * c10
            eff_1 = 1 / (M(k1) / torch.tanh(M(k1)) + k1 * L / gdl_mass_transfer_coeff)
            c01 = eff_1 * self.p0 * Hnr

            # Update OH-
            r_H2_CO2 = (2 * k0_CO + 6 * k0_C2H4) * c01
            c11 = OH_neg + (
                self.flow_channel_characteristics["K_L_OH"] * OH_neg
                - 1
                / (
                    1 / (L * (r_H2 + r_H2_CO2))
                    + 1 / (self.flow_channel_characteristics["K_L_HCO3"] * HCO3_neg)
                )
                + L * (r_H2 + r_H2_CO2)
            ) / (
                self.flow_channel_characteristics["K_L_OH"]
                + 2 * self.chemical_reaction_rates["k1f"] * L * eps * c01
            )

            # Step 3: CO3-- (log approximation)
            A_1 = (
                2 * self.flow_channel_characteristics["K_L_CO3"] * CO3_2neg
                + self.flow_channel_characteristics["K_L_HCO3"] * HCO3_neg
                + self.flow_channel_characteristics["K_L_OH"] * OH_neg
                + L * r_H2
                - self.flow_channel_characteristics["K_L_OH"] * c11
            ) / (2 * self.flow_channel_characteristics["K_L_CO3"])
            B_2 = (
                L
                * 2
                * k0
                * c01
                / (2 * self.flow_channel_characteristics["K_L_CO3"])
                * torch.exp(
                    -c11
                    * (
                        self.salting_out_exponents["h_OH"]
                        + self.salting_out_exponents["h_K"]
                    )
                )
            )
            C = self.salting_out_exponents["h_CO3"] + 2 * self.salting_out_exponents["h_K"]
            c20 = (
                A_1
                + torch.log(
                    1
                    + (B_2 * C * torch.exp(-A_1 * C))
                    / (1 + torch.log(torch.sqrt(1 + B_2 * C * torch.exp(-A_1 * C))))
                )
                / C
            )

            # Salting-out corrected CO2 and final OH-
            c02 = eff_1 * self.p0 * Hnr_c(c11, c20)
            r_H2_CO2 = (2 * k0_CO + 6 * k0_C2H4) * c02
            c12 = OH_neg + (
                self.flow_channel_characteristics["K_L_OH"] * OH_neg
                - 1
                / (
                    1 / (L * (r_H2 + r_H2_CO2))
                    + 1 / (self.flow_channel_characteristics["K_L_HCO3"] * HCO3_neg)
                )
                + L * (r_H2 + r_H2_CO2)
            ) / (
                self.flow_channel_characteristics["K_L_OH"]
                + 2 * self.chemical_reaction_rates["k1f"] * L * eps * c02
            )

            k2 = k0 + eps * self.chemical_reaction_rates["k1f"] * c12
            eff_2 = 1 / (
                torch.sqrt(k2 * L**2 / DCO2)
                / torch.tanh(torch.sqrt(k2 * L**2 / DCO2))
                + k2 * L / gdl_mass_transfer_coeff
            )

            # Final CO3-- with salting-out
            A_1_1 = (
                2 * self.flow_channel_characteristics["K_L_CO3"] * CO3_2neg
                + self.flow_channel_characteristics["K_L_HCO3"] * HCO3_neg
                + self.flow_channel_characteristics["K_L_OH"] * OH_neg
                + L * r_H2
                - self.flow_channel_characteristics["K_L_OH"] * c12
            ) / (2 * self.flow_channel_characteristics["K_L_CO3"])
            B_2_1 = (
                L
                * 2
                * k0
                * eff_2
                * Hnr
                * self.p0
                / (2 * self.flow_channel_characteristics["K_L_CO3"])
                * torch.exp(
                    -c12
                    * (
                        self.salting_out_exponents["h_OH"]
                        + self.salting_out_exponents["h_K"]
                    )
                )
            )
            c21 = A_1_1 + torch.log(
                1
                + (
                    B_2_1
                    * (
                        self.salting_out_exponents["h_CO3"]
                        + 2 * self.salting_out_exponents["h_K"]
                    )
                    * torch.exp(
                        -A_1_1
                        * (
                            self.salting_out_exponents["h_CO3"]
                            + 2 * self.salting_out_exponents["h_K"]
                        )
                    )
                )
                / (
                    1
                    + torch.log(
                        torch.sqrt(
                            1
                            + B_2_1
                            * (
                                self.salting_out_exponents["h_CO3"]
                                + 2 * self.salting_out_exponents["h_K"]
                            )
                            * torch.exp(
                                -A_1_1
                                * (
                                    self.salting_out_exponents["h_CO3"]
                                    + 2 * self.salting_out_exponents["h_K"]
                                )
                            )
                        )
                    )
                )
            ) / (
                self.salting_out_exponents["h_CO3"] + 2 * self.salting_out_exponents["h_K"]
            )
            c21 = torch.maximum(c21, torch.zeros_like(c21))
            c03 = self.p0 * Hnr_c(c12, c21) * eff_2

            # Output quantities specific to gas branch
            gdl_flux = gdl_mass_transfer_coeff * (
                Hnr_c(c12, c21) * self.p0
                - c03
                * torch.sqrt(k2 * L**2 / DCO2)
                / torch.tanh(torch.sqrt(k2 * L**2 / DCO2))
            )
            solubility = Hnr_c(c12, c21) / Hnr
            pH = torch.log10(c12 / 1000 / self.initial_carbonate_equilibria["Kw"])
            hco3 = torch.minimum(
                HCO3_neg + Hnr_c(c12, c21) * self.p0,
                c21 * 10 ** (3 - pH) / self.initial_carbonate_equilibria["K1"],
            )

        else:
            # ----------------------------------------------------------------
            # LIQUID / BPM BRANCH  (neutral bicarbonate electrolyzer)
            # Neumann BC at x=0: CO2 is generated in-situ at the BPM/catalyst
            #   interface by protons crossing the membrane and acidifying
            #   carbonate species.
            # Robin BC at x=L: mass transfer to bulk via K_L_CO2.
            # Exact effectiveness factor eta_c = c_avg / c(L) from cosh/sinh.
            #
            # Steady-state CO2 mass balance over the catalyst layer:
            #   J_consumed = k0*L*c03            (consumed by CO2RR -> CO/C2H4)
            #              + kMT*c03/eta_c       (lost to bulk by convective transfer)
            #              (+ k1f*c12*eps*L*c03)   (homogeneous sink: CO2 + OH- buffering)
            #   J_in = t_CO2 * current_density * 10 / F  (generated at BPM interface, proportional to current)
            #
            #   J_in = J_consumed
            #
            #   Two options:
            #   1. current_density = L * F / 10 * ( c03 * k0_electron + r_H2 )
            #   2. current_density = i_target_b
            #
            # c03 and c12 (OH-) are coupled. We resolve it with a single Picard
            # iteration ("two-step solve"): first c03 ignoring the parasitic sink,
            # then c12, then re-solve c03 with the parasitic sink, then refresh
            # c12. The residual c12 mismatch between the denominator (step 2) and
            # the reported value (step 4) is small and intentional.
            # ----------------------------------------------------------------
            # t_CO2: effective CO2-generation efficiency per unit ionic charge
            # crossing the BPM (fraction of the total current that ends up making
            # CO2 at the interface). NOTE: this replaces the old H+ transference
            # number t_H. Value 0.84 is the apparent t extracted from experimental
            # i-CO2 (sccm) data at ~200 mA/cm^2 (reverse scan); it bundles the
            # true faradaic component (~0.28) plus a ~5 sccm current-independent
            # CO2 crossover baseline, and is insensitive to small changes (FE_CO
            # varies <1% over t in [0.60, 1.00] at this operating point).
            t_CO2 = 0.84
            co2_transfer_coeff = self.flow_channel_characteristics["K_L_CO2"]

            M_val = torch.clamp(M(k0), min=1e-6)
            Sh = co2_transfer_coeff * L / DCO2

            # Exact volume-average / surface-concentration ratio
            # Replaced cosh/sinh with exp to prevent overflow for large M
            e2m = torch.exp(-2 * M_val)
            em = torch.exp(-M_val)
            eta_c = (
                (Sh * (1 + e2m - 2 * em) / M_val**2 + (1 - e2m) / M_val)
                / (Sh * (1 + e2m) + M_val * (1 - e2m))
            )

            k0_electron = 2 * k0_CO + 6 * k0_C2H4
            if i_target is None:
                numerator = t_CO2 * L * r_H2
                denom_0 = (k0 - t_CO2 * k0_electron) * L + co2_transfer_coeff / eta_c 

            else:
                i_target_b = i_target.view(-1, 1).expand(-1, phi_ext.shape[1])
                numerator = t_CO2 * i_target_b * 10.0 / F
                denom_0 = k0 * L + co2_transfer_coeff / eta_c

            # Two-step parasitic-aware solve for c03
            # Step 1: first-pass estimate without parasitic reaction
            c03_0 = numerator / torch.clamp(denom_0, min=1e-20)
            c03_0 = torch.clamp(c03_0, min=0.0, max=1e4)

            if self.extra_sink:
                # Step 2: compute OH- with first-pass c03
                r_OH_gen = r_H2 + k0_electron * c03_0
                c12 = OH_neg + (
                    self.flow_channel_characteristics["K_L_OH"] * OH_neg
                    + L * r_OH_gen
                ) / (
                    self.flow_channel_characteristics["K_L_OH"]
                    + 2 * self.chemical_reaction_rates["k1f"] * L * eps * c03_0
                )
                
                # Step 3: compute parasitic rate coefficient and re-solve c03
                parasitic_rate_coeff = (
                    self.chemical_reaction_rates["k1f"] * c12 * eps * L
                )
                denom_1 = denom_0 + parasitic_rate_coeff
                c03 = numerator / torch.clamp(denom_1, min=1e-20)
                c03 = torch.clamp(c03, min=0.0, max=1e4)
            else:
                c03 = c03_0
                
            # Step 4: recompute OH- with corrected c03 for consistency
            r_OH_gen = r_H2 + k0_electron * c03
            c12 = OH_neg + (
                self.flow_channel_characteristics["K_L_OH"] * OH_neg
                + L * r_OH_gen
            ) / (
                self.flow_channel_characteristics["K_L_OH"]
                + 2 * self.chemical_reaction_rates["k1f"] * L * eps * c03
            )

            # CO3-- balance (linear approximation, no salting-out)
            A_1_1 = (
                2 * self.flow_channel_characteristics["K_L_CO3"] * CO3_2neg
                + self.flow_channel_characteristics["K_L_HCO3"] * HCO3_neg
                + self.flow_channel_characteristics["K_L_OH"] * OH_neg
                + L * r_H2
                - self.flow_channel_characteristics["K_L_OH"] * c12
            ) / (2 * self.flow_channel_characteristics["K_L_CO3"])
            B_2_1 = (
                L * 2 * k0 * c03
                / (2 * self.flow_channel_characteristics["K_L_CO3"])
            )
            c21 = torch.maximum(A_1_1 + B_2_1, torch.zeros_like(A_1_1))

            # Output quantities specific to liquid branch
            gdl_flux = torch.zeros_like(c03)
            solubility = torch.ones_like(c03)
            pKw = -torch.log10(
                torch.as_tensor(
                    self.initial_carbonate_equilibria["Kw"],
                    dtype=c12.dtype,
                    device=c12.device,
                )
            )
            pH = pKw + torch.log10(torch.clamp(c12 / 1000, min=1e-14))
            hco3 = torch.minimum(
                torch.as_tensor(HCO3_neg, dtype=c21.dtype, device=c21.device)
                * torch.ones_like(c21),
                c21 * 10 ** (3 - pH) / self.initial_carbonate_equilibria["K1"],
            )

        # ----------------------------------------------------------------
        # COMMON OUTPUT (both branches)
        # ----------------------------------------------------------------
        potential_vs_rhe = phi_ext - R * T / F * torch.log(c12 / OH_neg)
        co_current_density = L * c03 * k0_CO * 2 * F / 10  # mA/cm^2
        c2h4_current_density = L * c03 * k0_C2H4 * 6 * F / 10  # mA/cm^2
        current_density = (
            co_current_density + c2h4_current_density + (F * L * r_H2) / 10
        )  # mA/cm^2
        current_density_safe = torch.clamp(current_density, min=1e-30)
        fe_co = torch.where(current_density > 1e-20, co_current_density / current_density_safe, torch.zeros_like(current_density))
        fe_c2h4 = torch.where(current_density > 1e-20, c2h4_current_density / current_density_safe, torch.zeros_like(current_density))
        parasitic = c03 * c12 * eps * L * self.chemical_reaction_rates["k1f"]
        electrode = L * k0 * c03

        if self.system_phase == "liquid":
            J_in = t_CO2 * current_density * 10 / F
            co2_utilization = torch.where(
                J_in > 1e-20,
                torch.clamp(electrode / J_in, min=0.0, max=1.0),
                torch.zeros_like(electrode),
            )
        else:
            co2_utilization = torch.where(k0 > 1e-20, k0 / k2, torch.zeros_like(k0))

        return {
            "phi_ext": phi_ext,
            "k0": k0,
            "r_H2": r_H2,
            "potential_vs_rhe": potential_vs_rhe,
            "h2_current_density": F * L * r_H2 / 10,
            "co_current_density": co_current_density,
            "c2h4_current_density": c2h4_current_density,
            "current_density": current_density,
            "co2": c03,
            "co3": c21,
            "oh": c12,
            "pH": pH,
            "fe_co": fe_co,
            "fe_c2h4": fe_c2h4,
            "parasitic": parasitic,
            "electrode": electrode,
            "co2_utilization": co2_utilization,
            "gdl_flux": gdl_flux,
            "hco3": hco3,
            "solubility": solubility,
        }

    def solve_current(
        self,
        i_target: Union[float, torch.Tensor],
        eps: torch.Tensor,
        r: torch.Tensor,
        L: torch.Tensor,
        thetas: torch.Tensor,
        gdl_mass_transfer_coeff: torch.Tensor,
        voltage_bounds: Tuple = (-2, 0),
        return_init_residual: bool = False,
        grid_size: int = 500,
    ):
        if not isinstance(i_target, torch.Tensor):
            i_target = torch.ones_like(eps) * i_target

        phi = torch.linspace(*sorted(voltage_bounds, reverse=True), grid_size).reshape(
            1, -1
        )  # monotonically decreasing
        solution = self.solve(
            phi, eps, r, L, thetas, gdl_mass_transfer_coeff, i_target=(i_target if self.constant_J_in else None)
        )  # monotonically increasing

        I = solution["current_density"].detach()
        I = torch.where(torch.isnan(I) | torch.isinf(I),
                        torch.zeros_like(I), I)

        i_target = i_target.reshape(-1, 1)
        idx = (
            torch.searchsorted(I, i_target, side="right") - 1
        )  # left values. Now interpolate
        idx = torch.clamp(idx, 0, I.shape[1] - 2)

        curr_left = I.gather(dim=1, index=idx)
        curr_right = I.gather(dim=1, index=idx + 1)
        denom = curr_right - curr_left
        p = torch.where(denom.abs() > 1e-20,
                        (i_target - curr_left) / denom,
                        torch.zeros_like(denom))

        out = {}
        for k, v in solution.items():
            if v.shape[0] == 1:
                v = v * torch.ones_like(eps)
            out[k] = (1 - p) * v.gather(dim=1, index=idx) + p * v.gather(
                dim=1, index=idx + 1
            )

        if (
            return_init_residual
        ):  # return the distance to closest original current density
            return out, torch.minimum(
                torch.abs(i_target - curr_left), torch.abs(i_target - curr_right)
            )
        return out


class CurrentSearchError(Exception):
    pass
