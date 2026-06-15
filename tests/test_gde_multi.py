from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch


def _load_gde_multi_module():
    module_path = Path(__file__).resolve().parents[1] / "carbondriver" / "gde_multi.py"
    spec = spec_from_file_location("gde_multi_local", module_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


gde_multi = _load_gde_multi_module()


def _make_system(system_phase="gas", method="CO2 eql"):
    erc = gde_multi.electrode_reaction_kinetics | {}
    erc["i_0_CO"] = torch.nn.parameter.Parameter(torch.tensor(erc["i_0_CO"]))
    erc["i_0_C2H4"] = torch.nn.parameter.Parameter(torch.tensor(erc["i_0_C2H4"]))
    erc["i_0_H2b"] = torch.nn.parameter.Parameter(torch.tensor(erc["i_0_H2b"]))
    erc["alpha_CO"] = torch.nn.parameter.Parameter(torch.tensor(erc["alpha_CO"]))
    erc["alpha_C2H4"] = torch.nn.parameter.Parameter(torch.tensor(erc["alpha_C2H4"]))
    erc["alpha_H2b"] = torch.nn.parameter.Parameter(torch.tensor(erc["alpha_H2b"]))
    return gde_multi.System(
        diffusion_coefficients=gde_multi.diffusion_coefficients,
        salting_out_exponents=gde_multi.salting_out_exponents,
        electrode_reaction_kinetics=erc,
        electrode_reaction_potentials=gde_multi.electrode_reaction_potentials,
        chemical_reaction_rates=gde_multi.chemical_reaction_rates,
        method=method,
        system_phase=system_phase,
    )


def _base_inputs():
    eps = torch.tensor(0.5)
    r = torch.tensor(3e-8)
    L = torch.tensor(1e-5)
    thetas = {
        "CO": torch.tensor(0.3),
        "C2H4": torch.tensor(0.3),
        "H2b": torch.tensor(0.4),
    }
    gdl_mass_transfer_coefficient = torch.tensor(0.015)
    return eps, r, L, thetas, gdl_mass_transfer_coefficient


@pytest.mark.parametrize("method", ["CO2 eql", "DIC"])
def test_gas_mode_keeps_requested_method(method):
    model = _make_system(system_phase="gas", method=method)
    assert model.method == method


def test_liquid_mode_forces_dic_method_even_when_co2_equilibrium_requested():
    model = _make_system(system_phase="liquid", method="CO2 eql")
    assert model.method == "DIC"


@pytest.mark.parametrize("system_phase", ["gas", "liquid"])
def test_solve_current_runs_for_each_physics_mode(system_phase):
    model = _make_system(system_phase=system_phase)
    eps, r, L, thetas, gdl_mass_transfer_coefficient = _base_inputs()

    solution = model.solve_current(
        i_target=233,
        eps=eps,
        r=r,
        L=L,
        thetas=thetas,
        gdl_mass_transfer_coeff=gdl_mass_transfer_coefficient,
        grid_size=1000,
        voltage_bounds=(-1.25, 0),
    )

    for key in ["gdl_flux", "current_density", "pH", "co2", "oh", "co3"]:
        assert key in solution
        assert torch.isfinite(solution[key]).all().item()

    if system_phase == "liquid":
        assert torch.allclose(
            solution["gdl_flux"], torch.zeros_like(solution["gdl_flux"])
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
