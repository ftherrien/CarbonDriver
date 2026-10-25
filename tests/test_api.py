from pathlib import Path

import pytest
import yaml

# Path to this test file
dir_above = Path(__file__).parent

# Path to data/gas.xlsx from repo root
data_path = dir_above.parent / "data" / "gas.xlsx"

# Fixed seed + reference (ei, next_pick) values captured on main before the
# acquisition/llm/plotting module split, used to check the refactor is a no-op.
SEED = 12345
VALUES_PATH = dir_above / "test_api_values.yaml"


def _check_or_update(key, actual, check_values, update_values):
    """Compare `actual` to the recorded reference value for `key`, or overwrite it."""
    with VALUES_PATH.open() as f:
        values = yaml.load(f, Loader=yaml.FullLoader)
    if update_values:
        values[key] = actual
        yaml.dump(values, VALUES_PATH.open("w"), sort_keys=False)
    elif check_values:
        assert actual == values[key]


def test_import():
    from carbondriver import GDEOptimizer

    gde = GDEOptimizer(output_dir="./tmp_test_out")


# Define test parameters for step_within_data tests
# Format: (model_type, config_overrides)
test_params = [
    # Basic tests with default config for all models
    ("GP+Ph", {}),
    ("GP", {}),
    ("BoTorch-GP", {}),
    ("Ph", {}),
    ("MLP", {}),
    ("LLM", {}),
    # Additional test for Ph and GP+Ph with liquid phase
    ("Ph", {"system_phase": "liquid"}),
    ("GP+Ph", {"system_phase": "liquid"}),
    # Additional tests for MLP with different acquisition functions
    ("MLP", {"acquisition": "EI"}),
    ("MLP", {"acquisition": "logEI"}),
    ("MLP", {"acquisition": "PI"}),
    ("MLP", {"acquisition": "UCB"}),
    ("MLP", {"variance_scaling": False}),
]


@pytest.mark.parametrize("model_type,config_overrides", test_params)
def test_gde_optimizer_within(model_type, config_overrides, check_values, update_values):
    from carbondriver import GDEOptimizer
    from carbondriver.loaders import load_gas_data

    df, current_density = load_gas_data(data_path)

    # Merge base config with overrides
    config = {"current_density": current_density, "torch_seed": SEED, **config_overrides}

    # Use acquisition if specified in config, otherwise default
    acquisition = config_overrides.get("acquisition", "EI")

    gde = GDEOptimizer(
        model_type, acquisition=acquisition, config=config, output_dir="./tmp_test_out"
    )

    df_train = df.iloc[:12]
    df_explore = df.iloc[12:]

    ei1, pick1 = gde.step_within_data(df_train, df_explore)

    print(f"First pick ({model_type}):", ei1, int(pick1))

    df_new = df_explore.loc[pick1]
    df_explore = df_explore.drop(index=df_new.name)

    ei2, pick2 = gde.step_within_data(df_new, df_explore)

    print(f"Second pick ({model_type}):", ei2, int(pick2))

    if model_type != "LLM":
        _check_or_update(
            f"within|{model_type}|{config_overrides}",
            {"ei1": ei1, "pick1": int(pick1), "ei2": ei2, "pick2": int(pick2)},
            check_values, update_values,
        )


@pytest.mark.parametrize("model_type,config_overrides", test_params)
def test_gde_optimizer_free(model_type, config_overrides, check_values, update_values):
    from carbondriver import GDEOptimizer
    from carbondriver.loaders import load_gas_data

    df, current_density = load_gas_data(data_path)

    # Merge base config with overrides
    config = {"current_density": current_density, "torch_seed": SEED, **config_overrides}

    # Use acquisition if specified in config, otherwise default
    acquisition = config_overrides.get("acquisition", "EI")

    config["extra_sink"] = True  # This stabilizes the liquid model

    gde = GDEOptimizer(
        model_type, acquisition=acquisition, config=config, output_dir="./tmp_test_out"
    )

    ei, next_pick = gde.step(df.iloc[:18])

    print(f"Result ({model_type}):", ei, next_pick)

    if model_type != "LLM":
        _check_or_update(
            f"free|{model_type}|{config_overrides}",
            {"ei": None if ei is None else ei.item(), "pick": next_pick.to_dict()},
            check_values, update_values,
        )


if __name__ == "__main__":
    test_gde_optimizer_free("Ph", {"system_phase": "liquid"})
