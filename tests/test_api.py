from pathlib import Path
import random
import pytest

# Path to this test file
dir_above = Path(__file__).parent

# Path to data/gas.xlsx from repo root
data_path = dir_above.parent / "data" / "gas.xlsx"


def test_import():
    from carbondriver import GDEOptimizer

    gde = GDEOptimizer(output_dir="./tmp_test_out")


# Define test parameters for step_within_data tests
# Format: (model_type, config_overrides)
within_test_params = [
    # Basic tests with default config for all models
    ("GP+Ph", {}),
    ("GP", {}),
    ("Ph", {}),
    ("MLP", {}),
    # Additional test for Ph and GP+Ph with liquid phase
    ("Ph", {"system_phase": "liquid"}),
    ("GP+Ph", {"system_phase": "liquid"}),
    # Additional tests for MLP with different acquisition functions
    ("MLP", {"acquisition": "EI"}),
    ("MLP", {"acquisition": "logEI"}),
    ("MLP", {"acquisition": "PI"}),
    ("MLP", {"acquisition": "UCB"}),
]

@pytest.mark.parametrize("model_type,config_overrides", within_test_params)
def test_gde_optimizer_within(model_type, config_overrides):
    from carbondriver import GDEOptimizer
    from carbondriver.loaders import load_gas_data

    import pandas as pd

    df, current_density = load_gas_data(data_path)

    # Merge base config with overrides
    config = {"current_density": current_density, **config_overrides}
    
    # Use acquisition if specified in config, otherwise default
    acquisition = config_overrides.get("acquisition", "EI")
    
    gde = GDEOptimizer(model_type, aquisition=acquisition, config=config, output_dir="./tmp_test_out")

    df_train = df.iloc[:12]
    df_explore = df.iloc[12:]

    ei, next_pick = gde.step_within_data(df_train, df_explore)

    print(f"First pick ({model_type}):", ei, int(next_pick))

    df_new = df_explore.iloc[int(next_pick)]
    df_explore = df_explore.drop(index=df_new.name)

    ei, next_pick = gde.step_within_data(df_new, df_explore)

    print(f"Second pick ({model_type}):", ei, int(next_pick))


# Define test parameters for step (free optimization) tests
# Format: (model_type, config_overrides)
free_test_params = [
    # Basic tests with default config for all models
    ("MLP", {}),
    ("GP+Ph", {}),
    ("GP", {}),
    ("Ph", {}),
    # Additional test for Ph and GP+Ph with liquid phase
    ("Ph", {"system_phase": "liquid"}),
    ("GP+Ph", {"system_phase": "liquid"}),
    # Additional tests for MLP with different acquisition functions
    ("MLP", {"acquisition": "EI"}),
    ("MLP", {"acquisition": "logEI"}),
    ("MLP", {"acquisition": "PI"}),
    ("MLP", {"acquisition": "UCB"}),
]

@pytest.mark.parametrize("model_type,config_overrides", free_test_params)
def test_gde_optimizer_free(model_type, config_overrides):
    from carbondriver import GDEOptimizer
    from carbondriver.loaders import load_gas_data

    df, current_density = load_gas_data(data_path)
    
    # Merge base config with overrides
    config = {"current_density": current_density, **config_overrides}
    
    # Use acquisition if specified in config, otherwise default
    acquisition = config_overrides.get("acquisition", "EI")

    config["extra_sink"] = True # This stabilizes the liquid model
    
    gde = GDEOptimizer(model_type, aquisition=acquisition, config=config, output_dir="./tmp_test_out")

    ei, next_pick = gde.step(df.iloc[:18])

    print(f"Result ({model_type}):", ei, next_pick)

if __name__ == "__main__":
        test_gde_optimizer_free("Ph", {"system_phase": "liquid"})
