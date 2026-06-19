from pathlib import Path
import random
import pytest

# Path to this test file
dir_above = Path(__file__).parent

# Path to paper/Characterization_data.xlsx from repo root
data_path = dir_above.parent / "paper" / "Characterization_data.xlsx"


def test_import():
    from carbondriver import GDEOptimizer

    gde = GDEOptimizer(output_dir="./tmp_test_out")


@pytest.mark.parametrize("model_type", ["GP+Ph", "GP", "Ph", "MLP"])
def test_gde_optimizer_within(model_type):
    from carbondriver import GDEOptimizer
    from carbondriver.loaders import load_gas_data

    import pandas as pd

    df, current_density = load_gas_data(data_path)

    gde = GDEOptimizer(model_type, config={"current_density": current_density}, output_dir="./tmp_test_out")

    df_train = df.iloc[:12]
    df_explore = df.iloc[12:]

    ei, next_pick = gde.step_within_data(df_train, df_explore)

    print("First pick:", ei, int(next_pick))

    df_new = df_explore.iloc[int(next_pick)]
    df_explore = df_explore.drop(index=df_new.name)

    ei, next_pick = gde.step_within_data(df_new, df_explore)

    print("Second pick", ei, int(next_pick))

@pytest.mark.parametrize("model_type", ["MLP", "GP+Ph", "GP", "Ph"])
def test_gde_optimizer_free(model_type):
    from carbondriver import GDEOptimizer
    from carbondriver.loaders import load_gas_data

    df, current_density = load_gas_data(data_path)
    
    gde = GDEOptimizer(model_type, config={"current_density": current_density}, output_dir="./tmp_test_out")

    ei, next_pick = gde.step(df.iloc[:12])

    print(ei, next_pick)

if __name__ == "__main__":
        test_gde_optimizer_free("MLP")
