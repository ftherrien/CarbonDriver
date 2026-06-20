# **Car**bon**Driver**

An api and a collection physics-based uncertainty-aware models to drive automated CO2RR laboratories

See our paper: [A physics-based data-driven model for CO2 gas diffusion electrodes to drive automated laboratories](https://arxiv.org/abs/2502.06323v1)

## Install

```
pip install git+https://github.com/ftherrien/CarbonDriver.git
```

## Example

```
from carbondriver import GDEOptimizer
from carbondriver.loaders import load_gas_data

# Data to start with (could be any df)
df, current_density = load_gas_data(data_path)

config = {"current_density": current_density}
             
gde = GDEOptimizer("Ph", config=config, output_dir="./tmp_test_out")

ei, next_pick = gde.step(df)

print(f"Your next experiment should be:", next_pick)

```
