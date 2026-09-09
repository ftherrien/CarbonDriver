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

## Integration controls

Physics-based optimization can keep measured inputs fixed and derive electrode
thickness while optimizing only the manipulated inputs:

```python
config = {
    "current_density": 100.0,
    "electrode_area_cm2": 4.0,
    "physics_mass_source": "Ink mass",
    "physics_mass_units": "mg",
    "torch_seed": 0,
}

gde = GDEOptimizer(
    "GP+Ph",
    config=config,
    bounds=bounds,
    input_labels=input_labels,
    output_labels=output_labels,
    quantity=output_labels[0],
)
acquisition_value, next_pick = gde.step(
    observations,
    fixed_features={"Ag weight": 8.0},
)
```

After a successful model-based step, `last_prediction_means`,
`last_prediction_stds`, and `acquisition_context` expose prediction and
diagnostic metadata without refitting the model. Model-fitting failures raise
`OptimizerTrainingError`; CarbonDriver does not replace them with a random
experimental recommendation.
