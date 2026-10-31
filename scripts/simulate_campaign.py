from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.interpolate import LinearNDInterpolator
import argparse

from carbondriver import GDEOptimizer
from carbondriver.loaders import (
    DEFAULT_ELECTRODE_AREA_CM2,
    Ag_DENSITY,
    load_campaign_data,
)

eps = 1e-3  # small value to avoid extrapolation in the corners

DEFAULT_ELECTRODE_AREA_CM2 = 4


def converter(
    df, direction="to_zlt", electrode_area_cm2: float = DEFAULT_ELECTRODE_AREA_CM2
):

    area = electrode_area_cm2  # cm^2
    A = area * 1e-4  # m^2

    if direction == "to_zlt":
        mass = df["Ink mass"] * 1e-6  # kg
        thickness = (mass / Ag_DENSITY) / A  # m
        df["zero_eps_thickness"] = thickness
        df = df.drop("Ink mass", axis=len(df.axes) - 1)
    elif direction == "from_zlt":
        df["Ink mass"] = df["zero_eps_thickness"] * A * Ag_DENSITY * 1e6  # kg
        df = df.drop("zero_eps_thickness", axis=len(df.axes) - 1)

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simulate a campaign")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_file = config.get("data_file", None)
    if data_file is not None:
        p = Path(data_file)
        data_file = p if p.is_absolute() else Path(__file__).resolve().parent / p

    property_name = config["property_name"]
    input_labels = config.get("input_labels", ["Hotplate temperature (catalyst)", "Ink mass"])
    bounds = torch.tensor(config.get("bounds", [[30, 20], [150, 100]]), dtype=torch.float32)
    # Only the Ph and GP+Ph models expect zero-eps-thickness inputs
    use_zlt = config["models"][0] in ("Ph", "GP+Ph")

    df, init_triplets = load_campaign_data(
        file=data_file, sheets=config.get("sheets", None), convert_to_zlt=False
    )

    df_triplet_means = df.groupby("triplet").mean()

    if config.get("simulation", True):
        xy = df_triplet_means[input_labels].to_numpy()
        z = df_triplet_means[property_name].to_numpy()
        
        # Adding 0s to the corners of the input space to avoid extrapolation
        corners = np.array(
            [
                [bounds[0, 0] - eps, bounds[0, 1] - eps],
                [bounds[0, 0] - eps, bounds[1, 1] + eps],
                [bounds[1, 0] + eps, bounds[0, 1] - eps],
                [bounds[1, 0] + eps, bounds[1, 1] + eps],
            ]
        )
        xy = np.vstack([xy, corners])
        z = np.hstack([z, [0, 0, 0, 0]])
        
        lab_simulator = LinearNDInterpolator(xy, z)
        
        # Just plotting the interpolation surface
        X = np.linspace(bounds[0, 0], bounds[1, 0])
        Y = np.linspace(bounds[0, 1], bounds[1, 1])
        
        X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
        Z = lab_simulator(X, Y)
        
        plt.pcolormesh(X, Y, Z, shading="auto")
        plt.plot(
            df_triplet_means.loc[:, input_labels[0]].to_numpy(),
            df_triplet_means.loc[:, input_labels[1]].to_numpy(),
            "ok",
            label="input point",
        )
        plt.legend()
        plt.colorbar()
        plt.axis("equal")
        plt.show()

    # Simulated campaign

    # df, init_triplets = load_campaign_data("simulated_campaign_results.xlsx", convert_to_zlt = False) # TMP

    data = df[df["triplet"].isin(init_triplets)]
    starting_df = converter(data.copy(), "to_zlt") if use_zlt else data.copy()
    data["Type"] = "init"
    data = data.drop(columns=["triplet"])
    data[property_name] = data[property_name] * 100

    data.to_csv("simulated_campaign_start.csv", index=False)

    if use_zlt:
        gde_input_labels = [input_labels[0], "zero_eps_thickness"]
        gde_bounds = torch.tensor(
            converter(
                pd.DataFrame(bounds.numpy(), columns=input_labels), "to_zlt"
            ).to_numpy(),
            dtype=torch.float32,
        )
    else:
        gde_input_labels = input_labels
        gde_bounds = bounds

    print("Bounds:\n", gde_bounds)
    print("Starting data:\n", starting_df)

    gde = GDEOptimizer(
        config["models"][0],
        acquisition=config["acquisition"],
        config=config,
        output_dir=config.get("run_name", "./sim_campaign"),
        input_labels=gde_input_labels,
        bounds=gde_bounds,
        quantity=property_name,
        output_labels=[property_name],
    )

    new_data = starting_df.copy()

    for i in range(10):

        ei, new_data = gde.step(new_data)

        new_data_converted = converter(new_data, "from_zlt").copy() if use_zlt else new_data.copy()

        print(f"Step {i+1}: Suggested experiment:", new_data_converted)

        if config.get("simulation", True):
            new_data[property_name] = float(
                lab_simulator(
                    new_data_converted[input_labels[0]], new_data_converted[input_labels[1]]
                )
            )

            # Fake duplicate
            new_data = pd.concat([new_data.to_frame().T]*config.get("n_replicates", 1))
            
        else:
            new_data = new_data.to_frame().T

            new_data = new_data.loc[new_data.index.repeat(config.get("n_replicates", 1))].reset_index(drop=True)
            
            for i in new_data.index:
                new_data.loc[i, property_name] = float(input(f"Result in %:")) / 100
        
        print("Sim lab result:", new_data[property_name])
        print("Acquisition function value:", ei)

        new_data_converted[property_name] = new_data[property_name] * 100
        new_data_converted["Type"] = "simulated"
        data = pd.concat([data, new_data_converted], axis=0)

    data = data.reset_index(drop=True)
    data = pd.concat(
        [pd.DataFrame(np.zeros((1, data.shape[1])), columns=data.columns), data], axis=0
    )
    data.index.name = "Sample ID"
    data.to_excel("simulated_campaign_results.xlsx", index=True)
