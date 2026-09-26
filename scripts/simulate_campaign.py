from carbondriver.loaders import load_campaign_data, DEFAULT_ELECTRODE_AREA_CM2, Ag_DENSITY
from scipy.interpolate import LinearNDInterpolator
import numpy as np
import matplotlib.pyplot as plt
import torch
from carbondriver import GDEOptimizer
import yaml
import pandas as pd

INPUT_LABELS = ["Hotplate temperature (catalyst)", "Ink mass"]

BOUNDS = torch.tensor([[40, 25], [150, 90]], dtype=torch.float32)

eps = 1e-3  # small value to avoid extrapolation in the corners

def converter(df, direction="to_zlt", electrode_area_cm2: float = DEFAULT_ELECTRODE_AREA_CM2):

    area = electrode_area_cm2  # cm^2
    A = area * 1e-4  # m^2
    
    if direction == "to_zlt":
        mass = df["Ink mass"] * 1e-6  # kg
        thickness = (mass / Ag_DENSITY) / A  # m
        df["zero_eps_thickness"]  = thickness
        df = df.drop("Ink mass", axis=len(df.axes) - 1)
    elif direction == "from_zlt":
        df["Ink mass"] = df["zero_eps_thickness"] * A * Ag_DENSITY * 1e6  # kg
        df = df.drop("zero_eps_thickness", axis=len(df.axes) - 1)

    return df
    

if __name__ == "__main__":

    df, init_triplets = load_campaign_data(convert_to_zlt = False)
    
    df_triplet_means = df.groupby('triplet').mean()
    xy = df_triplet_means[INPUT_LABELS].to_numpy()
    z= df_triplet_means["FE CO"].to_numpy()

    # Addin 0s to the corners of the input space to avoid extrapolation
    corners = np.array([[BOUNDS[0, 0] - eps, BOUNDS[0, 1] - eps], 
                        [BOUNDS[0, 0] - eps, BOUNDS[1, 1] + eps], 
                        [BOUNDS[1, 0] + eps, BOUNDS[0, 1] - eps], 
                        [BOUNDS[1, 0] + eps, BOUNDS[1, 1] + eps]])
    xy = np.vstack([xy, corners])
    z = np.hstack([z, [0, 0, 0, 0]])
    
    lab_simulator = LinearNDInterpolator(xy, z)
    
    # Just ploting the interpolation surface
    X = np.linspace(BOUNDS[0, 0], BOUNDS[1, 0])
    Y = np.linspace(BOUNDS[0, 1], BOUNDS[1, 1])

    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    Z = lab_simulator(X, Y)
    
    plt.pcolormesh(X, Y, Z, shading='auto')
    plt.plot(df_triplet_means.loc[:,INPUT_LABELS[0]].to_numpy(), df_triplet_means.loc[:,INPUT_LABELS[1]].to_numpy(), "ok", label="input point")
    plt.legend()
    plt.colorbar()
    plt.axis("equal")
    # plt.show()

    # Simulated campaign
    
    data = df[df["triplet"].isin(init_triplets)]
    starting_df = converter(data.copy(), "to_zlt")
    data["Type"] = "init"
    data = data.drop(columns=["triplet"])
    data["FE CO"] = data["FE CO"] * 100
    
    
    zlt_bounds = torch.tensor(converter(pd.DataFrame(BOUNDS.numpy(), columns=INPUT_LABELS), "to_zlt").to_numpy(), dtype=torch.float32)

    print("Bounds:\n", zlt_bounds)
    print("Starting data:\n", starting_df)
    
    with open("config_fixed_current_liquid.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
         
    gde = GDEOptimizer("GP+Ph", acquisition="UCB", config=config, output_dir="./sim_campaign", input_labels=["Hotplate temperature (catalyst)", "zero_eps_thickness"], bounds=zlt_bounds, quantity="FE CO", output_labels=["FE CO"])

    new_data = starting_df.copy()

    for i in range(10):
    
        ei, new_data = gde.step(new_data)

        new_data_converted = converter(new_data, "from_zlt").copy()

        print(f"Step {i+1}: Suggested experiment:", new_data_converted)
        
        new_data["FE CO"] = float(lab_simulator(new_data_converted[INPUT_LABELS[0]], new_data_converted[INPUT_LABELS[1]]))

        print("Sim lab result:", new_data["FE CO"])

        new_data_converted["FE CO"] = new_data["FE CO"] * 100
        new_data_converted["Type"] = "simulated"
        data = pd.concat([data, new_data_converted.to_frame().T], axis=0)

    data = data.reset_index(drop=True)
    data = pd.concat([pd.DataFrame(np.zeros((1,data.shape[1])), columns=data.columns), data], axis=0)
    data.index.name = "Sample ID"
    data.to_excel("simulated_campaign_results.xlsx", index=True)
