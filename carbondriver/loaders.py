from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json
import os

Ag_DENSITY = 10490  # kg/m^3
Cu_DENSITY = 8935  # kg/m^3

# Load results from folder
def load_results_from_folder(
    folder: Path, plot: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training losses from JSON files in folder.

    :param folder: path to folder containing losses*.json files
    :param plot: whether to plot training curves
    :returns: tuple of (train_df, val_df) DataFrames
    """
    df_res = pd.DataFrame()
    for f in Path(folder).glob("losses*.json"):
        d = json.loads(f.read_text())
        _df = pd.DataFrame(d).set_index("epochs")
        if plot:
            plt.plot(_df.index, _df["train"], c="0.5", lw=0.5)
            plt.plot(_df.index, _df["val"], c="k", linestyle="--", lw=0.5)
        i = int(f.stem.split("_")[-1])
        df_res = df_res.merge(
            _df, how="outer", left_index=True, right_index=True, suffixes=("", f"_{i}")
        )
    df_res

    # separate train and val
    cols = df_res.columns
    train_cols = [c for c in cols if "train" in c]
    val_cols = [c for c in cols if "val" in c]
    df_res_train = df_res[train_cols]
    df_res_val = df_res[val_cols]
    return df_res_train, df_res_val


def load_gas_data(file: Optional[Path] = None) -> pd.DataFrame:
    """Load experimental data from Excel and compute electrode thickness.

    :param file: path to Excel file (default: ./data/gas.xlsx)
    :returns:
        DataFrame with features and experimental Faradaic efficiencies
        current_density
    """
    if file is None:
        file = Path("./data/gas.xlsx")
    df = pd.read_excel(file, skiprows=[1], index_col=0)

    current = df["Current (A)"].unique()
    assert len(current) == 1
    
    df = df[
        [
            "AgCu Ratio",
            "Naf vol (ul)",
            "Sust vol (ul)",
            "Catalyst mass loading",
            "FE (Eth)",
            "FE (CO)",
        ]
    ]
    df = df.sort_values(by=["AgCu Ratio", "Naf vol (ul)"])
    df = df.dropna()
    df["FE (CO)"] = df["FE (CO)"] / 100
    df["FE (Eth)"] = df["FE (Eth)"] / 100

    dens_avg = (1 - df["AgCu Ratio"]) * Cu_DENSITY + df["AgCu Ratio"] * Ag_DENSITY
    mass = df["Catalyst mass loading"] * 1e-6  # kg
    area = 1.85**2  # cm^2
    A = area * 1e-4  # m^2
    thickness = (mass / dens_avg) / A  # m
    df.insert(3, column="zero_eps_thickness", value=thickness)

    # reshuffle triplets
    df["triplet"] = np.arange(len(df)) // 3

    current_density = current[0] * 1e3 / area  # mA/cm^2
    
    return df.astype(float), current_density

def load_bicarb_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Reads the Bicarb CSV and melts the multiple result columns into new entries.
    """

    if filepath is None:
        filepath = Path("./data/bicarb.xlsx")

    df = pd.read_excel(filepath, header=1, index_col=0).iloc[3:,:]

    df = separate_repeats(df)
    df = df.apply(pd.to_numeric, errors='coerce')

    df = df.drop(columns=[df.columns[0], "Voltage"])

    df["FE_CO"] = df["FE_CO"] / 100
    df["CO2 utilization"] = df["CO2 utilization"] / 100

    mass = df["Ag weight"] * 1e-6  # kg
    area = 1.85**2  # cm^2
    A = area * 1e-4  # m^2
    thickness = (mass / Ag_DENSITY) / A  # m
    df.insert(1, column="zero_eps_thickness", value=thickness)
    df.rename(columns={"Current density": "current_density"}, inplace=True)

    return df.astype(float)
    
def separate_repeats(df):
    
    df.columns = [c.strip() for c in df.columns]

    result_metrics = ['Voltage', 'FE_CO', 'CO2 utilization']

    # Find all columns that ARE NOT the repeating result metrics
    param_cols = [c for c in df.columns if not any(m in c for m in result_metrics)]

    # 3. Handle the "Multi-Result" pivot
    melted_frames = []

    # Pandas appends .1 and .2 to duplicate column names
    suffixes = ['', '.1', '.2']
    for i, suffix in enumerate(suffixes):
        v_col = f'Voltage{suffix}'
        f_col = f'FE_CO{suffix}'
        c_col = f'CO2 utilization{suffix}'
        # Check if this set exists
        if v_col in df.columns:
            cols_to_extract = param_cols + [v_col, f_col, c_col]
            subset = df[cols_to_extract].copy()
            # Rename to standard names (removing suffixes)
            subset = subset.rename(columns={
                v_col: 'Voltage',
                f_col: 'FE_CO',
                c_col: 'CO2 utilization' # We'll use underscore for the final DF
            })
            melted_frames.append(subset)
    # Combine all result sets
    final_df = pd.concat(melted_frames, ignore_index=True)

    # Drop rows where ALL result values are NaN
    final_df = final_df.dropna(subset=result_metrics, how='all')

    final_df["triplet"] = final_df.groupby(param_cols).ngroup()
    
    return final_df


def feature_stats(
    df: pd.DataFrame,
    all_means: Optional[pd.Series] = None,
    all_stds: Optional[pd.Series] = None,
    as_torch: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[pd.Series, pd.Series]:
    """
    Return per-feature means/stds (excluding the last two target columns).

    :param df: full dataframe with features first and last two columns as targets
    :param all_means: optional precomputed means across all columns
    :param all_stds: optional precomputed stds across all columns
    :param as_torch: return tensors if True, else pandas Series
    :returns: tuple of (means, stds) as torch.Tensor or pd.Series

    Notes
    - If all_means/all_stds are provided, we subset them to feature columns.
    - Otherwise, we compute stats over feature columns only.
    """
    feat_cols = df.columns[:-2]
    if all_means is not None and all_stds is not None:
        m = all_means[feat_cols]
        s = all_stds[feat_cols]
    else:
        m = df[feat_cols].mean()
        s = df[feat_cols].std(ddof=0)
    if as_torch:
        return (
            torch.tensor(m.values, dtype=torch.float32),
            torch.tensor(s.values, dtype=torch.float32),
        )
    else:
        return m, s
