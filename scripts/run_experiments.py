"""
External wrapper to run active learning experiments and generate plots.
This runs the test suite multiple times, collecting data at each step.
"""

from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from carbondriver import GDEOptimizer
from carbondriver.loaders import load_gas_data, load_bicarb_data
from typing import Tuple, Optional, Literal
import torch
import yaml
import sys
import argparse

# Keys that are always lists and never create cartesian-product combinations.
# Also never put 'dataset' or 'data_file' in a list — all combos must share the same data.
ALWAYS_LIST_KEYS = {'runs', 'input_labels', 'output_labels'}


def expand_config(base_config: dict):
    """Yield one sub-config per combination of list-valued params.

    Every param whose value is a list (except ALWAYS_LIST_KEYS) becomes an
    axis of the cartesian product.  Each sub-config has scalar values for
    those params and a run_name suffixed with their values, e.g. "run-GP-EI".
    """
    var_keys = [k for k, v in base_config.items()
                if isinstance(v, list) and k not in ALWAYS_LIST_KEYS]
    if not var_keys:
        yield base_config
        return
    for combo in itertools.product(*[base_config[k] for k in var_keys]):
        sub = {**base_config}
        for k, v in zip(var_keys, combo):
            sub[k] = v
        sub["run_name"] = base_config["run_name"] + "/" + "-".join(str(v) for v in combo)
        yield sub


_BLUE_SHADES  = ["royalblue", "navy", "dodgerblue", "steelblue", "cornflowerblue", "mediumblue"]
_GREEN_SHADES = ["limegreen", "darkgreen", "forestgreen", "mediumseagreen", "olivedrab", "seagreen"]
_MARKERS      = ["o", "s", "^", "D", "v", "P", "*", "X"]


def build_combo_style(combo_configs: list) -> tuple[dict, dict]:
    """Return (palette, markers) dicts mapping each combo run_name → color/marker.

    MLP/GP → blue shades, Ph/GP+Ph → green shades, baseline → gray.
    Shades and markers cycle within each family as more combos are added.
    """
    palette = {"baseline": "gray"}
    markers = {"baseline": "o"}
    bi = gi = 0
    for c in combo_configs:
        name  = c["run_name"]
        model = c["models"]
        if model in ("MLP", "GP"):
            palette[name] = _BLUE_SHADES[bi % len(_BLUE_SHADES)]
            markers[name] = _MARKERS[bi % len(_MARKERS)]
            bi += 1
        elif model in ("Ph", "GP+Ph"):
            palette[name] = _GREEN_SHADES[gi % len(_GREEN_SHADES)]
            markers[name] = _MARKERS[gi % len(_MARKERS)]
            gi += 1
        else:
            palette[name] = f"C{len(palette)}"
            markers[name] = _MARKERS[(bi + gi) % len(_MARKERS)]
    return palette, markers


def choose_base_inds_numpy(y: np.ndarray, num_choose: int, how: Literal['max','min'] = 'max', strategy: Literal['uniform','skewed'] = 'uniform', seed: Optional[int] = None):
    ind = np.argsort(y)
    N = y.shape[0]
    i = np.arange(N)
    if strategy=='skewed':
        if how=='max':
            p = (i - i.max())**2
        elif how=='min':
            p = i**2
    elif strategy=='uniform':
        p = np.ones_like(i)
    else: 
        raise ValueError
    p = p/p.sum()
    rng = np.random.default_rng(seed)
    return rng.choice(ind, size=num_choose, replace=False, p=p)

def run_active_learning_experiment(model_name: str, run_idx: int, config: dict):
    """Run a single active learning experiment for the given model."""
    
    print(f"  [Step 1/3] Preparing data for {model_name} run {run_idx}...")
    # Reset df for this run
    df_triplet_means = df.groupby('triplet').mean()
    best_id = int(df_triplet_means[config["property_name"]].idxmax())
    
    run_dir = OUTPUT_BASE / f'{model_name}_run_{run_idx:03d}'
    run_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"  [Step 2/3] Initializing {model_name} optimizer...")
    # Initialize optimizer
    acquisition = config.get("acquisition", "EI")
    gde = GDEOptimizer(
        model_name=model_name,
        aquisition=acquisition,
        quantity=config["property_name"],
        maximize=True,
        output_dir=str(run_dir),
        config=config,
        input_labels=config["input_labels"],
        output_labels=config["output_labels"],
    )
    
    # Choose initial triplets
    print(f"  [Step 3/3] Selecting initial triplets...")
    chosen_triplets_ids = choose_base_inds_numpy(
        df_triplet_means[config["property_name"]].values,
        num_choose=3,
        strategy='uniform',
        seed=run_idx
    ).tolist()

    bests = df_triplet_means.loc[chosen_triplets_ids][config["property_name"]].cummax().tolist()
    print(f"    Starting triplets: {chosen_triplets_ids}")
    print(f"    Starting best FE values: {[f'{b:.4f}' for b in bests]}")
    print(f"    Objective: {df_triplet_means.loc[best_id, config['property_name']]:.4f} at triplet {best_id}")

    # Track results
    expected_improvements = [None] * len(chosen_triplets_ids)
    nll_values = [None] * len(chosen_triplets_ids)  # NLL from training
    loss_values = [None] * len(chosen_triplets_ids)
    # Get rows for chosen triplets
    train_df = df[df['triplet'].isin(chosen_triplets_ids)].copy()
    # Get rows for withheld triplets
    withheld_df = df_triplet_means[~df_triplet_means.index.isin(chosen_triplets_ids)].copy().drop(columns=config["output_labels"])

    # Active learning loop
    iteration = 0
    
    print(f"\n  Starting active learning loop with {len(withheld_df)} candidates...")
    while len(withheld_df) > 0 and best_id not in chosen_triplets_ids:
        print(f"  Run {run_idx}, Iteration {iteration}: Evaluating acquisition function...")
        # Evaluate acquisition function
        best_ei, best_triplet, metrics = gde.step_within_data(train_df, withheld_df, return_metrics=True)
        best_triplet = withheld_df.loc[best_triplet]
        #This line ensures that we append the new triplet data to train_df, not replacing it
        train_df = df[df['triplet'] == best_triplet.name]
        withheld_df = withheld_df.drop(index=best_triplet.name)
        
        expected_improvements.append(best_ei if best_ei is None else float(best_ei))
        nll_values.append(metrics.get('nll', None))
        loss_values.append(metrics.get('loss', None))
        chosen_triplets_ids.append(int(best_triplet.name))

        current_best = df_triplet_means.loc[chosen_triplets_ids][config["property_name"]].max().item()
        bests.append(current_best)
        if best_ei is not None:
            print(f"    ✓ Selected triplet {int(best_triplet.name)}, Best FE: {current_best:.4f}, EI: {best_ei:.6f}, Remaining candidates: {len(withheld_df)}")
        else:
            print(f"    ✓ Selected triplet {int(best_triplet.name)}, Best FE: {current_best:.4f}, EI: None, Remaining candidates: {len(withheld_df)}")
        
        iteration += 1
        
    print(f"  ✓ Run {run_idx} completed in {iteration} iterations.\n")
    # Save results
    results_df = pd.DataFrame({
        'chosen_triplets': chosen_triplets_ids,
        'expected_improvements': expected_improvements,
        'nll': nll_values,
        'loss': loss_values,
    })
    results_df.to_csv(run_dir / 'chosen_triplets.csv')
    yaml.dump(config, open(run_dir / 'config.yaml', 'w'))
    print(f"  Results saved to {run_dir / 'chosen_triplets.csv'}")
    
    return run_dir

def process_runs_mean(model_name: str):
    """Process all runs for a given model and return aggregated DataFrame."""
    print(f"  Processing results for {model_name}...")
    all_df = []
    run_count = 0
    
    for run_dir in OUTPUT_BASE.glob(model_name + '_run_*/'):
        if not run_dir.is_dir():
            continue
        
        results_file = run_dir / 'chosen_triplets.csv'
        if not results_file.exists():
            continue
        
        # Read the results

        df_triplet_means = df.groupby('triplet').mean()
        chosen_df = pd.read_csv(results_file, index_col=0)

        empty_df = pd.DataFrame(index=np.arange(2,len(df_triplet_means)), columns=chosen_df.columns)

        empty_df['cummax FE'] = df_triplet_means.loc[:,config["property_name"]].max()
        
        # Calculate cummax FE for this run
        chosen_df['cummax FE'] = df_triplet_means.loc[
            chosen_df['chosen_triplets'], config["property_name"]
        ].cummax().values

        chosen_df = chosen_df.combine_first(empty_df)
        
        # Add step column (offset by 2 to match old convention)
        i0 = 2
        chosen_df['step'] = chosen_df.index - i0
        chosen_df['dname'] = run_dir.stem
        chosen_df['model'] = model_name
        
        all_df.append(chosen_df)
        run_count += 1
    
    print(f"    ✓ Loaded {run_count} runs for {model_name}")
    return pd.concat(all_df, axis=0) if all_df else pd.DataFrame()

def create_random_baseline(n_runs, property_name):
    """Create baseline runs"""
    
    all_df = []
    run_count = 0

    df_triplet_means = df.groupby('triplet').mean()
    
    for run in range(n_runs):

        current_run = pd.DataFrame(index=np.arange(len(df_triplet_means)))

        current_run["chosen_triplets"] = np.random.permutation(np.arange(len(df_triplet_means)))
        
        # Calculate cummax FE for this run
        current_run['cummax FE'] = df_triplet_means.loc[
            current_run['chosen_triplets'], property_name
        ].cummax().values
        
        # Add step column (offset by 2 to match old convention)
        i0 = 2
        current_run['step'] = current_run.index - i0
        current_run['dname'] = run
        current_run['model'] = "baseline"

        current_run = current_run[current_run['step'] >= 0]
        
        all_df.append(current_run)
    
    print(f"    ✓ Created {n_runs} baseline runs")
    return pd.concat(all_df, axis=0)

# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run active learning experiments')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('config', type=str, help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.load(f, Loader=yaml.FullLoader)
    base_configs = raw if isinstance(raw, list) else [raw]

    for base_config in base_configs:
        combo_configs = list(expand_config(base_config))
        print("=" * 70)
        print(f"BASE CONFIG: {base_config['run_name']}  ({len(combo_configs)} combination(s))")
        print("=" * 70)

        # Load data once — all combos in a base_config share the same dataset
        dataset = base_config.get("dataset", "gas")
        data_file = base_config.get("data_file", None)
        if data_file is not None:
            p = Path(data_file)
            if not p.is_absolute():
                p = Path(__file__).resolve().parent / p
            data_file = p
        if dataset == "bicarb":
            df = load_bicarb_data(filepath=data_file)
            output_labels = ["FE_CO", "CO2 utilization"]
        elif dataset == "gas":
            df, _cd = load_gas_data(file=data_file)
            output_labels = ["FE (Eth)", "FE (CO)"]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        exclude_cols = {"triplet"} | set(output_labels)
        input_labels = [c for c in df.columns if c not in exclude_cols and df[c].nunique() > 1]
        df = df[df.loc[:, output_labels].notna().all(axis=1)]
        df_triplet_means = df.groupby('triplet').mean()
        best_id = int(df_triplet_means[base_config["property_name"]].idxmax())
        print(f"  Loaded {len(df)} data points  |  inputs: {input_labels}")

        baseline_df = create_random_baseline(base_config["num_runs"] * 100, base_config["property_name"])
        baseline_df["combo"] = "baseline"
        combo_palette, combo_markers = build_combo_style(combo_configs)
        combo_dfs = []

        for config in combo_configs:
            print(f"\n{'─'*70}")
            print(f"  COMBO: {config['run_name']}")
            print(f"{'─'*70}")

            OUTPUT_BASE = Path(config["run_name"])
            OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
            k_step = config.get('k_step', 9)

            if dataset == "gas":
                config["current_density"] = _cd
            config["input_labels"] = input_labels
            config["output_labels"] = output_labels

            # ── run experiments ──────────────────────────────────────────────
            if not config["use_existing_results"]:
                torch.manual_seed(config["torch_seed"])
                model = config["models"]
                run_indices = list(config.get("runs", range(config["num_runs"])))
                for i, run_idx in enumerate(run_indices):
                    print(f"\n  RUN {i+1}/{len(run_indices)}: {model.upper()} (seed {run_idx})")
                    run_active_learning_experiment(model, run_idx, config)
                print(f"\n  ✓ All runs done for {config['run_name']}")
            else:
                print("  Using existing results.")

            # ── collect results (always, for stats and combined plot) ─────────
            _df_runs = process_runs_mean(config["models"])
            _df_runs = _df_runs[_df_runs['step'] >= 0].reset_index(drop=True)
            _df_runs["combo"] = config["run_name"]
            combo_dfs.append(_df_runs)

            if args.no_plot:
                continue

            # ── per-combo plots ──────────────────────────────────────────────
            name = config["run_name"]
            _df_runs.to_csv(OUTPUT_BASE / 'cummax_FE.csv')
            plt.figure(figsize=(5, 5))
            sns.lineplot(data=_df_runs, x='step', y='cummax FE',
                         hue='dname', legend=False)
            plt.axvline(x=k_step, color='gray', linestyle='--')
            plt.ylabel('Cumulative max of ' + config["property_name"])
            plt.xlabel('Step #')
            plt.tight_layout()
            plt.savefig(OUTPUT_BASE / 'cummax_FE.svg', bbox_inches='tight')
            plt.close()
            print(f"  ✓ {OUTPUT_BASE / 'cummax_FE.svg'}")

            for metric, ylabel, title in [
                ('nll',  'NLL',           'NLL vs step'),
                ('loss', 'Training loss', 'Loss vs step'),
            ]:
                if metric not in _df_runs.columns:
                    continue
                _df_runs[['step', 'dname', metric]].to_csv(OUTPUT_BASE / f'{metric}_over_steps.csv')
                plt.figure(figsize=(5, 4))
                sns.lineplot(data=_df_runs, x='step', y=metric, errorbar='sd')
                plt.ylabel(ylabel)
                plt.xlabel('Step')
                plt.title(title)
                plt.tight_layout()
                plt.savefig(OUTPUT_BASE / f'{metric}_over_steps.svg', bbox_inches='tight')
                plt.close()

        # ── summary statistics (once, over all combos + baseline) ────────────
        if not combo_dfs:
            continue

        k_step = base_config.get('k_step', 9)
        config = combo_configs[-1]  # any combo config; property_name is shared
        all_combos_df = pd.concat(combo_dfs + [baseline_df], ignore_index=True)
        combo_order = [c["run_name"] for c in combo_configs] + ["baseline"]

        print(f"\n{'='*70}\nSUMMARY STATISTICS\n{'='*70}")
        stats = {}
        for combo_name in combo_order:
            df_combo = all_combos_df[all_combos_df["combo"] == combo_name]
            steps_to_finish, final_nlls, val_at_kstep = [], [], []
            for run in df_combo['dname'].unique():
                chosen_df = df_combo[df_combo['dname'] == run]
                try:
                    final_nlls.append(chosen_df['nll'].dropna().iat[-1])
                except Exception:
                    pass
                filtered = chosen_df[chosen_df['cummax FE'] >= df_triplet_means.loc[best_id, config["property_name"]] - 1e-8]
                if not filtered.empty:
                    steps_to_finish.append(filtered['step'].iloc[0])
                val_at_kstep.append(chosen_df[chosen_df['step'] == k_step]['cummax FE'])
            if steps_to_finish:
                sf = np.clip(np.array(steps_to_finish), 0, None)
                accel = ((len(df_triplet_means) + 1) / 2 - 3) / sf.mean() if sf.mean() > 0 else np.inf
                stats[combo_name] = {"Mean Steps": sf.mean(), "Std Steps": sf.std(),
                                     "Acceleration Factor": accel,
                                     "Val at step k": np.mean(val_at_kstep),
                                     "Val at step k std": np.std(val_at_kstep)}
                print(f"  {combo_name}: {sf.mean():.1f}±{sf.std():.1f} steps  accel={accel:.2f}x  "
                      f"FE@{k_step}={np.mean(val_at_kstep):.4f}")

        base_out = Path(base_config["run_name"])
        base_out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(stats).T.to_csv(base_out / 'summary_statistics.csv')

        # ── combined comparison plot ──────────────────────────────────────────
        if args.no_plot or len(combo_dfs) < 2:
            continue

        all_combos_df.to_csv(base_out / 'comparison.csv')
        plt.figure(figsize=(6, 5))
        sns.lineplot(data=all_combos_df, x='step', y='cummax FE',
                     hue='combo', style='combo',
                     palette=combo_palette, markers=combo_markers,
                     hue_order=combo_order, dashes=False, markersize=5)
        plt.axvline(x=k_step, color='gray', linestyle='--')
        plt.ylabel('Cumulative max of ' + base_config["property_name"])
        plt.xlabel('Step #')
        plt.legend(title='Config', loc='lower right', fontsize=8)
        plt.tight_layout()
        plt.savefig(base_out / 'comparison.svg', bbox_inches='tight')
        plt.close()
        print(f"\n  ✓ Combined plot: {base_out / 'comparison.svg'}")
