"""
External wrapper to run active learning experiments and generate plots.
This runs the test suite multiple times, collecting data at each step.
"""

from pathlib import Path
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
    gde = GDEOptimizer(
        model_name=model_name,
        aquisition="EI",
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
    withheld_df = df_triplet_means[~df_triplet_means.index.isin(chosen_triplets_ids)].copy()

    # Active learning loop
    iteration = 0
    
    print(f"\n  Starting active learning loop with {len(withheld_df)} candidates...")
    while len(withheld_df) > 0 and best_id not in chosen_triplets_ids:
        print(f"  Run {run_idx}, Iteration {iteration}: Evaluating acquisition function...")
        # Evaluate acquisition function
        best_ei, best_row_idx, metrics = gde.step_within_data(train_df, withheld_df, return_metrics=True)
        best_triplet = withheld_df.iloc[int(best_row_idx)]
        #This line ensures that we append the new triplet data to train_df, not replacing it
        train_df = df[df['triplet'] == best_triplet.name]
        withheld_df = withheld_df.drop(index=best_triplet.name)
        
        expected_improvements.append(float(best_ei))
        nll_values.append(metrics.get('nll', None))
        loss_values.append(metrics.get('loss', None))
        chosen_triplets_ids.append(int(best_triplet.name))

        current_best = df_triplet_means.loc[chosen_triplets_ids][config["property_name"]].max().item()
        bests.append(current_best)
        print(f"    ✓ Selected triplet {int(best_triplet.name)}, Best FE: {current_best:.4f}, EI: {best_ei:.6f}, Remaining candidates: {len(withheld_df)}")
        
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

def create_random_baseline(n_runs):
    """Create baseline runs"""
    
    all_df = []
    run_count = 0

    df_triplet_means = df.groupby('triplet').mean()
    
    for run in range(n_runs):

        current_run = pd.DataFrame(index=np.arange(len(df_triplet_means)))

        current_run["chosen_triplets"] = np.random.permutation(np.arange(len(df_triplet_means)))
        
        # Calculate cummax FE for this run
        current_run['cummax FE'] = df_triplet_means.loc[
            current_run['chosen_triplets'], config["property_name"]
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

    print("="*70)
    print("ACTIVE LEARNING EXPERIMENT SUITE")
    print("="*70)
    
    print("\n[1/6] Loading configuration...")
    with open(sys.argv[1]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    OUTPUT_BASE = Path(config["run_name"])
    print(f"  Output directory: {OUTPUT_BASE}")
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    k_step = config.get('k_step', 9)
    
    # Load data once
    print("[2/6] Loading experimental data...")
    dataset = config.get("dataset", "gas")
    data_file = config.get("data_file", None)
    if data_file is not None:
        p = Path(data_file)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / p
        data_file = p
    if dataset == "bicarb":
        df = load_bicarb_data(filepath=data_file)
    elif dataset == "gas":
        df = load_gas_data(file=data_file)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    print(f"  ✓ Loaded {len(df)} data points from {dataset} dataset")
    
    # Determine output labels based on dataset
    if dataset == "bicarb":
        output_labels = ["FE_CO", "CO2 utilization"]
    elif dataset == "gas":
        output_labels = ["FE (Eth)", "FE (CO)"]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Determine input labels from non-constant columns
    exclude_cols = {"triplet"} | set(output_labels)
    input_labels = [col for col in df.columns 
                    if col not in exclude_cols and df[col].nunique() > 1]
    print(f"  Input features: {input_labels}")

    df = df[df.loc[:,output_labels].notna().all(axis=1)]

    df_triplet_means = df.groupby('triplet').mean()
    best_id = int(df_triplet_means[config["property_name"]].idxmax())    

    config["input_labels"] = input_labels
    config["output_labels"] = output_labels
    
    if not config["use_existing_results"]:
        
        print("\n[3/6] Running new experiments...")
        torch.manual_seed(config["torch_seed"])
        print(f"  Set torch seed to {config['torch_seed']}")
        
        total_runs = len(config["models"]) * len(list(config.get("runs", range(config["num_runs"]))))
        run_num = 0
        
        for model in config["models"]:
            print(f"\n  → Starting experiments for {model.upper()}...")
            
            for run_idx in config.get("runs", range(config["num_runs"])):
                run_num += 1
                print(f"\n{'='*70}")
                print(f"RUN {run_num}/{total_runs}: {model.upper()} (Seed {run_idx})")
                print(f"{'='*70}")
                run_active_learning_experiment(model, run_idx, config)
                print(f"✓ Completed {model} run {run_idx+1}/{config['num_runs']}")
                
        print("\n" + "="*70)
        print("✓ All experiments completed!")
        print("="*70)
        print("\nGenerating plots...")
    if config["use_existing_results"]:
        print("\n[3/6] Using existing results (use_existing_results=True)")
    
    # ========================================================================
    # Plot 1: 2x2 grid - one subplot per model
    # ========================================================================
    print("\n[4/6] Generating 2x2 grid plot...")
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(15, 8), sharex=True, sharey=True)
    all_data = []

    combined_csv = OUTPUT_BASE / 'all_runs_combined.csv'

    if not combined_csv.exists():

        for i, model_name in enumerate(config["models"]):
            print(f"  Processing subplot for {model_name}...")
            _df = process_runs_mean(model_name)
            _df = _df[_df['step'] >= 0]
        
            all_data.append(_df)
        
        all_df = pd.concat(all_data, axis=0)
        
        all_df.to_csv(combined_csv, index=False)

    else:

        print(f"  Loading combined data from {combined_csv}...")
        all_df = pd.read_csv(combined_csv)

    baseline_df = create_random_baseline(n_runs=config["num_runs"]*100)

    all_df_baseline = pd.concat([all_df, baseline_df], axis=0)
        
    for i, model_name in enumerate(config["models"]+["baseline"]):    

        _df = all_df_baseline[all_df_baseline['model'] == model_name]
        
        sns.lineplot(
            data=_df, 
            x='step', 
            y='cummax FE', 
            hue='dname', 
            legend=False, 
            ax=ax[i // 3, i % 3]
        )
        ax[i // 3, i % 3].set_title(model_name, fontsize=12, fontweight='bold')
        ax[i // 3, i % 3].set_ylabel('Cumulative max FE (Eth)')
        ax[i // 3, i % 3].set_xlabel('Step')
        
    
    fig.tight_layout()
    fig.savefig(OUTPUT_BASE / 'cummax_FE_grid.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_BASE / 'cummax_FE_grid.png'}")
    plt.close()
    
    # ========================================================================
    # Plot 2: Combined - all models on one plot
    # ========================================================================
    print("\n[5/6] Generating combined model comparison plot...")
    
    plt.figure(figsize=(5, 5))
    sns.lineplot(
        data=all_df_baseline, 
        x='step', 
        y='cummax FE', 
        hue='model', 
        marker='o', 
        ms=5
    )
    plt.axvline(x=k_step, color='gray', linestyle='--')
    plt.ylabel('Cumulative max FE (Eth)')
    plt.xlabel('Step')
    plt.legend(title='Model', loc='lower right')
    plt.tight_layout()
    plt.savefig(OUTPUT_BASE / 'active_learning_combined.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_BASE / 'active_learning_combined.png'}")
    plt.close()
    
    # Reset index to avoid duplicate label issues in seaborn
    all_df = all_df.reset_index(drop=True)
    
    # ========================================================================
    # Plot 3: NLL over steps (training quality metric)
    # ========================================================================
    if 'nll' in all_df.columns:
        print("  Generating NLL vs step plot...")
        plt.figure(figsize=(5, 4))
        sns.lineplot(
            data=all_df[all_df['step'] >= 0], 
            x='step', 
            y='nll', 
            hue='model',
            errorbar='sd'
        )
        plt.ylabel('NLL (Negative Log-Likelihood)')
        plt.xlabel('Step')
        plt.title('Model Uncertainty (NLL) vs Active Learning Step')
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(OUTPUT_BASE / 'nll_over_steps.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {OUTPUT_BASE / 'nll_over_steps.png'}")
        plt.close()
    
    # ========================================================================
    # Plot 4: Loss (MSE) over steps (training fit quality)
    # ========================================================================
    if 'loss' in all_df.columns:
        print("  Generating Loss vs step plot...")
        plt.figure(figsize=(5, 4))
        sns.lineplot(
            data=all_df[all_df['step'] >= 0], 
            x='step', 
            y='loss', 
            hue='model',
            errorbar='sd'
        )
        plt.ylabel('Training loss')
        plt.xlabel('Step')
        plt.title('Training Loss vs Active Learning Step')
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(OUTPUT_BASE / 'loss_over_steps.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {OUTPUT_BASE / 'loss_over_steps.png'}")
        plt.close()
    
    # ========================================================================
    # Summary statistics
    # ========================================================================
    print("\n[6/6] Computing summary statistics...")
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    stats = dict()
    
    for model_name in all_df_baseline["model"].unique():
        print(f"\n{model_name.upper()}:")
        steps_to_finish = []
        final_nlls = []
        val_at_kstep = []

        df_model = all_df_baseline.loc[all_df_baseline["model"]==model_name]
        
        for run in df_model['dname'].unique():
            chosen_df = df_model[df_model['dname'] == run]
            
            # Collect final NLL value (last non-NaN).
            try:
                final_nlls.append(chosen_df['nll'].dropna().iat[-1])
            except Exception:
                # missing column or all-NaN -> skip
                pass
            
            # Find step where cummax FE becomes max
            filtered = chosen_df[chosen_df['cummax FE'] >= df_triplet_means.loc[best_id, config["property_name"]] - 1e-8]
            if not filtered.empty:
                # First step where threshold was crossed
                step_to_max = filtered['step'].iloc[0]
                # Record steps to finish
                if step_to_max == 0:
                    print(f"    Run {run} started at max FE (step 0).")
                steps_to_finish.append(step_to_max)
            else:
                print(f"    Run {run} did not reach the target FE threshold.")

            val_at_kstep.append(chosen_df[chosen_df['step'] == k_step]['cummax FE'])
            
                
        if steps_to_finish:
            sf = np.array(steps_to_finish)
            sf[sf < 0] = 0
            mean_steps = np.mean(sf)
            std_steps = np.std(sf)
            accel_factor = ((len(df_triplet_means)+1)/2-3) / mean_steps if mean_steps > 0 else np.inf
            
            print(f"  Runs analyzed: {len(df_model['dname'].unique())}")
            print(f"  Mean steps to max FE: {mean_steps:.2f} ± {std_steps:.2f}")

            stats[model_name] = {"Mean Steps": mean_steps, "Std Steps": std_steps, "Acceleration Factor": accel_factor, "Val at step k": np.mean(val_at_kstep), "Val at step k std": np.std(val_at_kstep)}

            if final_nlls:
                print(f"  Final NLL: {np.mean(final_nlls):.4f} ± {np.std(final_nlls):.4f}")
                stats[model_name]["Final NLL mean"] = np.mean(final_nlls)
                stats[model_name]["Final NLL std"] = np.std(final_nlls)
            print(f"  Acceleration factor: {accel_factor:.2f}x")
            print(f"  FE at step {k_step}: {np.mean(val_at_kstep):.4f} ± {np.std(val_at_kstep):.4f}")
            
        else:
            print(f"  No data available for {model_name}") 

    stats_df = pd.DataFrame(stats).T
    stats_df.to_csv(OUTPUT_BASE / 'summary_statistics.csv')
            
    print("\n" + "="*70)
    print("✓ Analysis complete!")
    print("="*70 + "\n")
