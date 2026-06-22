"""
Assess NLL and Pearson R on held-out triplets as the training set grows.

Mirrors run_experiments.py: same YAML config format, same combo expansion,
and the same random triplet selection per run_idx across all combos.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import torch
import yaml
import argparse
from glob import iglob

from carbondriver import GDEOptimizer
from carbondriver.loaders import load_gas_data, load_bicarb_data

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import (
    choose_base_inds_numpy, expand_config, build_combo_style, DO_NOT_EXPAND_KEYS,
)

DO_NOT_EXPAND_KEYS = DO_NOT_EXPAND_KEYS | {"acquisition", "UCB_beta", "EI_reference"}

N_EVAL = 10  # held-out triplets for evaluation
N_INIT = 3   # initial training triplets


def assess_run(model_name: str, run_idx: int, config: dict,
               eval_triplet_ids: list, train_order: list) -> pd.DataFrame:
    """Train progressively and evaluate per-label NLL + Pearson R on eval set.

    eval_triplet_ids : fixed N_EVAL held-out triplet IDs (individual measurements)
    train_order      : ordered triplet IDs added one at a time; first N_INIT are init
    """
    df_eval  = df[df['triplet'].isin(eval_triplet_ids)]
    run_dir  = OUTPUT_BASE / f'{model_name}_run_{run_idx:03d}'
    run_dir.mkdir(exist_ok=True, parents=True)

    records = []
    cfg     = {**config, "make_plots": False}
    max_n   = config.get("max_train", len(train_order))

    for n in range(N_INIT, min(max_n, len(train_order)) + 1):
        train_df = df[df['triplet'].isin(train_order[:n])]

        gde = GDEOptimizer(
            model_name=model_name,
            quantity=config["property_name"],
            maximize=True,
            output_dir=str(run_dir),
            config=cfg,
            input_labels=config["input_labels"],
            output_labels=config["output_labels"],
        )
        gde.update_data(train_df)
        predictor, _ = gde.get_predictor()

        X_eval, y_eval = gde._get_data_tensors(data=df_eval)

        with torch.no_grad():
            posterior = predictor.posterior(X_eval.unsqueeze(1))
            # botorch posteriors have shape (..., q=1, n_out); squeeze q dim
            mu  = posterior.mean.squeeze(-2)      # (n_eval_pts, n_out)
            var = posterior.variance.squeeze(-2)  # (n_eval_pts, n_out)

        record = {"n_train": n, "run_idx": run_idx, "model": model_name}
        for i, label in enumerate(config["output_labels"]):
            r, _ = pearsonr(y_eval[:, i].cpu().numpy(), mu[:, i].cpu().numpy())
            record[f"corr_{label}"] = r
            nll_i = -torch.distributions.Normal(mu[:, i], var[:, i].sqrt()).log_prob(y_eval[:, i]).mean().item()
            record[f"nll_{label}"] = nll_i

        records.append(record)
        print("  n={:3d}  {}".format(n, "  ".join(
            f"{k}={v:.3f}" for k, v in record.items()
            if k.startswith(("nll_", "corr_"))
        )))

    result = pd.DataFrame(records)
    result.to_csv(run_dir / 'assessment.csv', index=False)
    return result


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Assess model NLL and Pearson R vs training set size'
    )
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('config', type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.load(f, Loader=yaml.FullLoader)
    base_configs = raw if isinstance(raw, list) else [raw]

    for base_config in base_configs:
        combo_configs = list(expand_config(base_config, do_not_expand=DO_NOT_EXPAND_KEYS))
        print("=" * 70)
        print(f"BASE CONFIG: {base_config['run_name']}  ({len(combo_configs)} combo(s))")
        print("=" * 70)

        # Load data once — all combos share the same dataset
        dataset   = base_config.get("dataset", "gas")
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
        exclude_cols     = {"triplet"} | set(output_labels)
        input_labels     = [c for c in df.columns if c not in exclude_cols and df[c].nunique() > 1]
        df               = df[df.loc[:, output_labels].notna().all(axis=1)]
        df_triplet_means = df.groupby('triplet').mean()
        all_triplet_ids  = df_triplet_means.index.tolist()
        print(f"  Loaded {len(df)} pts  |  {len(all_triplet_ids)} triplets  |  inputs: {input_labels}")

        combo_dfs = []

        for config in combo_configs:
            print(f"\n{'─'*70}\n  COMBO: {config['run_name']}\n{'─'*70}")

            OUTPUT_BASE = Path(config["run_name"] + "_perf")
            OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

            if dataset == "gas":
                config["current_density"] = _cd
            config["input_labels"]  = input_labels
            config["output_labels"] = output_labels

            if not config["use_existing_results"]:
                torch.manual_seed(config["torch_seed"])
                model       = config["models"]
                run_indices = list(config.get("runs", range(config["num_runs"])))
                all_run_dfs = []

                for run_idx in run_indices:
                    print(f"\n  RUN {run_idx+1}/{len(run_indices)}: {model.upper()} (seed {run_idx})")

                    # triplet selection — seed = run_idx only → identical across combos
                    rng      = np.random.default_rng(run_idx)
                    eval_ids = rng.choice(all_triplet_ids, size=N_EVAL, replace=False).tolist()
                    remaining = [t for t in all_triplet_ids if t not in eval_ids]

                    init_pos  = choose_base_inds_numpy(
                        df_triplet_means.loc[remaining, config["property_name"]].values,
                        num_choose=N_INIT, strategy='uniform', seed=run_idx,
                    )
                    init_ids      = [remaining[i] for i in init_pos]
                    rest          = [t for t in remaining if t not in init_ids]
                    rest_shuffled = np.random.default_rng(run_idx).permutation(rest).tolist()
                    train_order   = init_ids + rest_shuffled

                    try:
                        run_df = assess_run(model, run_idx, config, eval_ids, train_order)
                        all_run_dfs.append(run_df)
                    except Exception as e:
                        print(f"  Error in run {run_idx}: {e}")

                result_df = pd.concat(all_run_dfs, ignore_index=True)
                result_df.to_csv(OUTPUT_BASE / 'assessment.csv', index=False)
                yaml.dump(config, (OUTPUT_BASE / 'config.yaml').open('w'))
                
                result_df["combo"] = config["run_name"]
                combo_dfs.append(result_df)
                
            else:
                print("  Using existing results.")

        combo_configs = []
        for dirname in iglob(base_config["run_name"] + "/*"):
            
            OUTPUT_BASE = Path(dirname)

            if not OUTPUT_BASE.is_dir() or not (OUTPUT_BASE / 'assessment.csv').exists():
                continue
                
            result_df = pd.read_csv(OUTPUT_BASE / 'assessment.csv')

            result_df["combo"] = dirname
            combo_dfs.append(result_df)

            combo_configs.append({"models": result_df["model"].iloc[0], "run_name": dirname})

        combo_palette, combo_markers = build_combo_style(combo_configs)
        
        if args.no_plot or not combo_dfs:
            continue

        # ── 4-panel combined plot (2 metrics × 2 output labels) ──────────────
        base_out    = Path(base_config["run_name"])
        base_out.mkdir(parents=True, exist_ok=True)
        config      = base_config
        all_df      = pd.concat(combo_dfs, ignore_index=True)
        combo_order = [c["run_name"] for c in combo_configs]
        metrics     = [("nll", "NLL"), ("corr", "Pearson R")]

        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
        for col, (metric_key, metric_label) in enumerate(metrics):
            for row, label in enumerate(output_labels):
                ax       = axes[row, col]
                col_name = f"{metric_key}_{label}"
                safe     = label.replace(' ', '_').replace('/', '_')

                all_df[["n_train", "run_idx", "combo", col_name]].to_csv(
                    base_out / f"{metric_key}_{safe}.csv", index=False
                )
                sns.lineplot(data=all_df, x='n_train', y=col_name,
                             hue='combo', style='combo',
                             hue_order=combo_order, palette=combo_palette, markers=combo_markers,
                             dashes=False, markersize=4, ax=ax)
                ax.set_title(f"{metric_label} — {label}")
                ax.set_xlabel("# training triplets")
                ax.set_ylabel(metric_label)
                ax.set_ylim(top=0 if metric_key == "nll" else 1, bottom= 0 if metric_key == "corr" else -1)
                if row == 0 and col == 0:
                    ax.legend(title='Config', fontsize=7, loc='best')
                elif ax.get_legend():
                    ax.get_legend().remove()

        fig.tight_layout()
        fig.savefig(base_out / 'assessment.svg', bbox_inches='tight')
        plt.close()

        all_df.to_csv(base_out / 'assessment_all.csv', index=False)

        means = all_df.loc[:, ['n_train', 'corr_FE_CO', 'nll_FE_CO', 'corr_CO2 utilization', 'nll_CO2 utilization', 'combo']].groupby(["combo", "n_train"]).mean()

        means.to_csv(base_out / 'assessment_means.csv')
        
        print(f"\n  ✓ {base_out / 'assessment.svg'}")
