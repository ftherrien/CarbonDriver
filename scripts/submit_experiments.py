"""
Submit SLURM jobs for active learning experiments.

Generates per-model YAML configs and submits one SLURM job per (experiment, model).
After all jobs finish, re-run with ``use_existing_results: True`` to regenerate plots.

Usage:
    python submit_experiment.py

The EXPERIMENTS list below defines all experiment configurations.
"""
import sys
from pathlib import Path
import subprocess
import textwrap
import yaml

MODELS = ["MLP", "GP", "GP+Ph", "Ph"]

WALLTIME = "16:00:00"
NTASKS = 8
MEM = "16GB"
ACCOUNT = "def-peslherb"

SCRIPT_NAME = "run_experiment.py"
RESULTS_DIR = Path("cluster_results")

EXPERIMENTS = [
    # ── Bicarb data, liquid physics (4 models) ──────────────────
    {"dataset": "bicarb", "system_phase": "liquid", "acquisition": "EI", "EI_reference": "max", "property_name": "FE_CO"},
    {"dataset": "bicarb", "system_phase": "liquid", "acquisition": "EI", "EI_reference": "min", "property_name": "FE_CO"},
    {"dataset": "bicarb", "system_phase": "liquid", "acquisition": "PI", "EI_reference": "max", "property_name": "FE_CO"},
    {"dataset": "bicarb", "system_phase": "liquid", "acquisition": "UCB", "UCB_beta": 1.0, "property_name": "FE_CO"},
    {"dataset": "bicarb", "system_phase": "liquid", "acquisition": "UCB", "UCB_beta": 3.0, "property_name": "FE_CO"},
    # ── Bicarb data, gas physics (2 models) ─────────────────────
    {"dataset": "bicarb", "system_phase": "gas", "acquisition": "EI", "EI_reference": "max", "property_name": "FE_CO", "models": ["GP+Ph", "Ph"]},
    {"dataset": "bicarb", "system_phase": "gas", "acquisition": "EI", "EI_reference": "min", "property_name": "FE_CO", "models": ["GP+Ph", "Ph"]},
    {"dataset": "bicarb", "system_phase": "gas", "acquisition": "PI", "EI_reference": "max", "property_name": "FE_CO", "models": ["GP+Ph", "Ph"]},
    {"dataset": "bicarb", "system_phase": "gas", "acquisition": "UCB", "UCB_beta": 1.0, "property_name": "FE_CO", "models": ["GP+Ph", "Ph"]},
    {"dataset": "bicarb", "system_phase": "gas", "acquisition": "UCB", "UCB_beta": 3.0, "property_name": "FE_CO", "models": ["GP+Ph", "Ph"]},
]


def experiment_tag(exp: dict) -> str:
    dataset = exp["dataset"]
    phase = exp["system_phase"]
    acq = exp.get("acquisition", "EI")
    if acq == "UCB":
        beta = exp.get("UCB_beta", 1.0)
        return f"{dataset}_{phase}_{acq}_{beta}_{transfer}"
    ei_ref = exp.get("EI_reference", "max")
    return f"{dataset}_{phase}_{acq}_{ei_ref}_{transfer}"


def build_config(exp: dict, model: str) -> dict:
    tag = experiment_tag(exp)
    cfg = {
        "run_name": str(RESULTS_DIR / f"{tag}_results"),
        "dataset": exp["dataset"],
        "system_phase": exp["system_phase"],
        "num_runs": 100,
        "models": [model],
        "property_name": exp.get("property_name", "FE_CO"),
        "acquisition": exp.get("acquisition", "EI"),
        "num_iter": 101,
        "normalize_inputs": True,
        "normalize_outputs": False,
        "torch_seed": 0,
        "use_existing_results": False,
    }
    if exp.get("EI_reference"):
        cfg["EI_reference"] = exp["EI_reference"]
    if exp.get("UCB_beta"):
        cfg["UCB_beta"] = exp["UCB_beta"]
    return cfg


def create_slurm_script(job_name: str, config_path: str) -> str:
    return textwrap.dedent(f"""\
    #!/bin/bash
    #SBATCH --job-name={job_name}
    #SBATCH --output={job_name}_%j.out
    #SBATCH --error={job_name}_%j.err
    #SBATCH --account={ACCOUNT}
    #SBATCH --ntasks-per-node={NTASKS}
    #SBATCH --mem={MEM}
    #SBATCH --time={WALLTIME}

    set -euo pipefail

    cd /scratch/anizami/CO2_liq

    source ~/CO2_a/bin/activate

    python {SCRIPT_NAME} --no-plot {config_path}
    """)


def submit_all(dst_dir: Path) -> None:
    total_jobs = 0
    for exp in EXPERIMENTS:
        tag = experiment_tag(exp)
        models = exp.get("models", MODELS)
        print(f"\n{'─'*60}")
        print(f"  Experiment: {tag}")
        print(f"  dataset={exp['dataset']}  system_phase={exp['system_phase']}  acquisition={exp.get('acquisition', 'EI')}", end="")
        if exp.get("acquisition") == "UCB":
            print(f"  UCB_beta={exp.get('UCB_beta', 1.0)}")
        else:

            print(f"  EI_reference={exp['EI_reference']}")

        print(f"  models={models}")
        print(f"{'─'*60}")

        for model in models:
            config = build_config(exp, model)
            job_name = f"al_{model}_{tag}"
            config_path = dst_dir / f"_generated_{tag}_{model}.yaml"

            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            script_path = dst_dir / f"submit_{tag}_{model}.sh"
            script_text = create_slurm_script(job_name, str(config_path))
            script_path.write_text(script_text)
            script_path.chmod(0o755)

            try:
                res = subprocess.run(
                    ["sbatch", str(script_path)],
                    capture_output=True, text=True, cwd=str(dst_dir),
                )
            except FileNotFoundError as e:
                print(f"  ✗ sbatch not found: {e}")
                return
            except Exception as e:
                print(f"  ✗ error submitting {model}: {e}")
                continue

            if res.returncode == 0:
                print(f"  ✓ {model}: {res.stdout.strip()}")
            else:
                print(f"  ✗ {model}: {res.stderr.strip() or res.stdout.strip()}")


if __name__ == "__main__":
    repo_dir = Path(__file__).parent

    total_jobs = sum(len(e.get("models", MODELS)) for e in EXPERIMENTS)
    print(f"Submitting {total_jobs} SLURM jobs across {len(EXPERIMENTS)} experiments")
    print(f"Walltime: {WALLTIME}  |  Tasks: {NTASKS}  |  Mem: {MEM}")
    print(f"Runs per model: 100")

    submit_all(repo_dir)
    print(f"\nDone. {total_jobs} jobs submitted.")
