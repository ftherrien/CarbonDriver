"""
Submit SLURM jobs for active learning experiments.

Generates per-model YAML configs and submits one SLURM job per (experiment, model).
After all jobs finish, re-run with ``use_existing_results: True`` to regenerate plots.

Usage:
    python submit_experiment.py

The EXPERIMANTS list below defines all experiment configurations.
"""
import sys
from pathlib import Path
import subprocess
import textwrap
import yaml

MODELS = ["MLP", "GP", "GP+Ph", "Ph"]

WALLTIME = "13:00:00"
NTASKS = 8
MEM = "16GB"
ACCOUNT = "def-peslherb"

SCRIPT_NAME = "run_experiment.py"
RESULTS_DIR = Path("../cluster_results")

EXPERIMENTS = [
    {"dataset": "bicarb", "system_phase": "liquid", "EI_reference": "max", "models": ["MLP", "GP", "GP+Ph", "Ph"]},
    {"dataset": "bicarb", "system_phase": "liquid", "EI_reference": "min", "models": ["MLP", "GP", "GP+Ph", "Ph"]},
    {"dataset": "bicarb", "system_phase": "gas",    "EI_reference": "max", "models": ["GP+Ph", "Ph"]},
    {"dataset": "bicarb", "system_phase": "gas",    "EI_reference": "min", "models": ["GP+Ph", "Ph"]},
]


def experiment_tag(exp: dict) -> str:
    phase = exp["system_phase"]
    ei = exp["EI_reference"]
    return f"{phase}_liquid_EI_{ei}"


def build_config(exp: dict, model: str) -> dict:
    tag = experiment_tag(exp)
    return {
        "run_name": str(RESULTS_DIR / f"{tag}_results"),
        "dataset": exp["dataset"],
        "system_phase": exp["system_phase"],
        "num_runs": 100,
        "models": [model],
        "property_name": "FE_CO",
        "EI_reference": exp["EI_reference"],
        "num_iter": 101,
        "normalize_inputs": True,
        "normalize_outputs": False,
        "torch_seed": 0,
        "use_existing_results": False,
    }


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

    cd "$SLURM_SUBMIT_DIR"

    python {SCRIPT_NAME} --no-plot {config_path}
    """)


def submit_all(dst_dir: Path) -> None:
    total_jobs = 0
    for exp in EXPERIMENTS:
        tag = experiment_tag(exp)
        models = exp["models"]
        print(f"\n{'─'*60}")
        print(f"  Experiment: {tag}")
        print(f"  dataset={exp['dataset']}  system_phase={exp['system_phase']}  EI_reference={exp['EI_reference']}")
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

    total_jobs = sum(len(e["models"]) for e in EXPERIMENTS)
    print(f"Submitting {total_jobs} SLURM jobs across {len(EXPERIMENTS)} experiments")
    print(f"Walltime: {WALLTIME}  |  Tasks: {NTASKS}  |  Mem: {MEM}")
    print(f"Runs per model: 100")

    submit_all(repo_dir)
    print(f"\nDone. {total_jobs} jobs submitted.")
