"""
Submit SLURM jobs for active learning experiments.

Creates and submits a separate job for each model in the hard-coded list.
Each job runs ``run_experiment.py --model <model> <config>`` so that models
execute in parallel on the cluster.  After all jobs finish, re-run with
``--plot-only`` to regenerate the combined plots.

Usage:
    python submit_experiment.py [config_yaml]

    # default config: run_config.yaml
    python submit_experiment.py
    # custom config:
    python submit_experiment.py my_config.yaml
"""
import sys
from pathlib import Path
import subprocess
import textwrap

MODELS = ["MLP", "GP", "GP+Ph"]
NUM_RUNS = 50

WALLTIME = "16:00:00"
NTASKS = 8
MEM = "16GB"
ACCOUNT = "def-peslherb"

SCRIPT_NAME = "run_experiment.py"
DEFAULT_CONFIG = "run_config.yaml"


def create_slurm_script(model: str, config_path: str, num_runs: int) -> str:
    job_name = f"al_{model}"
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

    python {SCRIPT_NAME} --model {model} --no-plot --num-runs {num_runs} {config_path}
    """)


def submit_all(dst_dir: Path, config_path: str, num_runs: int) -> None:
    for model in MODELS:
        script_path = dst_dir / f"submit_{model}.sh"
        script_text = create_slurm_script(model, config_path, num_runs)
        script_path.write_text(script_text)
        script_path.chmod(0o755)

        try:
            res = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True, text=True, cwd=str(dst_dir),
            )
        except FileNotFoundError as e:
            print(f"✗ sbatch not found: {e}")
            return
        except Exception as e:
            print(f"✗ error while submitting {model}: {e}")
            continue

        if res.returncode == 0:
            print(f"✓ Submitted {model}: {res.stdout.strip()}")
        else:
            print(f"✗ Failed to submit {model}: {res.stderr.strip() or res.stdout.strip()}")

    print(f"\nAfter all jobs finish, regenerate plots with:")
    print(f"  python {SCRIPT_NAME} --plot-only --num-runs {num_runs} {config_path}")


if __name__ == "__main__":
    repo_dir = Path(__file__).parent
    config = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    print(f"Submitting one SLURM job per model ({WALLTIME}, {NTASKS} tasks, {MEM}, {NUM_RUNS} runs)...")
    print(f"Config: {config}")
    submit_all(repo_dir, config, NUM_RUNS)
    print("Done.")
