import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional


__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def create_slurm_script(
    job_name: str,
    log_dir: str,
    python_script: str,
    args: Dict,
    memory_gb: int = 64,
    time_hours: int = 24,
    cpus_per_task: int = 4,
    dependencies: Optional[str] = None,
) -> str:
    """Create SLURM job submission script."""
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}_%j.out
#SBATCH --error={log_dir}/{job_name}_%j.err
#SBATCH --time={time_hours}:00:00
#SBATCH --mem={memory_gb}G
#SBATCH --cpus-per-task={cpus_per_task}
"""
    if dependencies:
        script += f"#SBATCH --dependency={dependencies}\n"

    script += """
source ~/.bashrc
conda activate cellarr_env  # Modify as needed for your environment

python {python_script} '{json.dumps(args)}'
"""
    script_path = os.path.join(log_dir, f"{job_name}_submit.sh")
    with open(script_path, "w") as f:
        f.write(script)
    return script_path


def submit_slurm_job(script_path: str, dependency: Optional[str] = None) -> str:
    """Submit SLURM job and return job ID."""
    cmd = ["sbatch"]
    if dependency:
        cmd.extend(["--dependency", dependency])
    cmd.append(script_path)

    result = subprocess.run(cmd, capture_output=True, text=True)
    job_id = result.stdout.strip().split()[-1]
    return job_id


def main():
    parser = argparse.ArgumentParser(description="Build CellArrDataset using SLURM steps")
    parser.add_argument("--input-manifest", required=True, help="Path to JSON manifest file")
    parser.add_argument("--output-dir", required=True, help="Base output directory")
    parser.add_argument("--memory-per-job", type=int, default=64, help="Memory in GB per job")
    parser.add_argument("--time-per-job", type=int, default=24, help="Time in hours per job")
    parser.add_argument("--cpus-per-task", type=int, default=4, help="CPUs per task")
    args = parser.parse_args()

    # Create output directories
    base_dir = Path(args.output_dir)
    log_dir = base_dir / "logs"
    temp_dir = base_dir / "temp"
    final_dir = base_dir / "final"

    for d in [log_dir, temp_dir, final_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Read manifest file
    with open(args.input_manifest) as f:
        manifest = json.load(f)

    # Step 1: Gene Annotation Job
    gene_args = {
        "files": manifest["files"],
        "output_dir": str(final_dir),
        "gene_options": manifest.get("gene_options", {}),
        "temp_dir": str(temp_dir / "gene_annotation"),
    }

    gene_script = create_slurm_script(
        job_name="cellarr_gene_annot",
        log_dir=str(log_dir),
        python_script="process_gene_annotation.py",
        args=gene_args,
        memory_gb=args.memory_per_job,
        cpus_per_task=args.cpus_per_task,
    )
    gene_job_id = submit_slurm_job(gene_script)

    # Step 2: Sample Metadata Job
    sample_args = {
        "files": manifest["files"],
        "output_dir": str(final_dir),
        "sample_options": manifest.get("sample_options", {}),
        "temp_dir": str(temp_dir / "sample_metadata"),
    }

    sample_script = create_slurm_script(
        job_name="cellarr_sample_meta",
        log_dir=str(log_dir),
        python_script="process_sample_metadata.py",
        args=sample_args,
        memory_gb=args.memory_per_job,
        cpus_per_task=args.cpus_per_task,
        # dependencies=f"afterok:{gene_job_id}",
    )
    sample_job_id = submit_slurm_job(sample_script)

    # Step 3: Cell Metadata Job
    cell_args = {
        "files": manifest["files"],
        "output_dir": str(final_dir),
        "cell_options": manifest.get("cell_options", {}),
        "temp_dir": str(temp_dir / "cell_metadata"),
    }

    cell_script = create_slurm_script(
        job_name="cellarr_cell_meta",
        log_dir=str(log_dir),
        python_script="process_cell_metadata.py",
        args=cell_args,
        memory_gb=args.memory_per_job,
        cpus_per_task=args.cpus_per_task,
        # dependencies=f"afterok:{sample_job_id}",
    )
    cell_job_id = submit_slurm_job(cell_script)

    # Step 4: Matrix Jobs (one per matrix type)
    matrix_job_ids = []
    matrix_options = manifest.get("matrix_options", [{"matrix_name": "counts"}])
    if not isinstance(matrix_options, list):
        matrix_options = [matrix_options]

    for matrix_opt in matrix_options:
        matrix_args = {
            "files": manifest["files"],
            "output_dir": str(final_dir),
            "matrix_options": matrix_opt,
            "gene_annotation_file": str(temp_dir / "gene_annotation/gene_set.json"),
            "temp_dir": str(temp_dir / f"matrix_{matrix_opt['matrix_name']}"),
        }

        matrix_script = create_slurm_script(
            job_name=f"cellarr_matrix_{matrix_opt['matrix_name']}",
            log_dir=str(log_dir),
            python_script="process_matrix.py",
            args=matrix_args,
            memory_gb=args.memory_per_job * 2,  # More memory for matrix processing
            cpus_per_task=args.cpus_per_task,
            dependencies=f"afterok:{cell_job_id},{gene_job_id}",
        )
        job_id = submit_slurm_job(matrix_script)
        matrix_job_ids.append(job_id)

    # Step 5: Final Assembly Job
    final_args = {
        "input_dir": str(final_dir),
        "output_dir": str(final_dir),
        "matrix_names": [opt["matrix_name"] for opt in matrix_options],
    }

    final_script = create_slurm_script(
        job_name="cellarr_final_assembly",
        log_dir=str(log_dir),
        python_script="final_assembly.py",
        args=final_args,
        memory_gb=args.memory_per_job,
        cpus_per_task=args.cpus_per_task,
        dependencies=f"afterok:{','.join(matrix_job_ids)}",
    )
    submit_slurm_job(final_script)


if __name__ == "__main__":
    main()
