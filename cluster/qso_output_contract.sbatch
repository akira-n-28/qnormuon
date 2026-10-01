#!/bin/bash
#SBATCH --job-name=qso-output-contract
#SBATCH --partition=mat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=cluster/qso_output_contract-%j.log
#SBATCH --error=cluster/qso_output_contract-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -u cluster/qso_output_contract.py
"$QSO_PYTHON" -m pytest -q --basetemp="/home/prignano/qnormuon-runs/output-contract-tests-${SLURM_JOB_ID}" --junitxml="cluster/output-contract-tests-${SLURM_JOB_ID}.xml"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
