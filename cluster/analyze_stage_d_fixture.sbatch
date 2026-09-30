#!/bin/bash
#SBATCH --job-name=qso-fixture
#SBATCH --partition=mat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=cluster/analyze_stage_d_fixture-%j.log
#SBATCH --error=cluster/analyze_stage_d_fixture-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -u cluster/analyze_stage_d_fixture.py "$@"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
