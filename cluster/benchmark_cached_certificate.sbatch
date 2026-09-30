#!/bin/bash
#SBATCH --job-name=qso-cert-cache
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:12:00
#SBATCH --output=cluster/benchmark_cached_certificate-%j.log
#SBATCH --error=cluster/benchmark_cached_certificate-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -u cluster/benchmark_cached_certificate.py
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
