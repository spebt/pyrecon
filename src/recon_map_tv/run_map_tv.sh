#!/bin/bash
#SBATCH --job-name=map_tv_recon
#SBATCH --output=logs/map_tv_%j.out
#SBATCH --error=logs/map_tv_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

mkdir -p logs

CONFIG="${1:-configs/base_config.yml}"

echo "Running MAP-TV reconstruction with config: $CONFIG"
python map_tv_recon.py --config "$CONFIG"
