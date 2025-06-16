#!/home-nfs/lwtucker/lwtucker_venv/bin/
#SBATCH --job-name=basic_lwtucker_gpu
#SBATCH --output=basic_lwtucker_%j.log
#SBATCH --partition=gpu
#SBATCH --gpus=nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=4
python train_attack.py
