#!/home-nfs/lwtucker/lwtucker_venv/bin/
#SBATCH --job-name=basic_lwtucker_gpu
#SBATCH --output=basic_lwtucker_%j.log
#SBATCH --partition=gpu
#SBATCH --gpus=nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=4

# srun --pty -p gpu --gpus=nvidia_rtx_a6000:1  python train_attack.py --model_name whisper-base-multi --data_name librispeech --attack_method audio-raw --max_epochs 500 --clip_val -1 --attack_size 10240 --save_freq 10

# 
python train_attack.py
