#!/bin/bash
#SBATCH --job-name=krishna
#SBATCH --partition=gpu
#SBATCH --qos=preempt
#SBATCH --partition=preempt    
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH -C gmem48  # Removed the '=' sign

### Communication Settings
export MASTER_PORT=12349
export WORLD_SIZE=2
# Get the hostname of the first node
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "MASTER_ADDR="$MASTER_ADDR
echo "WORLD_SIZE="$WORLD_SIZE

### Load Environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dev

### Execution
# srun will pick up the SLURM env vars automatically. 
# Inside your Python code, 'dist.init_process_group' will use 
# the env variables we exported above.
srun python main.py --lr 1e-3 --epochs 50000