#!/bin/bash
#SBATCH --job-name=lightning     # create a short name for your job
#SBATCH --qos=preempt
#SBATCH --nodes=2               # node count
#SBATCH --ntasks-per-node=1      # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
##SBATCH --time=00:02:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --constraint=gmem48
##SBATCH --mail-type=begin        # send email when job begins
##SBATCH --mail-type=end          # send email when job ends


source ~/anaconda3/etc/profile.d/conda.sh
conda activate bolts-env


# 4. Diagnostic: Check if Torch is visible
echo "Using Python from: $(which python)"
# python -c "import torch; print(f'Success! Torch version: {torch.__version__} | CUDA: {torch.cuda.is_available()}')"

# echo "Allocated GPUs on each node:"
srun nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv

srun python myscript.py
