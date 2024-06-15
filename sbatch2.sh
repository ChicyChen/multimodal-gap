#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=multimodal2
#SBATCH --nodes=1
#SBATCH --mem=48GB
#SBATCH --time=02-00:00:00
#SBATCH --account=qingqu2
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
module purge
module load cuda/11.8.0 cudnn/11.8-v8.7.0
eval "$(conda shell.bash hook)"
conda activate mindgap
cd /scratch/qingqu_root/qingqu1/siyich/multimodal-gap
python train.py --TRAINER_CONFIG_PATH utils/train_config_1e-1.yaml \
         --saved_checkpoints 3_shrink_nw_train_checkpoints_nonorm_5e-4_1e-1 --logs 3_shrink_nw_logs_nonorm_5e-4 \
         --num_train_epochs 100
