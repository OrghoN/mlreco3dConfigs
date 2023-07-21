#!/bin/bash 

#SBATCH --account=neutrino
#SBATCH --partition=ampere

#SBATCH --job-name=val_uresnet_ppn
#SBATCH --output=batch_outputs/output-val_uresnet_ppn.txt 
#SBATCH --error=batch_outputs/output-val_uresnet_ppn.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4g
#SBATCH --time=12:00:00 
#SBATCH --gpus a100:1

singularity exec --bind /sdf/group/neutrino/anoronyo/,/sdf/data/neutrino/kterao/dunend_train_prod/ --nv /sdf/group/neutrino/images/develop.sif bash -c "python3 lartpc_mlreco3d/bin/run.py uresnet_ppn_val.cfg"
