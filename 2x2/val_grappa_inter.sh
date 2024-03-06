#!/bin/bash 

#SBATCH --account=neutrino:dune-ml
#SBATCH --partition=ampere

#SBATCH --job-name=val_grappa_inter
#SBATCH --output=batch_outputs/output-val_grappa_inter.txt 
#SBATCH --error=batch_outputs/output-val_grappa_inter.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4g
#SBATCH --time=36:00:00 
#SBATCH --gpus a100:1

singularity exec --bind $APP_DIR,/sdf/data/neutrino/kterao/dunend_train_prod/ --nv /sdf/group/neutrino/images/larcv2_ub20.04-cuda11.6-pytorch1.13-larndsim.sif bash -c "python3 $MLRECO3D_PATH/bin/run.py grappa_inter_val.cfg"
