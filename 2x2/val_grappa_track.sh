#!/bin/bash 

#SBATCH --account=neutrino
#SBATCH --partition=ampere

#SBATCH --job-name=val_grappa_track
#SBATCH --output=batch_outputs/output-val_grappa_track.txt 
#SBATCH --error=batch_outputs/output-val_grappa_track.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4g
#SBATCH --time=36:00:00 
#SBATCH --gpus a100:1

singularity exec --bind /sdf/group/neutrino/anoronyo/,/sdf/data/neutrino/kterao/dunend_train_prod/ --nv /sdf/group/neutrino/images/develop.sif bash -c "python3 lartpc_mlreco3d/bin/run.py grappa_track_val.cfg"
