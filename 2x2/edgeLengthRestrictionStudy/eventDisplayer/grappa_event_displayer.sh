#!/bin/bash 

#SBATCH --account=neutrino
#SBATCH --partition=ampere

#SBATCH --job-name=grappa_event_display
#SBATCH --output=batch_outputs/output-grappa_event_display.txt 
#SBATCH --error=batch_outputs/output-grappa_event_display.txt 

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4g
#SBATCH --time=00:15:00 
#SBATCH --gpus a100:1

singularity exec --bind $APP_DIR,/sdf/data/neutrino/kterao/dunend_train_prod/ --nv /sdf/group/neutrino/images/develop.sif bash -c "python3 grappa_event_displayer.py"
singularity exec --bind $APP_DIR,/sdf/data/neutrino/kterao/dunend_train_prod/ --nv /sdf/group/neutrino/images/develop.sif bash -c "python3 grappa_event_displayer.py no"

