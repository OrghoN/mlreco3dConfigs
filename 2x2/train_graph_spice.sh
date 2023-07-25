#!/bin/bash 

#SBATCH --account=neutrino
#SBATCH --partition=ampere

#SBATCH --job-name=train_graph_spice
#SBATCH --output=batch_outputs/output-train_graph_spice.txt 
#SBATCH --error=batch_outputs/output-train_graph_spice.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4g
#SBATCH --time=3-00:00:00 
#SBATCH --gpus a100:1

singularity exec --bind $APP_DIR,/sdf/data/neutrino/kterao/dunend_train_prod/ --nv /sdf/group/neutrino/images/develop.sif bash -c "python3 $MLRECO3D_PATH/bin/run.py graph_spice.cfg"
