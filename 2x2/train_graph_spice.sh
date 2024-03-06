#!/bin/bash 

#SBATCH --account=neutrino:dune-ml
#SBATCH --partition=ampere

#SBATCH --job-name=train_graph_spice
#SBATCH --output=batch_outputs/output-train_graph_spice.txt 
#SBATCH --error=batch_outputs/output-train_graph_spice.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4g
#SBATCH --time=3-00:00:00 
#SBATCH --gpus a100:1

#singularity exec --bind /sdf/ --nv /sdf/group/neutrino/images/larcv2_ub20.04-cuda11.6-pytorch1.13-larndsim.sif bash -c "python3 /sdf/data/neutrino/software/lartpc_mlreco3d/bin/run.py graph_spice.cfg"
singularity exec --bind /sdf/ --nv /sdf/group/neutrino/images/larcv2_ub20.04-cuda11.6-pytorch1.13-larndsim.sif bash -c "python3 /sdf/data/neutrino/software/lartpc_mlreco3d/bin/run.py graph_spice.cfg"

