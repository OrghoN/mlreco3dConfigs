#!/bin/bash 

#SBATCH --account=neutrino
#SBATCH --partition=ampere

#SBATCH --job-name=grappa_metrics_plot
#SBATCH --output=batch_outputs/output-grappa_metrics_plots.txt 
#SBATCH --error=batch_outputs/output-grappa_metrics_plots.txt 

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4g
#SBATCH --time=00:05:00 
#SBATCH --gpus a100:1

singularity exec --bind $APP_DIR,/sdf/data/neutrino/kterao/dunend_train_prod/ --nv /sdf/group/neutrino/images/develop.sif bash -c "python3 metrics_plotter.py metrics_names/grappa_inter_metrics_plot.json metrics_names/grappa_shower_metrics_plot.json metrics_names/grappa_track_metrics_plot.json metrics_names/grappa_inter_metrics_plot_zoom.json metrics_names/grappa_shower_metrics_plot_zoom.json metrics_names/grappa_track_metrics_plot_zoom.json "
