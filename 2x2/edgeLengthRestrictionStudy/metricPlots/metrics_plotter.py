import json
import sys
from multiprocessing import Pool

sys.path.append("lartpc_mlreco3d/")
from mlreco.visualization.training import draw_training_curves

def draw_single_plot(plot):
    draw_training_curves(plot["log_dir"],
                         models=plot["models"],
                         metrics=plot["metrics"],
                         limits=plot["limits"],
                         print_min=plot["print_min"],
                         print_max=plot["print_max"],
                         interactive=False,
                         same_plot=True,
                         train_prefix="train",
                         val_prefix="inference",
                         figure_name=(plot["plot_dir"]+plot["file_name"]))

if __name__ == '__main__':
    file_list = sys.argv[1:]

    for f in file_list:
        print(f)
        with open(f) as file:
            plots = json.load(file)

            with Pool() as pool:
                pool.map(draw_single_plot, plots)

        print("--------------------")
