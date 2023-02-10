import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import pickle
from pathlib import Path






def set_seed(seed_number):
    '''Set random seeds for replicability'''
    random.seed(seed_number)
    np.random.seed(seed_number)

    return ()


def dump_pickle_file(file, file_name, dir_path="./outputs/pickle"):
    file_path = Path(dir_path + "/" + file_name)
    f = open(file_path, 'wb')
    pickle.dump(file, f)

    return ()


def load_pickle_file(file_name, dir_path="./outputs/pickle"):
    file_path = Path(dir_path + "/" + file_name)
    f = open(file_path, 'rb')
    file = pickle.load(f)
    f.close()

    return (file)


def plot_bah_error(results_dict, col_names=None, y_axis_lim=None, x_axis_label="n"):
    df_bah_hinge = pd.DataFrame.from_dict(results_dict["hsvm_bah_errors"])
    df_bah_smooth = pd.DataFrame.from_dict(results_dict["ssvm_bah_errors"])

    if col_names is not None:
        df_bah_hinge.columns = col_names
        df_bah_smooth.columns = col_names
    else:
        pass

    fig, axs = plt.subplots(1, 2, figsize=(12, 3.5))

    sns.set_theme(style="whitegrid")
    sns.set_palette("pastel")
    palette = sns.color_palette("pastel")

    sns.boxplot(data=df_bah_hinge, ax=axs[0], color=palette[0], saturation=0.8,
                linewidth=1, fliersize=2)
    sns.boxplot(data=df_bah_smooth, ax=axs[1], color=palette[1], saturation=0.8,
                linewidth=1, fliersize=2)

    for ax in axs:
        ax.set_ylim(y_axis_lim)
        ax.set_xlabel(x_axis_label)

    axs[0].set_ylabel(r'$|| r ||_{2}$')

    axs[0].set_title("Hinge loss SVM")
    axs[1].set_title("Smooth SVM")

    fig.subplots_adjust(top=0.8)
    fig.suptitle("L2 norm of Bahadur remainder");

    return (fig)