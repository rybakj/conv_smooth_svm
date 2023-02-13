import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import pickle
from pathlib import Path
import os






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


def plot_l2_error(results_dict, col_names=None, y_axis_lim=None, x_axis_label="n",
                  plot_svm=False, drop_cols = []):
    '''
    Inputs:
    - results_dict: dict with keys "hsvm_errors", "ssvm_errors" and "svc_errors"
    - col_names: list
    - y_axis_lim: list with upper and lower bound as elements (e.g. [0,1])
    - x_axis_label: string

    Returns:
    - plot object

    '''

    # Turn dictionary into dataframes
    df_svm_lp = pd.DataFrame.from_dict(results_dict["hsvm_errors"])
    df_svm_smooth = pd.DataFrame.from_dict(results_dict["ssvm_errors"])
    df_svm = pd.DataFrame.from_dict(results_dict["svc_errors"])

    df_svm_lp.drop(drop_cols, axis = 1, inplace = True)
    df_svm_smooth.drop(drop_cols, axis=1, inplace=True)
    df_svm.drop(drop_cols, axis=1, inplace=True)


    # name columns if required (col. names constitute x-axis ticks)
    if col_names is not None:
        df_svm_lp.columns = col_names
        df_svm_smooth.columns = col_names
        df_svm.columns = col_names
    else:
        pass

    # Plot
    if plot_svm == True:
        width = 3
    else:
        width = 2

    fig, axs = plt.subplots(1, width, figsize=(12, 3.5))

    sns.set_theme(style="whitegrid")
    sns.set_palette("pastel")
    palette = sns.color_palette("pastel")

    sns.boxplot(data=df_svm_lp, ax=axs[0], color=palette[0], saturation=0.8,
                linewidth=1, fliersize=2, showfliers = False)
    sns.boxplot(data=df_svm_smooth, ax=axs[1], color=palette[1], saturation=0.8,
                linewidth=1, fliersize=2, showfliers = False)

    if plot_svm == True:
        sns.boxplot(data=df_svm, ax=axs[2], color=palette[2], saturation=0.8,
                    linewidth=1, fliersize=2)

    for ax in axs:
        ax.set_ylim(y_axis_lim)
        ax.set_xlabel(x_axis_label)

    axs[0].set_ylabel(r'$ \left\|| \hat{\theta} - \theta^{*} \right\||_{2} $')

    axs[0].set_title("SVM")
    axs[1].set_title("Smooth SVM")

    if plot_svm == True:
        axs[2].set_title("SVM")

    return (fig)


def plot_bah_error(results_dict, col_names=None, y_axis_lim=None, x_axis_label="n", drop_cols = []):

    df_bah_hinge = pd.DataFrame.from_dict(results_dict["hsvm_bah_errors"])
    df_bah_smooth = pd.DataFrame.from_dict(results_dict["ssvm_bah_errors"])

    df_bah_hinge.drop(drop_cols, axis = 1, inplace = True)
    df_bah_smooth.drop(drop_cols, axis=1, inplace=True)


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
                linewidth=1, fliersize=2, showfliers = False)
    sns.boxplot(data=df_bah_smooth, ax=axs[1], color=palette[1], saturation=0.8,
                linewidth=1, fliersize=2, showfliers = False)

    for ax in axs:
        ax.set_ylim(y_axis_lim)
        ax.set_xlabel(x_axis_label)
        ax.grid(axis='y', color = "grey", alpha = 0.3)

    axs[0].set_ylabel(" L2 norm of Bah. remainder ")

    axs[0].set_title("SVM")
    axs[1].set_title("Smooth SVM")

    fig.subplots_adjust(top=0.8)
#     fig.suptitle("L2 norm of Bahadur remainder");

    return (fig)


def df_save(df, file_name):

    dir_path = './outputs/data'
    # check directory exists (if not, create it)
    isdir = os.path.isdir(dir_path)

    if isdir == False:
        os.mkdir(dir_path)
        print("Directory created.")
    else:
        pass

    # save file to the directory

    file_path = os.path.join(dir_path, file_name)

    df.to_csv(file_path)

    return ()


############## ROBUSTNESS ###############

from sklearn.linear_model import LinearRegression

angle_coef = 1


# c = np.array([1,1]) # psvm.coefs
# c = c/np.linalg.norm(c, ord = 2)
# c = c/2

# lpm_coef = c

# p = 2
# n = 100
# Sigma = np.eye(p) # /p # np.array([[1, 0], [0, 1]])
# mu1 = np.zeros(p)/10 #/np.sqrt(p) # np.array([1, 1])


def generate_x_lpm(n, p, mu1, Sigma, lpm_coef):
    '''
    Generate data set of features, which is compatible with
    the specified LPM coefficient.
    '''

    X1 = np.random.multivariate_normal(mu1, Sigma, n)

    probs = 1 / 2 * (1 + (X1 @ lpm_coef + 0))

    X1 = X1[(probs > 0).squeeze() & (probs < 1).squeeze(), :]

    while X1.shape[0] < n:
        print(X1.shape[0])
        X1_new = np.random.multivariate_normal(mu1, Sigma, n)
        probs = 1 / 2 * (1 + (X1_new @ lpm_coef + 0))
        X1_new = X1_new[(probs > 0).squeeze() & (probs < 1).squeeze(), :]

        X1 = np.concatenate([X1, X1_new])

    X1 = X1[:n, :]

    return (X1)


def get_logreg_coef(lpm_coef, alpha, reverse=False):
    vector1 = np.array([1, -1])  # psvm.coefs
    vector1 = vector1 / np.linalg.norm(vector1, ord=2)
    vector1 = vector1 / 2

    if reverse == False:
        vector2 = np.array([1, 1])  # psvm.coefs
    if reverse == True:
        vector2 = -1 * np.array([1, 1])

    vector2 = vector2 / np.linalg.norm(vector2, ord=2)
    vector2 = vector2 / 2

    logreg_coef = alpha * vector1 + (1 - alpha) * vector2
    logreg_coef = logreg_coef / np.sqrt(alpha ** 2 + (1 - alpha) ** 2)
    # logreg_coef = angle_coef * lpm_coef

    cosine = (logreg_coef.T @ lpm_coef) / (np.linalg.norm(logreg_coef) * np.linalg.norm(lpm_coef))

    return (logreg_coef, cosine)


def get_probs(X, lpm_coef, logreg_coef):
    '''
    For a given feature dataset, and given LPM and logistic regression coefficient,
    return the cond. probablities of Y = 1.
    '''


    probs_lpm = 1 / 2 * (1 + (X @ lpm_coef + 0))
    probs_logreg = 1 / (1 + np.exp(- (X @ logreg_coef)))

    return (probs_lpm, probs_logreg)


def generate_labels(probs_lpm, probs_logreg):
    '''
    Given the probabilities P(Y = 1|X), generate binary (+/-1) labels.
    '''


    Y_lpm = np.random.binomial(n=1, p=probs_lpm)
    Y_logreg = np.random.binomial(n=1, p=probs_logreg)

    Y_lpm = 2 * Y_lpm - 1
    Y_logreg = 2 * Y_logreg - 1

    return (Y_lpm, Y_logreg)


def estimate_models(X, Y):
    '''
    Estimate OLS and Smooth SVM models.
    '''


    n = X.shape[0]
    p = X.shape[1]

    ssvm = SmoothSVM(h=(p / n) ** (1 / 4))
    ssvm.fit(X=X, Y=Y)
    ssvm_coefs = ssvm.coefs  # disregard intercept

    ols = LinearRegression(fit_intercept=True)
    ols.fit(X=X, y=Y)
    ols_coefs = ols.coef_

    svc_model = SVC(kernel='linear', random_state=32, C=1e4, tol=1e-4)
    svc_model.fit(X, Y)

    svc_coefs = svc_model.coef_

    return (ssvm_coefs, ols_coefs, svc_coefs)

# def estimate_ols_svc(X, Y):

#   ols = LinearRegression(fit_intercept = True)
#   ols.fit(X = X, y = Y)
#   ols_coefs = ols.coef_


#   svc_model = SVC(kernel='linear', random_state=32, C=1e4, tol = 1e-4)
#   svc_model.fit(X, Y)

#   svc_coefs = svc_model.coef_

#   return(svc_coefs, ols_coefs)

