import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def plot_con_matrix(con_matrix):
    disp = metrics.ConfusionMatrixDisplay(con_matrix)
    disp = disp.plot(cmap="Blues", values_format=".0f")
    return disp.figure_


def plot_gcd_ci(all, normal, novel):
    """Plot confidence interval for GCD results
    Args:
        all (tuple): (mean, ci_low, ci_high) for all results
        normal (tuple): (mean, ci_low, ci_high) for normal results
        novel (tuple): (mean, ci_low, ci_high) for novel results
    Returns:
        plt.Figure: Figure with confidence interval plot
    """
    fig, ax = plt.subplots()
    x = np.arange(3)
    means = np.array([all[0], normal[0], novel[0]])
    ci_low = np.array([all[1], normal[1], novel[1]])
    ci_high = np.array([all[2], normal[2], novel[2]])
    ax.bar(x, means, yerr=[means - ci_low, ci_high - means], capsize=5, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["All", "Normal", "Novel"])
    ax.set_ylabel("Accuracy")
    return fig


def plot_auroc(data, model_names=['SSKMeans', 'GMM', 'Baseline']):
    """
    plot the multi-class AUROC graph

    args:
        data: numpy array, shape (num_params, num_models, num_groups, class_split)
        model_names: list of strings, length num_models
    """
    _, num_models, num_groups, _ = data.shape
    colors = ['red', 'blue', 'green']

    # plot the AUROC chart for each group
    # use subplot to plot multiple charts in one figure
    fig, ax = plt.subplots(1, num_groups, figsize=(num_groups * 6, 5.5))
    for i in range(num_groups):
        # line plot for each model
        for j in range(num_models):
            # overall AUROC
            ax[i].plot(data[:, j, i, 0], label=f'{model_names[j]} overall', color=colors[j])
            # normal AUROC
            ax[i].plot(data[:, j, i, 1], color=colors[j],
                       linestyle='--', label=f'{model_names[j]} normal')
            # novel AUROC
            ax[i].plot(data[:, j, i, 2], color=colors[j],
                       linestyle=':', label=f'{model_names[j]} novel')
        ax[i].set_ylabel('Multiclass ROC AUC', fontsize=14)
        ax[i].set_ylim(0, 1)
        ax[i].tick_params(labelsize=14)
    ax[num_groups // 2].legend(bbox_to_anchor=(-1, 1.1, 3., .102), loc=10,
                               ncol=num_models, mode="expand", borderaxespad=0.)
    plt.tight_layout()
    # return the figure and axes
    return fig, ax
