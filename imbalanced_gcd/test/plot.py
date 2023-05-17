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
