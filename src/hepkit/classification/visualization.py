from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hist import Hist

from ..histograms import multi_hist1d_comparison
from .evaluation import calculate_roc_curve


def plot_correlations(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    xlabels: Sequence[str] | None = None,
    fig_size: tuple[float, float] = (5, 5),
    **kwargs: Any,
) -> None:
    """
    Plot a correlation matrix heatmap for the given DataFrame.
    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - columns (list, optional): The columns to include in the correlation matrix. If None, all columns are included. Default is None.
    - transform (callable, optional): A function to transform the data before calculating the correlation matrix. Default is None.
    - xlabels (list, optional): The labels for the x-axis. If None, the column names are used. Default is None.
    - fig_size (tuple, optional): The size of the figure. Default is (5, 5).
    - **kwargs: Additional keyword arguments to be passed to the correlation calculation.
    Returns:
    - None
    """

    data = df.copy()
    if columns is not None:
        data = data[columns]

    if transform is not None:
        data = transform(data)

    corrmat = data.corr(**kwargs)
    ax1 = plt.subplots(ncols=1, figsize=fig_size)[1]
    opts = {"cmap": plt.get_cmap("RdBu"), "vmin": -1, "vmax": +1}
    heatmap1 = ax1.pcolor(corrmat.values, **opts)
    plt.colorbar(heatmap1, ax=ax1)
    ax1.set_title("Correlations")

    if xlabels is None:
        xlabels = list(corrmat.columns)
    for ax in (ax1,):
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_yticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=90, ha="right")
        ax.set_xticklabels(xlabels, minor=False, rotation=90)
        ax.tick_params(axis="x", labelrotation=90)
        # remove gridlines
        ax.grid(False)

    # save_fig(fig_id)
    return None


def plot_signal_background_comparison(
    signal_hists: dict[str, Hist],
    background_hists: dict[str, Hist],
    signal_label: str = "Signal",
    background_label: str = "Background",
    histtypes: list[str] | None = None,
    colors: list[str] | None = None,
    **kwargs,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Quick comparison between signal and background histograms.

    Args:
        signal_hists: Dictionary of signal histograms
        background_hists: Dictionary of background histograms
        signal_label: Label for signal histograms (default: "Signal")
        background_label: Label for background histograms (default: "Background")
        histtypes: List of histogram types (default: ["step", "stepfilled"])
        colors: List of colors (default: ["red", "blue"])
        **kwargs: Additional arguments passed to multi_hist1d_comparison
    """
    if histtypes is None:
        histtypes = ["step", "step"]
    if colors is None:
        colors = ["red", "blue"]

    return multi_hist1d_comparison(
        hists=[signal_hists, background_hists],
        legends=[signal_label, background_label],
        histtypes=histtypes,
        colors=colors,
        **kwargs,
    )


def plot_train_test_response(clf, X_train, y_train, X_test, y_test, bins=30, log_y=True):
    """
    Compare the classifier response on training and testing data.
    Parameters:
    - clf: The classifier object.
    - X_train: The training data features.
    - y_train: The training data labels.
    - X_test: The testing data features.
    - y_test: The testing data labels.
    - bins: The number of bins for the histogram (default: 30).
    Returns:
    None
    """
    decisions = []
    for X, y in ((X_train, y_train), (X_test, y_test)):
        d1 = clf.predict_proba(X[y > 0.5])[:, 1].ravel()
        d2 = clf.predict_proba(X[y < 0.5])[:, 1].ravel()
        decisions += [d1, d2]

    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low, high)

    plt.figure()
    plt.hist(
        decisions[0],
        color="r",
        alpha=0.5,
        range=low_high,
        bins=bins,
        histtype="stepfilled",
        density=True,
        label="Signal (train)",
    )
    plt.hist(
        decisions[1],
        color="b",
        alpha=0.5,
        range=low_high,
        bins=bins,
        histtype="stepfilled",
        density=True,
        label="Background (train)",
    )

    hist, bins = np.histogram(decisions[2], bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt="o", c="r", label="Signal (test)")

    hist, bins = np.histogram(decisions[3], bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt="o", c="b", label="Background (test)")

    plt.xlabel("BDT output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc="best")
    plt.grid()

    # show in log y axis
    if log_y:
        plt.yscale("log")


def plot_roc_auc(y_true, y_scores, label=None):
    """Plot ROC curve with AUC."""
    fpr, tpr, _, auc = calculate_roc_curve(y_true, y_scores)

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.3f})" if label is None else label)
    plt.plot([0, 1], [0, 1], "k--", lw=1)  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.grid()


def plot_roc_auc_comparison(y_true_list, y_scores_list, labels):
    """Compare multiple ROC curves."""
    plt.figure()
    for y_true, y_scores, label in zip(y_true_list, y_scores_list, labels, strict=False):
        fpr, tpr, _, auc = calculate_roc_curve(y_true, y_scores)
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid()


def plot_signal_efficiency_vs_background_rejection(y_true, y_scores):
    """Plot signal efficiency vs background rejection (HEP style ROC)."""
    fpr, tpr, _, auc = calculate_roc_curve(y_true, y_scores)
    background_rejection = 1 - fpr
    signal_efficiency = tpr

    plt.figure()
    plt.plot(signal_efficiency, background_rejection, lw=2, label=f"AUC = {auc:.3f}")
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Rejection")
    plt.title("Signal Efficiency vs Background Rejection")
    plt.legend(loc="best")
    plt.grid()
