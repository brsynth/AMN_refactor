import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def plot_classification(pred, true, saving_file=None):
    roc = roc_curve(true, pred)
    auc = roc_auc_score(true, pred)
    plt.title('AUC = %.2f'% auc)

    sns.set(font='arial', palette="colorblind", style="whitegrid", font_scale=2.5, rc={'figure.figsize':(11,11)})
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.ylim((0,1))
    plt.xlim((0,1))
    plt.xticks([0.2,0.4,0.6,0.8,1.0])
    sns.lineplot(x=roc[0], y=roc[1])

    if saving_file:
        plt.savefig(saving_file, format="png", dpi=600, bbox_inches='tight')

    plt.show()
    plt.close("all")


def plot_regression(pred, true, pred_label, true_label, title, saving_file=None):
    plt.title(title)
    sns.set(font='arial', palette="colorblind", style="whitegrid", font_scale=2.5, rc={'figure.figsize':(11,11)})
    sns.regplot(x=true, y=pred, fit_reg=0, marker='+', color='black', scatter_kws={'s':40, 'linewidths':0.7})
    plt.xlabel(true_label)
    plt.ylabel(pred_label)
    
    p1 = max(max(pred), max(true))
    p2 = min(min(pred), min(true))
    
    plt.xlim(p2, p1)
    plt.ylim(p2, p1)

    plt.plot([p1, p2], [p1, p2], 'b-')

    if saving_file:
        plt.savefig(saving_file, format="png", dpi=600, bbox_inches='tight')
    plt.show()
    plt.close("all")


def plot_accuracies(accuracies, labels, y_ticks = [0.8, 0.9, 1.00], save_file=None):
    """This function plot all accuracies given as argument."""

    # sns.set_theme(style='whitegrid', font='arial', font_scale=2, palette='colorblind')
    sns.set_theme(style='whitegrid', font_scale=2, palette='colorblind')
    sns.barplot(x=labels, y=accuracies, color="grey")
    plt.yticks(y_ticks)
    plt.ylabel("Accuracy")
    plt.ylim(y_ticks[0], 1)
    if save_file:
        plt.savefig(save_file, format="png", dpi=800, bbox_inches = 'tight')
    plt.show()
    plt.close('all')
