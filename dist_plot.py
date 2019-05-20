import numpy as np
import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt

def scatter_matrix(samples, show=False):
    """
    Plots a scatter matrix of samples.

    :param samples: a NxD np.array where N is the number of data points and D is the dimension.
    :return:
    """
    samples = [x/np.linalg.norm(x) for x in samples]
    D = len(samples[0])
    sns.set(style="white")
    columns = ["w" + str(i + 1) for i in range(D)]
    df = pd.DataFrame(samples, columns=columns)
    g = sns.pairplot(df, diag_kind="kde", plot_kws={"s": 5})
    ax = g.axes
    for (i, j) in zip(range(D), range(D)):
        ax[i, j].set_xlim(-1.2, 1.2)
        ax[i, j].set_ylim(-1.2, 1.2)
    if show:
        plt.show()
        plt.savefig("plots/plt" + str(time.time()) + ".png")
    return ax


