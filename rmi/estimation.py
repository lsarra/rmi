# estimation.py

# Copyright (C) 2021

# Code by Leopoldo Sarra and Florian Marquardt
# Max Planck Institute for the Science of Light, Erlangen, Germany
# http://www.mpl.mpg.de


# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# If you find this code useful in your work, please cite our article
# "Renormalized Mutual Information for Artificial Scientific Discovery", Leopoldo Sarra, Andrea Aiello, Florian Marquardt, arXiv:2005.01912

# available on

# https://arxiv.org/abs/2005.01912

# ------------------------------------------

'''
This package allows to easily estimate renormalized mutual information between a given set of Samples and a chosen feature (i.e. function of the Samples).
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp
import scipy.special


def produce_P(samples, n_bins=180):
    """Estimate probability density of given dataset

    Please NOTE: since a histogram is used, only low-dimensional distributions can be considered

    Args:
        samples (array_like): [N_samples, dim]
        n_bins (int, optional): Number of bins of the histogram. Defaults to 180.

    Returns:
        histogram (array_like): [n_bins]**dim; probability distribution
        bin_spacing (list): [dim] list of the size of the bins in each dimension
    """
    Prob_y, y_edges = np.histogramdd(samples, bins=n_bins, density=True)
    dy = np.mean(np.gradient(np.array(y_edges), axis=-1), -1)
    return Prob_y, dy


def Entropy(Px, dx):
    """Entropy of the given distribution

    It calculates the Entropy of a given probability distribution. The result is multiplied by the product of the given list dx (volume element)

    Args:
        Px (array_like): [n_bins]**dim  histogram of the probability density
        dx (list): [dim] size of a bin in each dimension

    Returns:
        (float): calculated entropy
    """
    return -np.sum(sp.special.xlogy(Px, Px))*np.prod(dx)


def RegTerm(grad_feature):
    """Calculate Renormalizing term

    It calculates the renormalizing term of renormalized mutual information (second term of RMI),
    -1/2 <log det gradF. gradF>

    Args:
        grad_feature (): [N_samples, N_y, N_x]

    Returns:
        (float): result
    """
    return - 1/2 * np.mean(FeatureGrad_LogDet(grad_feature))


def FeatureGrad_LogDet(grad_feature):
    """Part of the RegTerm inside the integral
    It calculates the logarithm of the determinant of the matrix [N_y x N_y] given by the scalar product of the gradients along the N_x axis.

    Args:
        grad_feature (array_like): [N_samples, N_y, N_x], where N_x is the input space and N_y the feature space.

    Returns:
        (array_like): [N_samples]
    """

    # Case of 1d feature
    if len(grad_feature.shape) == 2:
        grad_feature = grad_feature[:, np.newaxis, :]

    matrix_j = grad_feature@grad_feature.swapaxes(1, -1)
    s, d = np.linalg.slogdet(matrix_j)
    # return s*d
    # We remove terms with zero s (i.e. errors)
    return s[s != 0]*d[s != 0]

def RenormalizedMutualInformation(feature, grad_feature, n_bins=180):
    """Calculate Renormalized Mutual Information

    It calculates Renormalized Mutual Information (as the sum of Entropy of the feature and Renormalized Term). 
    To increase the accuracy, you can change the n_bins option of the histogram used to estimate the Entropy
    It is not necessary to give the original Samples.

    Please NOTE: since a histogram is used to calculate the entropy, only low dimensional features can be
    considered at the moment

    Args:
        feature (array_like): [N_samples, N_y]
        grad_feature (array_like): [N_samples, N_y, N_x]
        n_bins (int, optional): number of bins. Defaults to 180.

    Returns:
        (float): result
    """
    FirstTerm = RegTerm(grad_feature)

    Prob_y, dy = produce_P(feature, n_bins)
    SecondTerm = Entropy(Prob_y, dy)
    return FirstTerm+SecondTerm


def MutualInformation_2d(feature, n_bins=180):
    """(usual) Mutual Information 

    It calculates the usual mutual information I(y1, y2) of a given two-dimensional system. 
    This can be for example a two-dimensional feature of a larger system.
    To increase the accuracy, you can increase the n_bins option of the histogram used to estimate the Entropy
    This quantity is useful to estimate the inequality I(x, y) >= I(x, y1) + I(x, y2) - Ireal(y1, y2)

    Args:
        feature (array_like): [N_samples, 2]
        n_bins (int, optional): Number of bins to calculate the entropies. Defaults to 180.

    Returns:
        (float): result
    """
    y2, dy2 = produce_P(feature)

    y21, dy21 = produce_P(feature[:, 0], n_bins)
    y22, dy22 = produce_P(feature[:, 1], n_bins)

    Hy2 = Entropy(y2, dy2)
    Hy21 = Entropy(y21, dy21)
    Hy22 = Entropy(y22, dy22)

    return Hy21 + Hy22 - Hy2


def print_feature_batch(samples, feature_list, label_list, is_printing=True):
    """Batch calculation of RMI

    It calculates renormalized mutual information for all the feature functions in the feature list.
    If is_printing, it also prins a table with the results and the given labels in label_list
    It returns the list of the results: for each feature, Mutual Information, Entropy and Regularizing Term.

    Args:
        samples (array_like): [N_samples, N_x]
        feature_list (list): [N_features] list of functions (samples)=>(feature,grad_feature)
        label_list (list): [N_features] labels for the features
        is_printing (bool, optional): Print the values. Defaults to True.

    Returns:
        outputs (list): [N_features, 4] list of results, containing [RMI, H, Fterm, Label] 
    """

    if is_printing:
        print("{} MI       H     MI-H".format("Feature".ljust(20)))

    outputs = []
    for f, l in zip(feature_list, label_list):
        feature = f(samples)
        mi = RenormalizedMutualInformation(*feature)
        h = Entropy(*produce_P(feature[0]))
        outputs.append([mi, h, mi-h, l])
        if is_printing:
            print("{} {:+1.2f}  {:+1.2f}  {:+1.2f}".format(l.ljust(20)
                                                           [:20], *outputs[-1][0:-1]))
    return outputs


def plot_feature_batch(x_linspace, outputs_array, ylims=None, fig="par", save=False, cols=None, markers=None):
    """Plot the behavior of the RMI of some features when changing some parameter 

    Plots the RMI and H of various given features in different regimes for comparison

    Args:
        x_linspace (array_like): [N] values of the changing parameter
        outputs_array (array_like): [N, N_features, 4] containing [RMI, H, Fterm, Label] associated to each x in x_linspace for each feature
        ylims (list, optional): [2] plot range (for the first and second plot). Defaults to None.
        fig (str, optional): name of the plot. Defaults to "par".
        save (bool, optional): Whether to save the plot. Defaults to False.
        cols (list, optional): [N_features] Color to use for each data serie. Defaults to None.
        markers (list, optional): [N_features] marker to use for each data serie. Defaults to None.
    """
    prop_cycle = plt.rcParams['axes.prop_cycle']
    def_cols = prop_cycle.by_key()['color']

    if cols is None:
        cols = def_cols

    if markers is None:
        markers = ["o", "v", "d", "^", "s", "*", "x"]

    if save:
        font = {'family': 'DejaVu Serif',
                'weight': 'regular',
                'size': 22}
        mpl.rc('font', **font)
        # markersize = 8
        linewidth = 2
        plt.figure(figsize=[5, 5], dpi=300)
    else:
        font = {'family': 'DejaVu Sans',
                'weight': 'regular',
                'size': 12}
        mpl.rc('font', **font)
        markersize = 2
        linewidth = 2
        plt.figure(figsize=[10, 5], dpi=300)
        plt.subplot(1, 2, 1)
        plt.title("Renormalized Mutual Information")
        plt.xlabel(fig)
        plt.ylabel("RMI")

    if np.size(ylims) != 1:
        plt.ylim(ylims[0])

    for j in range(outputs_array.shape[1]):
        lw = linewidth
        if cols[j] == def_cols[3]:
            lw = 4
        plt.plot(x_linspace,
                 outputs_array[:, j, 0].astype(np.float),
                 label=outputs_array[0, j, -1],
                 color=cols[j],
                 marker=markers[j],
                 linewidth=lw,
                 markersize=0)  # markersize)

    if save:
        plt.savefig(fig+"_MI.pdf")

        plt.figure(figsize=[3, 3], dpi=500)
        plt.axes(frameon=False)
        plt.xticks([0.2, 0.6], ["0.2", "0.6"])
        plt.yticks([-5, 0, 5], ["-5", "0", "5"])
    else:
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title("Feature Entropy")
        plt.xlabel(fig)
        plt.ylabel("H(y)")
    if np.size(ylims) != 1:
        plt.ylim(ylims[1])

    for j in range(outputs_array.shape[1]):
        lw = linewidth
        if cols[j] == def_cols[3]:
            lw = 4
        plt.plot(x_linspace,
                 outputs_array[:, j, 1].astype(np.float),
                 label=outputs_array[0, j, -1],
                 color=cols[j],
                 marker=markers[j],
                 linewidth=lw,
                 markersize=0)
    if save:
        plt.savefig(fig+"_H.pdf")
    plt.show()

    mpl.style.use('default')


def H_gaussian_2d_theoretical(mu, sigma):
    """Entropy of a 2d Gaussian distribution

    Args:
        mu (array_like): [2] mean vector
        sigma (array_like): [2x2] covariance matrix

    Returns:
        (float): entropy
    """
    return 0.5*np.log(np.linalg.det(2*np.pi*np.exp(1)*sigma))


def H_uniform_2d_theoretical(a, b):
    """Entropy of a 2d uniform distribution

    Args:
        a (float): min
        b (float): max

    Returns:
        (float): entropy
    """
    return np.sum(np.log(b-a))


def H_gaussian_1d_theoretical(mu, sigma):
    """Entropy of a 1d Gaussian distribution

    Args:
        mu (float): mean
        sigma (float): variance

    Returns:
        (float): entropy
    """
    return np.log(sigma*np.sqrt(2*np.pi*np.exp(1)))


def H_uniform_1d_theoretical(a, b):
    """Entropy of a 1d uniform distribution

    Args:
        a (float): min
        b (float): max

    Returns:
        (float): theoretical entropy
    """
    return np.log(b-a)
