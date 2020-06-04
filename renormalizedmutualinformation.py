'''
renormalizedmutualinformation.py

Copyright (C) 2020

Code by Leopoldo Sarra and Florian Marquardt
Max Planck Institute for the Science of Light, Erlangen, Germany
http://www.mpl.mpg.de

This work is licensed under the Creative Commons Attribution 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

If you find this code useful in your work, please cite our article
"Renormalized Mutual Information for Artificial Scientific Discovery", Leopoldo Sarra, Andrea Aiello, Florian Marquardt, arXiv:2005.01912

available on

https://arxiv.org/abs/2005.01912

------------------------------------------

This package allows to easily estimate renormalized mutual information between a given set of Samples and a chosen feature (i.e. function of the Samples).

- The class Information implements the calculation of required quantities;

Usage:
    - from renormalizedmutualinformation import *
    - inf = Information()
'''

import numpy as np
import matplotlib.pyplot as plt


class Information:
    '''
    This class implements Renormalized Mutual Information. The following functions are provided:
    - produce_P(samples, n_bins=180)
    - Entropy(Px, dx)
    - RegTerm(grad_feature)
    - FeatureGrad_LogDet(grad_feature)
    - MutualInformation(feature, grad_feature, n_bins=180)
    - MutualInformation_real2d(feature, n_bins=180)
    - print_feature_batch(samples, feature_list, label_list, is_printing=True)
    '''

    # Quantity added to logarithms and denominators to avoid numerical divergence
    epsilon = 1e-30

    def produce_P(self, samples, n_bins=180):
        '''
        Estimates the probability distribution density of a given set of data by making a histogram. It is reasonable to use it only in a low-dimensional setting.
        - Samples should be given with the shape [N_samples, dimension]
        - to increase the accuracy, you can increase the n_bins option of the histogram

        This function returns the probability distribution Prob_y and a list of the size of the bins in each dimension, dy
        '''
        Prob_y, y_edges = np.histogramdd(samples, bins=n_bins, density=True)
        dy = np.mean(np.gradient(np.array(y_edges), axis=-1), -1)
        return Prob_y, dy

    def Entropy(self, Px, dx):
        '''
        It calculates the Entropy of a given probability distribution. The result is multiplied by the product of the given list dx (volume element)
        '''
        return -np.sum(Px*np.log(Px+self.epsilon))*np.prod(dx)

    def RegTerm(self, grad_feature):
        '''
        It calculates the renormalized term of renormalized mutual information.
        - grad_feature should have shape [N_samples, N_y, N_x], where N_x is the input space and N_y the feature space. If N_y = 1, it is allowed to give the gradient with shape [N_samples, N_x]
        '''
        return - 1/2 * np.mean(self.FeatureGrad_LogDet(grad_feature))

    def FeatureGrad_LogDet(self, grad_feature):
        '''
        It calculates the logarithm of the determinant of the matrix [N_y x N_y] given by the scalar product of the gradients along the N_x axis.
        - grad_feature should have shape [N_samples, N_y, N_x], where N_x is the input space and N_y the feature space. If N_y = 1, it is allowed to give the gradient with shape [N_samples, N_x]
        It returns the result, an array with size N_samples
        '''

        # Case of 1d feature
        if len(grad_feature.shape) == 2:
            grad_feature = grad_feature[:, np.newaxis, :]

        matrix_j = grad_feature@grad_feature.swapaxes(1, -1)
        return np.log(np.linalg.det(matrix_j))

    def MutualInformation(self, feature, grad_feature, n_bins=180):
        '''
        It calculates Renormalized Mutual Information (as the sum of Entropy of the feature and Renormalized Term). It is necessary to input:
        - feature: the value of the feature with shape [N_samples, N_y]
        - grad_feature: the gradient of the feature with respect to inputs, [N_samples, N_y, N_x]
        - to increase the accuracy, you can increase the n_bins option of the histogram used to estimate the Entropy
        It is not necessary to give the original Samples.
        '''
        FirstTerm = self.RegTerm(grad_feature)

        Prob_y, dy = self.produce_P(feature, n_bins)
        SecondTerm = self.Entropy(Prob_y, dy)
        return FirstTerm+SecondTerm

    def MutualInformation_real2d(self, feature, n_bins=180):
        '''
        It calculates the usual mutual information I(y1, y2) of a given two-dimensional system. This can be for example a two-dimensional feature of a larger system.
        - feature should have shape[N_samples, N_y]
        - to increase the accuracy, you can increase the n_bins option of the histogram used to estimate the Entropy
        This quantity is useful to estimate the inequality I(x, y) >= I(x, y1) + I(x, y2) - Ireal(y1, y2)
        '''

        y2, dy2 = self.produce_P(feature)

        y21, dy21 = self.produce_P(feature[:, 0], n_bins)
        y22, dy22 = self.produce_P(feature[:, 1], n_bins)

        Hy2 = inf.Entropy(y2, dy2)
        Hy21 = inf.Entropy(y21, dy21)
        Hy22 = inf.Entropy(y22, dy22)

        return Hy21 + Hy22 - Hy2

    def print_feature_batch(self, samples, feature_list, label_list, is_printing=True):
        '''
        It calculates renormalized mutual information for all the feature functions in the feature list.
        If is_printing, it also prins a table with the results and the given labels in label_list
        It returns the list of the results: for each feature, Mutual Information, Entropy and Regularizing Term.
        '''
        if is_printing:
            print("{} MI       H     MI-H".format("Feature".ljust(20)))

        outputs = []
        for f, l in zip(feature_list, label_list):
            feature = f(samples)
            mi = self.MutualInformation(*feature)
            h = self.Entropy(*self.produce_P(feature[0]))
            outputs.append([mi, h, mi-h, l])
            if is_printing:
                print("{} {:+1.2f}  {:+1.2f}  {:+1.2f}".format(l.ljust(20)
                                                               [:20], *outputs[-1][0:-1]))
        return outputs

